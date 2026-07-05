"""
tasks_registers.py -- trains the addressed-component model (--address
registers): K explicit HH register points + a router that composes them.
Subclasses tasks_lm.LMTask, so all sampling/eval_only/balancer behavior
carries over; what it adds is the objective that makes registers MEAN
something.

THE OBJECTIVE (idea 2 made trainable): "loss goes down AND the address
explains the content." Three terms:

  1. CE            -- the usual next-char loss, through the mixed address.
  2. IDENTIFIABILITY -- the router's own logits must CLASSIFY the source.
     Every training batch comes from a known source, so this is free
     supervision: cross-entropy between router logits and the source's
     register label. An address that predicts well but says nothing about
     what produced it gets pushed toward one that does. This is the
     supervised scaffold; the unsupervised generalization (InfoNCE:
     identify which window among distractors produced this address) is a
     drop-in later step once the scaffold works -- cheapest test first.
  3. LOAD BALANCE  -- small KL(mean routing || uniform) so the router
     can't satisfy 1+2 by collapsing everything onto one register (the
     encoder-collapse failure, one level up).

REGISTER LABELING: named sources (tasks=a+b+c) map to registers 0,1,2...
in order. The LAST register is always the AGENT slot: never supervised,
never assigned a source. The identifiability target for labeled sources
uses label smoothing (--agent_smoothing, default 0.1) with ALL smoothed
mass placed on the agent slot -- so the router is told "this is mostly
the pirouette register, but recruiting the agent is never wrong." That
keeps the agent slot alive and lets the router learn to lean on it for
content the domain registers don't cover, without ever forcing it.
Folder sources (root=...) are unlabeled: they train CE + load balance
only, and route freely.

WHAT TO WATCH (all reported in eval):
  - register_points: the K coordinates. These ARE your per-domain
    registers + agent register; feed any of them to generate.py --hh, or
    watch their separation grow (the contrastive result suggests real
    separation is learnable once there's a reason for it -- this gives a
    stronger reason than margin repulsion: classification pressure).
  - route_<source>: mean routing distribution for held-out text of each
    source. Diagonal-dominant = registers are doing their job.
  - route_<eval_only>: routing for NEVER-TRAINED text -- the interesting
    one. Watch whether the agent slot picks it up.
  - agent_mass: overall traffic through the agent slot.

USAGE:
  python core_runner_v3.py \
    --modules tasks_registers:tasks=pirouette+classics_1,eval_only=classics_3,seq_len=256,batch=16,ident_weight=0.1 \
    --pirouette_path ... --classics_path ... --skip_shakespeare \
    --pos alibi --address registers --n_registers 3 \
    --steps 30000 --save_to gen_v6_registers.pt

n_registers should be (number of named sources) + 1 for the agent slot;
more slots than sources is fine (spares are unlabeled, like folder
sources); fewer means the overflow sources go unlabeled (warned at setup).
"""
import numpy as np
import torch
import torch.nn.functional as F

from tasks_lm import LMTask
from core_runner_v3 import HH_DIMS


class RegistersTask(LMTask):
    def __init__(self, ident_weight=0.1, lb_weight=0.01, agent_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ident_weight = ident_weight
        self.lb_weight = lb_weight
        self.agent_smoothing = agent_smoothing
        self.name += f"+registers(ident={ident_weight},lb={lb_weight})"

    def setup(self, model, shared_ctx):
        super().setup(model, shared_ctx)
        if getattr(model, 'address_mode', None) != 'registers' or model.registers is None:
            raise SystemExit("tasks_registers requires --address registers")
        self.K = model.registers.n_registers
        self.agent_idx = self.K - 1
        # named sources get labels in listed order; agent slot stays free
        labelable = self.task_list[:self.K - 1]
        self.label_of = {s: i for i, s in enumerate(labelable)}
        overflow = [s for s in self.task_list if s not in self.label_of]
        print(f"  [registers] {self.K} slots: "
              + ", ".join(f"R{i}<-{s}" for s, i in self.label_of.items())
              + f", R{self.agent_idx}=AGENT (unsupervised)")
        if overflow:
            print(f"  [registers] WARNING: {overflow} exceed available labeled slots -- "
                  f"they train unlabeled (raise --n_registers to label them)")

    def _ident_target(self, label):
        """Smoothed one-hot: 1-s on the source's register, s on the agent."""
        t = torch.zeros(self.K, device=self.device)
        t[label] = 1.0 - self.agent_smoothing
        t[self.agent_idx] += self.agent_smoothing
        return t

    def loss(self, model, shared_ctx, batch):
        ce_loss, weight, log = super().loss(model, shared_ctx, batch)
        logits = model.last_router_logits          # [B, K], stashed by resolve_hh
        if logits is None:
            return ce_loss, weight, log
        log = dict(log)
        combined = ce_loss

        probs = F.softmax(logits, dim=-1)
        mean_route = probs.mean(dim=0)              # [K]

        src = batch['source']
        if self.ident_weight > 0 and src in self.label_of:
            target = self._ident_target(self.label_of[src]).unsqueeze(0).expand_as(probs)
            ident = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            combined = combined + self.ident_weight * ident
            log['ident'] = round(float(ident.detach()), 4)

        if self.lb_weight > 0:
            uniform = torch.full_like(mean_route, 1.0 / self.K)
            lb = F.kl_div(mean_route.clamp_min(1e-9).log(), uniform,
                          reduction='sum')
            combined = combined + self.lb_weight * lb
            log['lb'] = round(float(lb.detach()), 4)

        log['route'] = [round(float(p), 3) for p in mean_route.detach()]
        return combined, weight, log

    @torch.no_grad()
    def _route_of(self, model, ids, n=8):
        x, _ = self._windows(ids, n)
        logits = model.registers.route_logits(x)
        return F.softmax(logits, dim=-1).mean(dim=0)

    def eval(self, model, shared_ctx):
        out = super().eval(model, shared_ctx)
        model.eval()
        with torch.no_grad():
            pts = model.registers.register_points()
            out['register_points'] = {
                (f"R{i}<-{s}" if i != self.agent_idx else f"R{i}=AGENT"):
                    dict(zip(HH_DIMS, [round(float(v), 4) for v in pts[i]]))
                for i in range(self.K)
                for s in [next((t for t, j in self.label_of.items() if j == i), '?')]
            }
            # pairwise register separation -- the number to watch growing
            if self.K > 1:
                d = torch.cdist(pts, pts)
                iu = torch.triu_indices(self.K, self.K, offset=1)
                out['register_min_sep'] = round(float(d[iu[0], iu[1]].min()), 4)
            routes = {}
            agent_mass = []
            for s in self.task_list:
                val = self.sources[s][1]
                if len(val) > self.seq_len + 1:
                    r = self._route_of(model, val)
                    routes[f"route_{s}"] = [round(float(p), 3) for p in r]
                    agent_mass.append(float(r[self.agent_idx]))
            for eo, (_, val) in self.eval_sources.items():
                if len(val) > self.seq_len + 1:
                    r = self._route_of(model, val)
                    routes[f"route_{eo}(eval_only)"] = [round(float(p), 3) for p in r]
                    agent_mass.append(float(r[self.agent_idx]))
            out.update(routes)
            if agent_mass:
                out['agent_mass'] = round(float(np.mean(agent_mass)), 3)
        model.train()
        return out


def build(config):
    if 'tasks' not in config and 'root' not in config:
        raise SystemExit("tasks_registers requires tasks=<name>[+...] and/or root=<folder>")
    return RegistersTask(
        tasks=str(config.get('tasks', '')),
        root=str(config['root']) if 'root' in config else None,
        weight=float(config.get('weight', 1.0)),
        seq_len=int(config.get('seq_len', 256)),
        batch=int(config.get('batch', 16)),
        eval_only=str(config.get('eval_only', '')),
        priority_temp=float(config.get('priority_temp', 1.0)),
        floor_threshold=float(config.get('floor_threshold', 1.5)),
        floor_patience=int(config.get('floor_patience', 0)),
        max_files=int(config['max_files']) if 'max_files' in config else None,
        eval_batches=int(config.get('eval_batches', 8)),
        ident_weight=float(config.get('ident_weight', 0.1)),
        lb_weight=float(config.get('lb_weight', 0.01)),
        agent_smoothing=float(config.get('agent_smoothing', 0.1)),
    )
