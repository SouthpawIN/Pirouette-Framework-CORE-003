"""
tasks_moe.py -- trains the DeepSeek-style MoE addressing (--address moe):
a shared register + n_routed fine-grained registers, routed PER LAYER.
Subclasses tasks_lm.LMTask (all sampling/eval_only/balancer behavior
carries over); adds the objective pieces MoE needs to not collapse.

TERMS (each maps to a named DeepSeek-MoE component):
  1. CE                -- next-char loss, through the per-layer addresses.
  2. LOAD BALANCE      -- DeepSeek's auxiliary balance loss, applied per
     layer: KL(mean routing_L || uniform) summed over layers. Without it,
     fine-grained routing collapses onto a few registers (the encoder
     collapse, one level up and per-layer). Small weight -- it's a
     regularizer, not the objective.
  3. ROUTER Z-LOSS     -- tiny penalty on router logit magnitude
     (logsumexp^2), the standard MoE trick to keep routing logits from
     drifting to extreme values that make the softmax brittle. Off by
     default (z_weight=0); turn on if routing entropy looks unstable.

There is deliberately NO identifiability term here (unlike
tasks_registers): fine-grained MoE registers aren't meant to map 1:1 to
named sources -- they're meant to partition the manifold in whatever way
minimizes loss, which is the point of learned routing. Supervised
register<->source binding is what tasks_registers is for; this module is
the unsupervised, capacity-oriented sibling. Use registers when you want
interpretable per-domain slots; use moe when you want raw efficiency and
per-layer specialization.

WHAT TO WATCH (eval):
  - per-layer routing entropy: high = balanced use, low = collapse.
  - shared_vs_routed: norm of the shared register's perturbation vs. the
    mean routed perturbation -- tells you how much work the shared
    register is doing (DeepSeek expects it to carry the "common" load).
  - layer_addresses: the n_layers x 5 address matrix for a probe -- watch
    whether different LAYERS settle at different coordinates (the whole
    point of per-layer addressing; if all rows are identical, per-layer
    routing bought nothing and single-address registers are simpler).

USAGE:
  python core_runner_v3.py \
    --modules tasks_moe:tasks=pirouette+classics_1,seq_len=256,batch=16,lb_weight=0.01 \
    --pirouette_path ... --classics_path ... --skip_shakespeare \
    --pos alibi --address moe --n_routed 8 --top_k_active 4 \
    --steps 30000 --save_to gen_v7_moe.pt
"""
import numpy as np
import torch
import torch.nn.functional as F

from tasks_lm import LMTask
from core_runner_v3 import HH_DIMS


class MoETask(LMTask):
    def __init__(self, lb_weight=0.01, z_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lb_weight = lb_weight
        self.z_weight = z_weight
        self.name += f"+moe(lb={lb_weight})"

    def setup(self, model, shared_ctx):
        super().setup(model, shared_ctx)
        if getattr(model, 'address_mode', None) != 'moe' or model.moe is None:
            raise SystemExit("tasks_moe requires --address moe")
        self.n_routed = model.moe.n_routed
        self.n_layers = model.moe.n_layers
        print(f"  [moe] {self.n_routed} routed + 1 shared register, "
              f"per-layer over {self.n_layers} layers, "
              f"top_k_active={model.moe.top_k_active or 'dense'}")

    def loss(self, model, shared_ctx, batch):
        ce_loss, weight, log = super().loss(model, shared_ctx, batch)
        all_logits = model.last_moe_logits          # [n_layers, B, R]
        if all_logits is None:
            return ce_loss, weight, log
        combined = ce_loss
        log = dict(log)

        # per-layer load balance: mean routing per layer should be ~uniform
        probs = F.softmax(all_logits, dim=-1)        # [n_layers, B, R]
        mean_route = probs.mean(dim=1)               # [n_layers, R]
        uniform = torch.full_like(mean_route, 1.0 / self.n_routed)
        lb = F.kl_div(mean_route.clamp_min(1e-9).log(), uniform,
                      reduction='batchmean')
        if self.lb_weight > 0:
            combined = combined + self.lb_weight * lb
            log['lb'] = round(float(lb.detach()), 4)

        if self.z_weight > 0:
            z = (torch.logsumexp(all_logits, dim=-1) ** 2).mean()
            combined = combined + self.z_weight * z
            log['zloss'] = round(float(z.detach()), 4)

        # routing entropy (higher = more balanced), averaged over layers
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        log['route_entropy'] = round(float(ent.detach()), 3)
        return combined, weight, log

    def eval(self, model, shared_ctx):
        out = super().eval(model, shared_ctx)
        model.eval()
        with torch.no_grad():
            shared = model.moe.shared_point()
            routed = model.moe.routed_points()
            shared_pert = float((shared - model.moe.hh_default).norm())
            routed_pert = float((routed - model.moe.hh_default).norm(dim=1).mean())
            out['shared_perturbation'] = round(shared_pert, 4)
            out['mean_routed_perturbation'] = round(routed_pert, 4)
            # per-layer address matrix for a probe from the first source
            if self.source_ids:
                val = self.sources[self.source_ids[0]][1]
                if len(val) > self.seq_len + 1:
                    x, _ = self._windows(val, 4)
                    addrs, logits = model.moe(x)      # [n_layers, 5]
                    out['layer_address_spread'] = round(
                        float(addrs.std(dim=0).mean()), 4)   # across-layer variation
                    ent = -(F.softmax(logits, -1) *
                            F.softmax(logits, -1).clamp_min(1e-9).log()).sum(-1).mean()
                    out['route_entropy'] = round(float(ent), 3)
        model.train()
        return out


def build(config):
    if 'tasks' not in config and 'root' not in config:
        raise SystemExit("tasks_moe requires tasks=<name>[+...] and/or root=<folder>")
    return MoETask(
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
        lb_weight=float(config.get('lb_weight', 0.01)),
        z_weight=float(config.get('z_weight', 0.0)),
    )
