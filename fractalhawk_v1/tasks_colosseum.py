"""
tasks_colosseum.py (v3) -- the game, inverted and governed.

WHAT THE GAME IS NOW
------------------------
v1/v2's game was a chase: Solver iteratively steers coordinates toward a
target it can only sense through a distance ping. v3 changes the game to
ADDRESS IDENTIFICATION: the Architect picks a point on the manifold, the
model's real machinery runs a real text probe under that address, and the
Solver must name the point from the BEHAVIOR alone.

Why this game: winning it requires exactly the map idea 1 asks for --
"forward every weight address to the fractal to learn where they belong."
A Solver that wins has learned the inverse of the weight generator:
behavior -> address. That inverse map is the load-bearing piece of the
whole addressed-component design -- it's what a router/aggregator needs
to know to assemble domain registers into an agent register, and it's
learned by self-play over the WHOLE box, not just the handful of points
training data happens to visit. (infer_address() below exposes it
directly: hand it text, get the address whose behavior it most resembles.)

WHAT KILLED v2, AND THE TWO FIXES
------------------------------------
Certified null: mean_error grew ~442K -> 19.9B over 8k steps,
boundary_saturation climbing -- an unbounded arms race, unsafe in a joint
total_loss. Two independent governors fix it:

  1. BOUNDED ERROR (fixes the metric): the identification error is
     per-dimension scaled by HH_RANGE, then squashed: sat = e/(1+e),
     always in [0,1). No address choice can blow up the shared loss --
     the Architect can make the game HARD, never make the loss INFINITE.

  2. DIFFICULTY GOVERNOR (fixes the curriculum -- your "model sets the
     game, harder when too easy", with a thermostat): the Architect only
     proposes addresses inside a trust region of radius r (fraction of
     the full box) around HH_DEFAULT. r is controlled by the Solver's
     recent solve rate: consistently ABOVE the band -> game too easy,
     r grows; below -> too hard, r shrinks; r clamped to [r_min, 1.0].
     The Architect's own objective is also band-seeking, not maximal:
     it's rewarded for challenges near the EDGE of the Solver's ability
     ((sat_err - target_err)^2 -> 0), not for unreachable ones. An
     arms race has no equilibrium; a thermostat does.

GRAPH DISCIPLINE (carried from v2, same reasons):
  - Only the Solver-side loss enters the shared total_loss; it pushes
    gates/ln_pre in the same minimize direction as everything else.
  - The Architect's update is fully self-contained inside loss() (own
    optimizer, own backward, on its own freshly-computed graph).
  - Probe embeddings are detached (embed gets gradient elsewhere;
    keeping it live would share graph between the two backwards).
  - Solver's own parameters ride the zero_grad()/post_backward() hooks.

USAGE (alongside a register or LM module -- it's a colosseum, not a
curriculum for the LM loss itself):

  python core_runner_v3.py \
    --modules tasks_registers:tasks=pirouette+classics_1,seq_len=256 \
              tasks_colosseum:weight=0.05,run_every=2 \
    --pirouette_path ... --classics_path ... --skip_shakespeare \
    --pos alibi --address registers --n_registers 3 --steps 30000

Watch in eval: solve_rate should hover inside [band_lo, band_hi] while
radius climbs -- that curve IS the model charting its own manifold
outward. radius stuck at r_min = the game is too hard even at its
easiest (raise weight or solver capacity); radius pinned at 1.0 with
solve_rate high = the box is mastered (a real result: the inverse map
covers the whole manifold).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from task_module import TaskModule
from core_runner_v3 import HH_DEFAULT, HH_RANGE


class ScrewGeneratorBatched(nn.Module):
    """Per-example weight generation, reusing the model's own FixedBasis.
    Unchanged math from v2."""
    def __init__(self, basis, k, n_layers):
        super().__init__()
        self.basis = basis
        self.k, self.n = k, n_layers

    def forward(self, hh_batch):
        B = hh_batch.shape[0]
        screws = []
        for L in range(self.n):
            J1, J2, Phi, E, Ksi = [hh_batch[:, i:i+1] for i in range(5)]
            ksi = torch.sigmoid(Ksi)
            log_sigma_L = J2 * torch.exp(-torch.abs(E) * L / max(self.n - 1, 1))
            delta_theta_L = ksi * (L / max(self.n - 1, 1)) * (np.pi / 16)
            phi = J1 * L
            u_L = F.normalize(torch.cos(phi).unsqueeze(-1) * self.basis.u0 +
                              torch.sin(phi).unsqueeze(-1) * self.basis.v0, dim=1)
            n_raw = (torch.cos(phi).unsqueeze(-1) * self.basis.n0 +
                     torch.sin(phi).unsqueeze(-1) * self.basis.m0)
            n_L = F.normalize(n_raw - (n_raw * u_L).sum(1, keepdim=True) * u_L, dim=1)
            th = Phi + delta_theta_L
            U_raw = torch.cos(th).unsqueeze(-1) * u_L + torch.sin(th).unsqueeze(-1) * n_L
            spin = ksi * self.basis.spin_template[L].unsqueeze(0)
            S = torch.zeros(B, self.k, self.k, device=hh_batch.device)
            idx = torch.triu_indices(self.k, self.k, offset=1)
            S[:, idx[0], idx[1]] = spin
            S[:, idx[1], idx[0]] = -spin
            I = torch.eye(self.k, device=hh_batch.device).unsqueeze(0).expand(B, -1, -1)
            C = torch.linalg.solve(I + S, I - S)
            U_L = torch.bmm(U_raw, C.transpose(1, 2))
            V_scaled = self.basis.V.unsqueeze(0) * torch.exp(log_sigma_L).unsqueeze(-1)
            screws.append(torch.bmm(V_scaled, U_L.transpose(1, 2)))
        return torch.stack(screws, dim=1)   # [B, n, d_ff, d]


class TheArchitect(nn.Module):
    """noise -> address inside the CURRENT trust region:
    hh = HH_DEFAULT + radius * tanh(net(noise)) * HH_RANGE."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU(),
                                 nn.Linear(64, 5))

    def forward(self, noise, radius, device):
        raw = torch.tanh(self.net(noise))
        return (HH_DEFAULT.to(device).unsqueeze(0)
                + radius * raw * HH_RANGE.to(device).unsqueeze(0))


class TheSolver(nn.Module):
    """pooled behavior [B, d] -> predicted address, bounded to the box."""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, 5))

    def forward(self, pooled, device):
        raw = torch.tanh(self.net(pooled))
        return (HH_DEFAULT.to(device).unsqueeze(0)
                + raw * HH_RANGE.to(device).unsqueeze(0))


def scaled_sq_err(pred, target):
    """Per-example, per-dim range-normalized squared error, meaned over
    dims. Raw (unbounded above, but O(1) for in-box points)."""
    return (((pred - target) / HH_RANGE.to(pred.device)) ** 2).mean(dim=1)


def saturate(e):
    """[0, inf) -> [0, 1). The reason v3 can't reproduce the v2 blow-up."""
    return e / (1.0 + e)


class ColosseumTask(TaskModule):
    kind = 'loss'
    name = 'colosseum'

    def __init__(self, weight=0.05, batch_size=16, probe_len=48, run_every=1,
                 lr=1e-3, radius_init=0.15, radius_min=0.05, radius_step=1.05,
                 band_lo=0.4, band_hi=0.7, success_err=0.05, target_err=0.10,
                 solve_ema=0.98):
        self.weight, self.batch_size, self.probe_len = weight, batch_size, probe_len
        self.run_every, self.lr = run_every, lr
        self.radius = radius_init
        self.radius_min, self.radius_step = radius_min, radius_step
        self.band_lo, self.band_hi = band_lo, band_hi
        self.success_err = success_err     # scaled sq err below this = 'solved'
        self.target_err = target_err       # the edge the Architect aims for (saturated units)
        self.solve_ema = solve_ema
        self.solve_rate = 0.5              # optimistic-neutral init
        self._call_count = 0

    def setup(self, model, shared_ctx):
        self.device = shared_ctx['device']
        self.arena = ScrewGeneratorBatched(model.basis, model.k, model.n).to(self.device)
        self.architect = TheArchitect().to(self.device)
        self.solver = TheSolver(model.d).to(self.device)
        self.opt_A = optim.Adam(self.architect.parameters(), lr=self.lr)
        self.opt_S = optim.Adam(self.solver.parameters(), lr=self.lr)
        self.task_names = [n for n, _ in shared_ctx['members']]
        print(f"  [colosseum] identification game: radius={self.radius} "
              f"band=[{self.band_lo},{self.band_hi}] success_err<{self.success_err}")

    # ---- machinery ----
    def _embed_probe(self, model, shared_ctx, n):
        """Real text through the model's embedding. Detached (see header);
        v3 note: model.pos is None under alibi/rope -- position lives in
        attention there, and this chain has no attention, so nothing is
        lost by its absence."""
        task = self.task_names[np.random.randint(len(self.task_names))]
        data = shared_ctx['data'][task][0]
        idx = torch.randint(len(data) - self.probe_len, (n,))
        chars = torch.stack([data[i:i + self.probe_len] for i in idx]).to(self.device)
        x = model.embed(chars)
        if model.pos is not None:
            x = x + model.pos(torch.arange(chars.shape[1], device=self.device))
        return x.detach()

    def _behavior(self, model, x_probe, hh_batch):
        """Run the probe through the REAL gated chain under per-example
        addresses; pool to a behavior summary. Touches model.gates /
        model.ln_pre -- this is where the shared backbone gets gradient."""
        weight_stack = self.arena(hh_batch)
        x = x_probe
        for L in range(model.n):
            h_up = F.gelu(model.gates[L](model.ln_pre[L](x)))
            x = x + torch.bmm(h_up, weight_stack[:, L])
        return x.mean(dim=1)   # [B, d]

    # ---- TaskModule contract ----
    def step_batch(self, model, shared_ctx):
        self._call_count += 1
        if self._call_count % self.run_every != 0:
            return None
        return {'x_probe': self._embed_probe(model, shared_ctx, self.batch_size),
                'noise': torch.randn(self.batch_size, 16, device=self.device)}

    def loss(self, model, shared_ctx, batch):
        if batch is None:
            return torch.zeros((), device=self.device), 0.0, {'skipped': True}
        x_probe, noise = batch['x_probe'], batch['noise']

        # ---- Architect's own move (self-contained graph + backward) ----
        target_live = self.architect(noise, self.radius, self.device)
        behavior_A = self._behavior(model, x_probe, target_live)
        pred_A = self.solver(behavior_A, self.device)
        sat_A = saturate(scaled_sq_err(pred_A.detach(), target_live)).mean()
        loss_A = (sat_A - self.target_err) ** 2      # band-seeking, not maximal
        self.opt_A.zero_grad(); loss_A.backward(); self.opt_A.step()

        # ---- Solver's move (enters the shared total_loss) ----
        target = target_live.detach()
        behavior = self._behavior(model, x_probe, target)   # fresh graph: gates train here
        pred = self.solver(behavior, self.device)
        raw = scaled_sq_err(pred, target)
        solver_loss = saturate(raw).mean()                  # bounded in [0, 1)

        # ---- governor bookkeeping ----
        solved = float((raw.detach() < self.success_err).float().mean())
        self.solve_rate = self.solve_ema * self.solve_rate + (1 - self.solve_ema) * solved

        return solver_loss, self.weight, {
            'sat_err': round(float(solver_loss.detach()), 4),
            'raw_err': round(float(raw.detach().mean()), 4),
            'solved_frac': round(solved, 3),
            'radius': round(self.radius, 3),
        }

    def zero_grad(self):
        self.opt_S.zero_grad()

    def post_backward(self, step):
        self.opt_S.step()

    def eval(self, model, shared_ctx):
        # the governor acts here (eval cadence = --log_interval), so the
        # difficulty curve is visible in the same log that shows solve_rate
        if self.solve_rate > self.band_hi:
            self.radius = min(1.0, self.radius * self.radius_step)
        elif self.solve_rate < self.band_lo:
            self.radius = max(self.radius_min, self.radius / self.radius_step)
        return {'solve_rate': round(self.solve_rate, 3),
                'radius': round(self.radius, 3)}

    # ---- the payoff: the learned inverse map, exposed ----
    @torch.no_grad()
    def infer_address(self, model, shared_ctx, text, hh_probe=None):
        """behavior -> address for arbitrary text: embed it, run it through
        the gated chain under a NEUTRAL probe address (HH_DEFAULT unless
        given), and ask the Solver whose behavior this most resembles.
        This is the trained 'which register does this belong to' map --
        compare its output against your registers / atlas points."""
        stoi = shared_ctx['stoi']
        ids = torch.tensor([stoi[c] for c in text[:self.probe_len] if c in stoi],
                           dtype=torch.long, device=self.device).unsqueeze(0)
        x = model.embed(ids)
        if model.pos is not None:
            x = x + model.pos(torch.arange(ids.shape[1], device=self.device))
        hh = (HH_DEFAULT.to(self.device) if hh_probe is None else hh_probe).unsqueeze(0)
        behavior = self._behavior(model, x, hh)
        return self.solver(behavior, self.device).squeeze(0)

    # ---- checkpoint continuity: the game's difficulty state must survive ----
    def state_dict(self):
        return {'radius': self.radius, 'solve_rate': self.solve_rate,
                'solver': self.solver.state_dict(),
                'architect': self.architect.state_dict()}

    def load_state(self, state):
        self.radius = state.get('radius', self.radius)
        self.solve_rate = state.get('solve_rate', self.solve_rate)
        if 'solver' in state:
            self.solver.load_state_dict(state['solver'])
        if 'architect' in state:
            self.architect.load_state_dict(state['architect'])
        print(f"  [colosseum] restored: radius={self.radius:.3f} "
              f"solve_rate={self.solve_rate:.3f}")


def build(config):
    return ColosseumTask(
        weight=float(config.get('weight', 0.05)),
        batch_size=int(config.get('batch_size', 16)),
        probe_len=int(config.get('probe_len', 48)),
        run_every=int(config.get('run_every', 1)),
        lr=float(config.get('lr', 1e-3)),
        radius_init=float(config.get('radius_init', 0.15)),
        radius_min=float(config.get('radius_min', 0.05)),
        radius_step=float(config.get('radius_step', 1.05)),
        band_lo=float(config.get('band_lo', 0.4)),
        band_hi=float(config.get('band_hi', 0.7)),
        success_err=float(config.get('success_err', 0.05)),
        target_err=float(config.get('target_err', 0.10)),
    )
