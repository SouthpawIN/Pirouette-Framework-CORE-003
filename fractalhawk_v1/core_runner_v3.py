"""
CORE-003 v3  ·  Testbed Core Runner -- redesigned for generation
Pirouette Framework Volume 8  ·  Keaton Smith  ·  session w/ Claude

WHAT CHANGED FROM v2 AND WHY (each change maps to a certified finding
in the consolidated checkpoint):

1. POSITIONS: learned absolute embeddings are GONE by default, replaced
   by a --pos {alibi, rope, learned} flag (default alibi).
   -> Certified finding: generation coherence collapses almost exactly at
      the training seq_len because pos rows beyond the trained window
      never receive gradient. That is a property of ABSOLUTE positions +
      fixed-offset training windows, and no address-side fix can touch
      it. ALiBi (linear attention bias, no position parameters at all)
      is the strongest known extrapolator at small scale; RoPE is the
      standard alternative. 'learned' is kept only so the old behavior
      remains reproducible as a control. This is Open Question 3 from
      the checkpoint, now runnable as a pre-registered A/B: same corpus,
      same steps, three --pos values, measure the collapse length.

2. ADDRESSING: --address {table, encoder, blend} (default encoder).
   'table' is the old design: one free HHCoordinates per registered task
   name. 'encoder' replaces the lookup with a small AddressEncoder that
   READS the input window and produces bounded HH coordinates -- the
   address is computed from content, end-to-end, every step.
   -> This dissolves three certified problem classes at once:
      a) The optimizer-registration bug class: there are no lazily-created
         parameters anymore. The encoder exists before the optimizer is
         built, period. (sync_optimizer is retained for table mode and
         any module that still owns lazy parameters.)
      b) Untrained-address garbling: an address that "was never trained"
         cannot exist -- every forward pass runs under coordinates the
         encoder produced, and the encoder gets gradient every step.
      c) Manifold organization: similar text maps to nearby HH points BY
         CONSTRUCTION (the encoder is a smooth function of content),
         instead of hoping 2700 independently-optimized 5-vectors happen
         to organize (the +0.306 accident). The Q1/Q2 encoder-feasibility
         question inverts: instead of regressing text -> noisy learned
         targets post-hoc at n=226, the encoder and the manifold co-adapt
         against the actual LM loss from step 0.
   'blend' interpolates: hh = (1-a)*table[task] + a*encoder(x), for
   warm-starting encoder mode from a table-mode checkpoint.

3. CAPACITY DILUTION: in encoder mode there is no per-document address to
   dilute. The certified direct-vs-microtraining gap was about ~2700
   thin slices of address-learning; the encoder is ONE shared pathway
   that every batch trains. (Backbone capacity is still finite -- d/n/k
   remain the levers -- but the failure mode "low recorded loss, garbled
   generation because the address never really cohered" is gone.)

4. ATTENTION: fixed-size causal mask buffer replaced with
   F.scaled_dot_product_attention (+ a length-cached additive bias for
   ALiBi). No fixed seq ceiling baked into buffers; context length at
   generation time is limited by memory, not by a registered mask.

UNCHANGED (locked-down in spirit): FixedBasis, ComputedScrew,
HHCoordinates' parameterization and defaults, the gated_screw residual
shape, the task-module contract, checkpoint continuity with
vocabulary growth by character identity, and sync_optimizer.

OPTIONAL: --hh_film 1 adds a tiny shared MLP mapping hh -> per-layer
residual gains for the attention and screw branches. The 5-dim screw
bottleneck only touches the MLP down-projection; if encoder mode still
under-delivers coherence, widening the address's INFLUENCE (film) is the
next lever to try before blaming training. Off by default so runs stay
comparable to v2.
"""

import argparse, importlib, json, math, os, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
BASIS_SEED = 1337

# The canonical HH parameterization. DEFAULT is HHCoordinates' historical
# init; RANGE is the per-dimension scale the inspect/analyze scripts
# already use. The AddressEncoder emits DEFAULT + tanh(raw)*RANGE, so its
# output is always inside a sane, numerically-stable box around the
# known-good starting point (angles stay angles, E stays small, etc.).
HH_DIMS    = ['J1', 'J2', 'Phi', 'E', 'Ksi']
HH_DEFAULT = torch.tensor([math.radians(14.0), 0.0, math.radians(89.3), 0.05, 0.1])
HH_RANGE   = torch.tensor([math.pi, 2.0, math.pi, 0.2, 2.0])
# J2 range widened 0.5→2.0: the registers README confirmed J2 saturated its tanh
# bound (pinned at 0.4999+) across nearly all pirouette samples — a saturated
# dimension is a dead gradient direction. The wider range gives J2 room to
# differentiate without changing its role (still controls log_sigma_L magnitude).


# ─── Locked-down shared machinery (unchanged from v2) ─────────────────────────

def cayley(B):
    I = torch.eye(B.shape[-1], device=B.device, dtype=B.dtype)
    return torch.linalg.solve(I + B, I - B)

def skew(p, k):
    B = torch.zeros(k, k, device=p.device, dtype=p.dtype)
    idx = torch.triu_indices(k, k, offset=1)
    B[idx[0], idx[1]] = p
    B[idx[1], idx[0]] = -p
    return B

class FixedBasis(nn.Module):
    def __init__(self, d_model, d_ff, k, n_layers):
        super().__init__()
        gen = torch.Generator(); gen.manual_seed(BASIS_SEED)
        def fixed_qr(shape):
            raw = torch.randn(*shape, generator=gen)
            q, _ = torch.linalg.qr(raw)
            return q[:, :shape[1]] if q.shape[1] >= shape[1] else q
        u0 = fixed_qr((d_model, k)); v0 = fixed_qr((d_model, k))
        n0 = fixed_qr((d_model, k)); m0 = fixed_qr((d_model, k))
        V  = fixed_qr((d_ff, k))
        spin_dim = k * (k - 1) // 2
        spin_template = torch.randn(n_layers, spin_dim, generator=gen) * 0.01
        self.register_buffer('u0', u0); self.register_buffer('v0', v0)
        self.register_buffer('n0', n0); self.register_buffer('m0', m0)
        self.register_buffer('V', V);   self.register_buffer('spin_template', spin_template)

class ComputedScrew(nn.Module):
    def __init__(self, d_model, d_ff, k, n_layers, layer_idx):
        super().__init__()
        self.k = k; self.n = n_layers; self.L = layer_idx

    def compute_weight(self, basis, hh_global):
        J1, J2, Phi, E, Ksi = hh_global
        L, n, k = self.L, self.n, self.k
        ksi = torch.sigmoid(Ksi)
        log_sigma_L = (J2 * torch.exp(-torch.abs(E) * L / max(n - 1, 1))).expand(k)
        delta_theta_L = (ksi * (L / max(n - 1, 1)) * (math.pi / 16)).expand(k)
        phi = J1 * L
        u_L = F.normalize(torch.cos(phi) * basis.u0 + torch.sin(phi) * basis.v0, dim=0)
        n_L = torch.cos(phi) * basis.n0 + torch.sin(phi) * basis.m0
        n_L = F.normalize(n_L - (n_L * u_L).sum(0, keepdim=True) * u_L, dim=0)
        th = Phi + delta_theta_L
        U_raw = torch.cos(th).unsqueeze(0) * u_L + torch.sin(th).unsqueeze(0) * n_L
        spin = ksi * basis.spin_template[L]
        C = cayley(skew(spin, k))
        U_L = U_raw @ C.T
        sigma_L = torch.exp(log_sigma_L)
        return (basis.V * sigma_L.unsqueeze(0)) @ U_L.T

    def forward(self, h, basis, hh_global):
        return h @ self.compute_weight(basis, hh_global)

class HHCoordinates(nn.Module):
    def __init__(self):
        super().__init__()
        self.J1  = nn.Parameter(HH_DEFAULT[0].clone())
        self.J2  = nn.Parameter(HH_DEFAULT[1].clone())
        self.Phi = nn.Parameter(HH_DEFAULT[2].clone())
        self.E   = nn.Parameter(HH_DEFAULT[3].clone())
        self.Ksi = nn.Parameter(HH_DEFAULT[4].clone())

    def copy_from(self, other):
        with torch.no_grad():
            self.J1.copy_(other.J1); self.J2.copy_(other.J2)
            self.Phi.copy_(other.Phi); self.E.copy_(other.E); self.Ksi.copy_(other.Ksi)

    def forward(self):
        return self.J1, self.J2, self.Phi, self.E, self.Ksi


# ─── Positions ────────────────────────────────────────────────────────────────

def alibi_slopes(nh):
    """Standard ALiBi head slopes: geometric sequence 2^(-8/nh * i)."""
    start = 2.0 ** (-8.0 / nh)
    return torch.tensor([start ** (i + 1) for i in range(nh)])

def rope_freqs(dh, base=10000.0):
    return 1.0 / (base ** (torch.arange(0, dh, 2).float() / dh))

def apply_rope(x, freqs):
    """x: [B, nh, T, dh]. Rotates pairs of channels by position-dependent
    angles. Relative by construction: attention scores depend only on
    position DIFFERENCES, so a window starting at offset 0 during training
    teaches the same geometry generation uses at any offset."""
    B, nh, T, dh = x.shape
    t = torch.arange(T, device=x.device).float()
    ang = torch.outer(t, freqs.to(x.device))          # [T, dh/2]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class CausalAttn(nn.Module):
    """v3: no registered fixed-size mask buffer -- causality via
    F.scaled_dot_product_attention (or a length-cached additive bias when
    ALiBi needs one). Any T works, at training or generation time."""
    def __init__(self, d, nh, pos='alibi'):
        super().__init__()
        self.nh = nh; self.dh = d // nh; self.pos = pos
        self.qkv  = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        if pos == 'alibi':
            self.register_buffer('slopes', alibi_slopes(nh), persistent=False)
        elif pos == 'rope':
            self.register_buffer('freqs', rope_freqs(self.dh), persistent=False)
        self._bias_cache = (0, None)  # (T, bias) -- rebuilt only when T changes

    def _alibi_bias(self, T, device, dtype):
        cT, cached = self._bias_cache
        if cT == T and cached is not None and cached.device == device:
            return cached
        pos = torch.arange(T, device=device)
        rel = (pos[None, :] - pos[:, None]).clamp(max=0).float()   # <= 0 below/on diag
        bias = self.slopes.to(device).view(-1, 1, 1) * rel[None]   # [nh, T, T]
        bias = bias.masked_fill(pos[None, :] > pos[:, None], float('-inf'))
        bias = bias.to(dtype).unsqueeze(0)                          # [1, nh, T, T]
        self._bias_cache = (T, bias)
        return bias

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = [t.view(B, T, self.nh, self.dh).transpose(1, 2)
                   for t in self.qkv(x).split(C, dim=2)]
        if self.pos == 'rope':
            q, k = apply_rope(q, self.freqs), apply_rope(k, self.freqs)
        if self.pos == 'alibi':
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self._alibi_bias(T, x.device, x.dtype))
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


# ─── Address encoder ──────────────────────────────────────────────────────────

class AddressEncoder(nn.Module):
    """Text -> bounded HH coordinates. Deliberately cheap: reads the SAME
    character embedding table the backbone uses (joint gradient), mean+max
    pools over the window, then a 2-layer MLP to 5 raw values squashed
    into HH_DEFAULT +/- HH_RANGE.

    Design notes:
      - Bounded output keeps every produced address inside the numerically
        sane box the screw math was designed around -- the encoder cannot
        wander into a degenerate Cayley/exp regime chasing loss.
      - The last layer is zero-initialized, so at step 0 EVERY input maps
        exactly to HH_DEFAULT: encoder mode starts as "one shared global
        address" and differentiates outward only as gradient justifies it.
        This makes early training identical to a single-address run --
        the certified-good direct_lm regime -- with content-dependence
        growing on top, instead of starting from 2700 arbitrary points.
      - enc_ctx caps how many leading tokens the pooling sees, so the
        address a window trains under matches the address a PROMPT of
        similar length would produce at generation time."""
    def __init__(self, embed, hidden=64, enc_ctx=256):
        super().__init__()
        self.embed = embed            # shared reference, not a copy
        self.enc_ctx = enc_ctx
        d = embed.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.GELU(),
            nn.Linear(hidden, 5),
        )
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('hh_default', HH_DEFAULT.clone(), persistent=False)
        self.register_buffer('hh_range',   HH_RANGE.clone(),   persistent=False)

    def forward(self, idx):
        """idx: [B, T] -> hh [5] (batch-pooled: one address per batch, the
        analogue of table mode's one-address-per-task-batch). Use encode_each
        if you ever want per-example addresses."""
        e = self.embed(idx[:, :self.enc_ctx])                  # [B, t, d]
        pooled = torch.cat([e.mean(dim=(0, 1)), e.amax(dim=1).mean(dim=0)], dim=-1)
        raw = self.net(pooled)
        return self.hh_default + torch.tanh(raw) * self.hh_range   # [5]

    def encode_text(self, idx):
        """Single sequence [T] or [1, T] -> hh [5]. Generation-time entry."""
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        return self.forward(idx)


class RegisterBank(nn.Module):
    """The addressed-component model: K explicit HH points ('registers')
    plus a router that composes them per input. The effective address is

        hh_eff(x) = sum_i softmax(router(x))_i * R_i

    a CONVEX combination -- it can never leave the hull of the registers,
    so however wrong the router is, the address stays inside territory
    the registers themselves occupy. Registers are bounded the same way
    the encoder's outputs are (HH_DEFAULT + tanh(raw)*HH_RANGE).

    Design intent (the 'agent register' idea):
      - registers 0..K-2 are the per-domain slots a training module can
        supervise (router should send pirouette text to the pirouette
        register, etc. -- see tasks_registers.py's identifiability loss);
      - register K-1 is the AGENT slot: never supervised, free for the
        router to recruit for mixed/unseen content. Its coordinates are
        ordinary parameters, tracked in the checkpoint like any other --
        'aggregator weights whose HH address is tracked separately'.

    Init: register raws start at small random around 0 (all registers
    near HH_DEFAULT, slightly separated so gradients can tell them
    apart); the router's last layer is zero-init, so step 0 routes
    UNIFORMLY -> hh_eff = mean of near-default registers ~= HH_DEFAULT.
    Same philosophy as the encoder: start as the certified-good single
    shared address, differentiate outward only as gradient justifies."""
    def __init__(self, embed, n_registers=4, hidden=64, enc_ctx=256, init_seed=7):
        super().__init__()
        self.embed = embed
        self.n_registers = n_registers
        self.enc_ctx = enc_ctx
        d = embed.embedding_dim
        gen = torch.Generator(); gen.manual_seed(init_seed)
        self.reg_raw = nn.Parameter(0.05 * torch.randn(n_registers, 5, generator=gen))
        self.router = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.GELU(),
            nn.Linear(hidden, n_registers),
        )
        nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self.register_buffer('hh_default', HH_DEFAULT.clone(), persistent=False)
        self.register_buffer('hh_range',   HH_RANGE.clone(),   persistent=False)

    def register_points(self):
        """[K, 5] -- every register's current coordinates."""
        return self.hh_default + torch.tanh(self.reg_raw) * self.hh_range

    def route_logits(self, idx):
        """idx: [B, T] -> per-example router logits [B, K]."""
        e = self.embed(idx[:, :self.enc_ctx])
        pooled = torch.cat([e.mean(dim=1), e.amax(dim=1)], dim=-1)   # [B, 2d]
        return self.router(pooled)

    def forward(self, idx):
        """[B, T] -> (hh_eff [5], logits [B, K]). Batch-pooled mixture,
        mirroring the encoder's one-address-per-batch convention."""
        logits = self.route_logits(idx)
        probs = F.softmax(logits, dim=-1).mean(dim=0)                # [K]
        hh_eff = probs @ self.register_points()                      # [5]
        return hh_eff, logits

    def encode_text(self, idx):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        hh, _ = self.forward(idx)
        return hh


class MoERegisterBank(nn.Module):
    """DeepSeek-MoE port of the register idea. Three upgrades over the plain
    RegisterBank, each a named DeepSeek trick, each ~free because a register
    is 5 floats:

      1. SHARED + ROUTED split (DeepSeek's shared-expert isolation). One
         always-on SHARED register captures what every input needs;
         n_routed registers specialize. Effective address is
             hh = shared + sum_i g_i * routed_i         (g = router softmax)
         so the shared register is a learned baseline the routed ones
         perturb -- the analogue of "common knowledge in the shared expert,
         specialization in the routed ones."

      2. FINE-GRAINED routing (DeepSeek's expert segmentation). Many small
         routed registers with top-p_active kept per input, instead of a
         few coarse ones. More registers = finer partition of the manifold
         at no extra weight cost. top_k_active gives you MoE sparsity: only
         the top few routed registers contribute, the rest are masked.

      3. PER-LAYER addresses (this is the big one for the screw design).
         Instead of one address for the whole stack, each of the n layers
         gets its OWN mixture -- a [n, 5] address matrix. This is what the
         screw architecture wants: ComputedScrew already indexes basis by
         layer L, so a per-layer address means each layer's weight matrix
         is generated from coordinates chosen for THAT layer's job. One
         router head per layer, all cheap.

    Convexity/hull safety is preserved per layer: routed weights are a
    softmax (sum to 1), shared is added at fixed unit weight, all points
    bounded to the box -- so a per-layer effective address stays within
    shared +/- (box radius), never diverging.

    Init: shared register at HH_DEFAULT (its raw is zero-init), routed
    registers clustered near zero perturbation, all per-layer router heads
    zero-init -> at step 0 every layer's address == HH_DEFAULT exactly,
    reproducing the certified single-address start. Structure grows only
    as gradient earns it, per layer independently."""
    def __init__(self, embed, n_routed=8, n_layers=8, top_k_active=0,
                 hidden=64, enc_ctx=256, init_seed=11):
        super().__init__()
        self.embed = embed
        self.n_routed = n_routed
        self.n_layers = n_layers
        self.top_k_active = top_k_active     # 0 == dense (all routed used)
        self.enc_ctx = enc_ctx
        d = embed.embedding_dim
        gen = torch.Generator(); gen.manual_seed(init_seed)
        # shared register: zero raw -> exactly HH_DEFAULT
        self.shared_raw = nn.Parameter(torch.zeros(5))
        # routed registers: small perturbations around default
        self.routed_raw = nn.Parameter(0.05 * torch.randn(n_routed, 5, generator=gen))
        # one router head PER LAYER: pooled features -> routed logits
        self.routers = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(),
                          nn.Linear(hidden, n_routed))
            for _ in range(n_layers)])
        for r in self.routers:
            nn.init.zeros_(r[-1].weight); nn.init.zeros_(r[-1].bias)
        self.register_buffer('hh_default', HH_DEFAULT.clone(), persistent=False)
        self.register_buffer('hh_range',   HH_RANGE.clone(),   persistent=False)

    def _pts(self, raw):
        return self.hh_default + torch.tanh(raw) * self.hh_range

    def shared_point(self):
        return self._pts(self.shared_raw)                 # [5]

    def routed_points(self):
        return self._pts(self.routed_raw)                 # [n_routed, 5]

    def _pool(self, idx):
        e = self.embed(idx[:, :self.enc_ctx])
        return torch.cat([e.mean(dim=1), e.amax(dim=1)], dim=-1)   # [B, 2d]

    def forward(self, idx):
        """[B, T] -> (hh_per_layer [n_layers, 5], all_logits [n_layers, B, n_routed]).
        Batch-pooled per layer, matching the one-address-per-batch convention."""
        pooled = self._pool(idx)
        shared = self.shared_point()
        routed = self.routed_points()                     # [R, 5]
        addrs, all_logits = [], []
        for L in range(self.n_layers):
            logits = self.routers[L](pooled)              # [B, R]
            if self.top_k_active and self.top_k_active < self.n_routed:
                kth = torch.topk(logits, self.top_k_active, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            g = F.softmax(logits, dim=-1).mean(dim=0)     # [R] batch-pooled gate
            addrs.append(shared + g @ routed)             # [5]
            all_logits.append(self.routers[L](pooled))    # raw logits for load-balance loss
        return torch.stack(addrs), torch.stack(all_logits)

    def encode_text(self, idx):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        addrs, _ = self.forward(idx)
        return addrs      # [n_layers, 5] -- caller may mean over layers for a scalar address


class GridHead(nn.Module):
    """The 5x5 consensus grid: a top 'head' made of G=25 HH coordinates
    whose weights are found where loss is jointly acceptable to all
    components -- 'thinking room.'

    MECHANISM: G registers arranged as a 5x5 grid, each generating a screw
    weight [d_ff, d] from its 5 coordinates via the shared basis at a
    dedicated head layer index. A gate (fixed learned prior, no routing --
    this is a HEAD, it sees the final hidden state) mixes the G screw
    outputs. The grid's coordinates move to wherever minimizes the shared
    loss -- so at convergence they sit at the consensus point 'acceptable
    to all components' feeding the head. Because each of 25 coordinates
    generates a full [d_ff, d] matrix, the head expresses ~25 * d_ff * d
    effective weights from 25 * 5 = 125 learnable scalars: the 'silly
    number of model weights for that task' you're after, at 125-float cost.

    It's a projection-augmenting head: standard head is Linear(d, vocab);
    this adds a grid-screw transform of the final hidden state BEFORE that
    linear, giving the readout its own generated-weight capacity. Zero-init
    gate -> at step 0 the grid contributes nothing and the head is exactly
    the plain Linear, so turning the grid on never destabilizes a trained
    head."""
    def __init__(self, d, basis, k, grid=5, head_layer_idx=0, init_seed=13):
        super().__init__()
        self.d, self.k = d, k
        self.G = grid * grid
        self.grid = grid
        self.basis = basis
        # a ComputedScrew at a fixed head layer index (its own phase of the basis);
        # compute_weight returns [d_ff, d], so it maps d_ff -> d
        self.screw = ComputedScrew(d, d * 4, k, n_layers=max(head_layer_idx + 1, 1),
                                   layer_idx=head_layer_idx)
        gen = torch.Generator(); gen.manual_seed(init_seed)
        self.grid_raw = nn.Parameter(0.05 * torch.randn(self.G, 5, generator=gen))
        # the grid head's own up-projection d -> d_ff (doesn't reuse backbone gates)
        self.up = nn.Linear(d, d * 4)
        # gate over grid cells from the final hidden state; zero-init
        self.gate = nn.Linear(d, self.G)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        # scalar scale on the whole contribution; zero-init so head == plain
        # Linear at step 0 (turning the grid on never destabilizes a warm head)
        self.out_scale = nn.Parameter(torch.zeros(1))
        self.register_buffer('hh_default', HH_DEFAULT.clone(), persistent=False)
        self.register_buffer('hh_range',   HH_RANGE.clone(),   persistent=False)

    def grid_points(self):
        return self.hh_default + torch.tanh(self.grid_raw) * self.hh_range   # [G, 5]

    def forward(self, h):
        """h: [B, T, d] final pre-head hidden. Returns h + consensus term."""
        pts = self.grid_points()                          # [G, 5]
        gate = F.softmax(self.gate(h), dim=-1)            # [B, T, G]
        up = F.gelu(self.up(h))                           # [B, T, d_ff]
        acc = 0.0
        for gi in range(self.G):
            hh = tuple(pts[gi, j] for j in range(5))
            w = self.screw.compute_weight(self.basis, hh)  # [d_ff, d]
            contrib = up @ w                               # [B, T, d]
            acc = acc + gate[..., gi:gi+1] * contrib
        return h + self.out_scale * acc                    # [B, T, d]


# ─── The shared backbone ──────────────────────────────────────────────────────

class SharedSwarm(nn.Module):
    def __init__(self, vocab, d, n, k, seq_len=512, pos='alibi',
                 address='encoder', blend_alpha=0.5, enc_hidden=64,
                 enc_ctx=256, hh_film=False, n_registers=4,
                 n_routed=8, top_k_active=0, grid_head=0):
        super().__init__()
        self.d, self.n, self.k = d, n, k
        self.seq_len = seq_len            # default TRAINING window; not a hard ceiling unless pos=='learned'
        self.pos_mode = pos
        self.address_mode = address
        self.blend_alpha = blend_alpha
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d) if pos == 'learned' else None
        self.drop  = nn.Dropout(0.1)
        self.ln_f  = nn.LayerNorm(d)
        self.head  = nn.Linear(d, vocab, bias=False)
        nh = self._safe_nh(d)
        self.attn    = nn.ModuleList([CausalAttn(d, nh, pos=pos) for _ in range(n)])
        self.ln_attn = nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
        self.gates   = nn.ModuleList([nn.Linear(d, d * 4) for _ in range(n)])
        self.ln_pre  = nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
        self.basis   = FixedBasis(d, d * 4, k, n)
        self.screws  = nn.ModuleList([ComputedScrew(d, d * 4, k, n, L) for L in range(n)])
        self.hh_by_task = nn.ModuleDict()   # used by table/blend; harmless empty in encoder mode
        self.encoder = AddressEncoder(self.embed, hidden=enc_hidden, enc_ctx=enc_ctx) \
            if address in ('encoder', 'blend') else None
        self.registers = RegisterBank(self.embed, n_registers=n_registers,
                                      hidden=enc_hidden, enc_ctx=enc_ctx) \
            if address == 'registers' else None
        self.moe = MoERegisterBank(self.embed, n_routed=n_routed, n_layers=n,
                                   top_k_active=top_k_active, hidden=enc_hidden,
                                   enc_ctx=enc_ctx) \
            if address == 'moe' else None
        self.grid_head = GridHead(d, self.basis, k, grid=5) if grid_head else None
        self.last_router_logits = None   # stashed per forward in registers mode,
                                         # so a loss module can add identifiability
                                         # terms without a second routing pass
        self.last_moe_logits = None      # [n_layers, B, n_routed], for load-balance loss
        # optional: hh -> per-layer residual gains (attn branch, screw branch)
        self.film = None
        if hh_film:
            self.film = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 2 * n))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)

    @staticmethod
    def _safe_nh(d):
        t = max(1, d // 64)
        while t > 1 and d % t != 0: t -= 1
        return t

    def add_task(self, name):
        device = next(self.parameters()).device
        self.hh_by_task[name] = HHCoordinates().to(device)
        return self.hh_by_task[name]

    # ---- addressing ----
    def resolve_hh(self, idx, task=None):
        """One place that decides what coordinates this forward runs under.
        Returns a 5-tuple of scalars (table) or a [5] tensor unpacked to a
        tuple (encoder/blend) -- ComputedScrew accepts either."""
        if self.address_mode == 'table':
            if task is None:
                raise ValueError("table addressing needs a task name")
            return self.hh_by_task[task]()
        if self.address_mode == 'registers':
            hh_eff, logits = self.registers(idx)
            self.last_router_logits = logits
            return tuple(hh_eff[i] for i in range(5))
        if self.address_mode == 'moe':
            addrs, all_logits = self.moe(idx)      # [n, 5], [n, B, R]
            self.last_moe_logits = all_logits
            return addrs                            # per-layer matrix; forward_with_hh handles it
        hh_enc = self.encoder(idx)
        if self.address_mode == 'encoder' or task is None or task not in self.hh_by_task:
            return tuple(hh_enc[i] for i in range(5))
        a = self.blend_alpha
        tbl = torch.stack(list(self.hh_by_task[task]()))
        hh = (1 - a) * tbl + a * hh_enc
        return tuple(hh[i] for i in range(5))

    # ---- forward paths ----
    def gated_screw(self, x, L, hh, block_grad=False):
        if block_grad:
            with torch.no_grad():
                h_up = F.gelu(self.gates[L](self.ln_pre[L](x)))
        else:
            h_up = F.gelu(self.gates[L](self.ln_pre[L](x)))
        return self.screws[L](h_up, self.basis, hh)

    def _hh_for_layer(self, hh, L):
        """hh may be a single address (5-tuple or [5] tensor -> same for all
        layers) or a per-layer matrix [n, 5] -> row L. Returns a 5-tuple."""
        if torch.is_tensor(hh) and hh.dim() == 2:
            row = hh[L]
            return tuple(row[i] for i in range(5))
        if torch.is_tensor(hh) and hh.dim() == 1:
            return tuple(hh[i] for i in range(5))
        return hh   # already a 5-tuple

    def forward_with_hh(self, idx, hh, embed_scale=None):
        """Explicit-coordinates forward. hh: a single address (5-tuple or
        [5] tensor, applied to every layer) OR a per-layer matrix [n, 5]
        (moe mode -- each layer gets its own address). Returns
        (logits, pre-head hidden [B, T, d])."""
        per_layer = torch.is_tensor(hh) and hh.dim() == 2
        B, T = idx.shape
        x = self.embed(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device).clamp(max=self.pos.num_embeddings - 1))
        x = self.drop(x)
        if embed_scale is not None:
            x = x * embed_scale.view(1, -1, 1)
        gains = None
        if self.film is not None and not per_layer:
            hh_tuple = self._hh_for_layer(hh, 0)
            g = torch.tanh(self.film(torch.stack(
                [h if torch.is_tensor(h) else torch.tensor(h) for h in hh_tuple])))
            gains = 1.0 + 0.5 * g
        for L in range(self.n):
            hh_L = self._hh_for_layer(hh, L)
            ga = gains[2 * L]     if gains is not None else 1.0
            gs = gains[2 * L + 1] if gains is not None else 1.0
            x = x + ga * self.attn[L](self.ln_attn[L](x))
            x = x + gs * self.gated_screw(x, L, hh_L)
        x = self.ln_f(x)
        if self.grid_head is not None:
            x = self.grid_head(x)
        return self.head(x), x

    def forward(self, idx, task=None, embed_scale=None, return_hidden=False):
        """Back-compatible: model(x, task) -> logits. In encoder mode the
        task name is accepted and ignored (address comes from content)."""
        hh = self.resolve_hh(idx, task)
        logits, hidden = self.forward_with_hh(idx, hh, embed_scale=embed_scale)
        return (logits, hidden) if return_hidden else logits

    def connectivity_loss(self, task, ksi_target=0.89):
        _, _, _, _, Ksi = self.hh_by_task[task]()
        return (torch.sigmoid(Ksi) - ksi_target) ** 2


# ─── Corpus / vocab (unchanged in behavior) ───────────────────────────────────

def chunk_text(text, n_chunks):
    if n_chunks <= 1: return [text]
    n = len(text)
    target = [round(i * n / n_chunks) for i in range(1, n_chunks)]
    cuts = []
    for t in target:
        lo, hi = max(0, t - 2000), min(n, t + 2000)
        bpos = text[lo:hi].find('\n\n')
        cuts.append(t if bpos == -1 else lo + bpos)
    cuts = sorted(set([0] + cuts + [n]))
    return [text[cuts[i]:cuts[i+1]] for i in range(len(cuts) - 1)]

def build_union_vocab(texts):
    chars = set()
    for t in texts: chars.update(t)
    chars = sorted(chars)
    return chars, {c: i for i, c in enumerate(chars)}

def to_tensor(text, stoi):
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)

def load_corpora(args):
    members = []
    if not args.skip_shakespeare:
        if not os.path.exists(args.shakespeare_path):
            urllib.request.urlretrieve(SHAKESPEARE_URL, args.shakespeare_path)
        members.append(('shakespeare', open(args.shakespeare_path, encoding='utf-8', errors='replace').read()))
    if args.pirouette_path:
        members.append(('pirouette', open(args.pirouette_path, encoding='utf-8', errors='replace').read()))
    if args.classics_path:
        classics_text = open(args.classics_path, encoding='utf-8', errors='replace').read()
        for i, ch in enumerate(chunk_text(classics_text, args.classics_chunks)):
            members.append((f'classics_{i+1}', ch))
        del classics_text
    return members

def build_shared_context(args, device, extra_chars=None):
    members = load_corpora(args)
    vocab_chars, _ = build_union_vocab([t for _, t in members])
    if extra_chars:
        vocab_chars = sorted(set(vocab_chars) | set(extra_chars))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    data = {}
    for name, text in members:
        n = int(0.9 * len(text))
        data[name] = (to_tensor(text[:n], stoi).to(device), to_tensor(text[n:], stoi).to(device))
    return {'members': members, 'vocab_chars': vocab_chars, 'stoi': stoi, 'data': data, 'device': device}


# ─── Checkpoint continuity ────────────────────────────────────────────────────

def load_checkpoint_into_model(model, ckpt, merged_chars):
    """Vocab rows copied by character identity (unchanged from v2); every
    other matching tensor restored; pos-scheme-specific tensors that the
    current model doesn't have (e.g. old 'pos.weight' when migrating
    learned->alibi) are reported and skipped rather than fatal."""
    old_chars = ckpt['vocab_chars']
    old_index = {c: i for i, c in enumerate(old_chars)}

    for name in ckpt.get('task_names', []):
        if name not in model.hh_by_task:
            model.add_task(name)

    state = dict(ckpt['model_state'])
    old_embed = state.pop('embed.weight')
    old_head = state.pop('head.weight')
    # encoder shares the embed table by reference; its copy of the key
    # (if the ckpt has one) is the same rows -- drop to avoid double-copy
    state.pop('encoder.embed.weight', None)
    state.pop('registers.embed.weight', None)
    state.pop('moe.embed.weight', None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m not in ('encoder.embed.weight', 'registers.embed.weight', 'moe.embed.weight')]  # shared-by-ref; copied below
    if missing or unexpected:
        print(f"  [checkpoint] partial restore -- missing={missing} unexpected={unexpected}")

    grown = len(merged_chars) - len(old_chars)
    with torch.no_grad():
        for c, old_i in old_index.items():
            new_i = merged_chars.index(c)
            model.embed.weight[new_i].copy_(old_embed[old_i])
            model.head.weight[new_i].copy_(old_head[old_i])
    print(f"  [checkpoint] restored {len(old_chars)} vocab rows by character identity"
          + (f"; {grown} new characters start fresh" if grown else ""))
    return missing, unexpected


def load_task_modules(module_specs):
    modules = []
    for spec in module_specs:
        path, _, cfg_str = spec.partition(':')
        cfg = {}
        for kv in cfg_str.split(',') if cfg_str else []:
            if not kv: continue
            key, _, val = kv.partition('=')
            for cast in (int, float):
                try: val = cast(val); break
                except ValueError: pass
            cfg[key] = val
        mod = importlib.import_module(path)
        modules.append(mod.build(cfg))
    return modules


def sync_optimizer(model, opt):
    """Retained from v2 (it fixed a certified, load-bearing bug there).
    In encoder mode nothing should ever trip it -- if it reports newly
    synced parameters in an encoder-mode run, that's a red flag that some
    module is still creating lazy parameters, worth investigating."""
    known = set(id(p) for group in opt.param_groups for p in group['params'])
    new_params = [p for p in model.parameters() if id(p) not in known]
    if new_params:
        opt.add_param_group({'params': new_params})
    return len(new_params)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modules', nargs='+', required=True)
    p.add_argument('--d', type=int, default=128)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--seq_len', type=int, default=512,
                   help="default training window / learned-pos table size. NOT a "
                        "generation ceiling under --pos alibi/rope.")
    p.add_argument('--pos', choices=['alibi', 'rope', 'learned'], default='alibi')
    p.add_argument('--address', choices=['encoder', 'table', 'blend', 'registers', 'moe'], default='encoder')
    p.add_argument('--n_registers', type=int, default=4,
                   help="registers mode: number of HH register slots (last one is the"
                        " unsupervised 'agent' slot)")
    p.add_argument('--n_routed', type=int, default=8,
                   help="moe mode: number of routed (fine-grained) registers; one shared "
                        "register is always added on top")
    p.add_argument('--top_k_active', type=int, default=0,
                   help="moe mode: keep only top-k routed registers per input (0 = dense)")
    p.add_argument('--grid_head', type=int, default=0,
                   help="add the 5x5 consensus GridHead on top of the backbone")
    p.add_argument('--blend_alpha', type=float, default=0.5)
    p.add_argument('--enc_hidden', type=int, default=64)
    p.add_argument('--enc_ctx', type=int, default=256)
    p.add_argument('--hh_film', type=int, default=0)
    p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--log_interval', type=int, default=350)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shakespeare_path', default='tinyshakespeare.txt')
    p.add_argument('--skip_shakespeare', action='store_true')
    p.add_argument('--pirouette_path', default=None)
    p.add_argument('--classics_path', default=None)
    p.add_argument('--classics_chunks', type=int, default=4)
    p.add_argument('--out', default='testbed_results.json')
    p.add_argument('--init_from', default=None)
    p.add_argument('--save_to', default='model_checkpoint_v3.pt')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ckpt = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        print(f"Loaded checkpoint: {args.init_from} (prior lifetime steps: {ckpt.get('total_steps_trained', 0)})")

    task_modules = load_task_modules(args.modules)

    extra_chars = set(ckpt['vocab_chars']) if ckpt else set()
    for m in task_modules:
        c = m.discover_vocab_chars()
        if c:
            extra_chars |= set(c)
    shared_ctx = build_shared_context(args, device, extra_chars=extra_chars or None)

    if ckpt:
        d, n, k = ckpt['d'], ckpt['n'], ckpt['k']
        if (d, n, k) != (args.d, args.n, args.k):
            print(f"  NOTE: continuing from checkpoint architecture ({d},{n},{k}); "
                  f"--d/--n/--k ignored.")
        pos = ckpt.get('pos_mode', args.pos)
        address = ckpt.get('address_mode', args.address)
        if pos != args.pos:
            print(f"  NOTE: checkpoint was --pos {pos}; keeping that (use migrate_checkpoint.py "
                  f"to change position scheme).")
        if address != args.address:
            print(f"  NOTE: checkpoint was --address {address}; this run uses --address {args.address} "
                  f"(legal: table<->blend<->encoder all share the backbone; blend is the "
                  f"recommended bridge from a table checkpoint).")
            address = args.address
        seq_len = ckpt.get('seq_len', args.seq_len)
        if pos == 'learned' and seq_len != args.seq_len:
            print(f"  NOTE: learned positions -- keeping checkpoint seq_len={seq_len}.")
    else:
        d, n, k, pos, address, seq_len = args.d, args.n, args.k, args.pos, args.address, args.seq_len

    n_registers = ckpt.get('n_registers', args.n_registers) if ckpt else args.n_registers
    n_routed = ckpt.get('n_routed', args.n_routed) if ckpt else args.n_routed
    top_k_active = ckpt.get('top_k_active', args.top_k_active) if ckpt else args.top_k_active
    grid_head = ckpt.get('grid_head', args.grid_head) if ckpt else args.grid_head
    model = SharedSwarm(len(shared_ctx['vocab_chars']), d, n, k, seq_len=seq_len,
                        pos=pos, address=address, blend_alpha=args.blend_alpha,
                        enc_hidden=args.enc_hidden, enc_ctx=args.enc_ctx,
                        hh_film=bool(args.hh_film), n_registers=n_registers,
                        n_routed=n_routed, top_k_active=top_k_active,
                        grid_head=bool(grid_head)).to(device)

    if ckpt:
        load_checkpoint_into_model(model, ckpt, shared_ctx['vocab_chars'])

    if address in ('table', 'blend'):
        for name, _ in shared_ctx['members']:
            if name not in model.hh_by_task:
                model.add_task(name)

    for m in task_modules:
        m.setup(model, shared_ctx)
        if ckpt and m.name in ckpt.get('task_states', {}):
            m.load_state(ckpt['task_states'][m.name])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  Modules: {[m.name for m in task_modules]}")
    print(f"Arch: d={d} n={n} k={k} pos={pos} address={address} "
          f"vocab={len(shared_ctx['vocab_chars'])} params={n_params:,}")

    loss_modules  = [m for m in task_modules if m.kind == 'loss']
    epoch_modules = [m for m in task_modules if m.kind == 'epoch']

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    results = {'modules': [m.name for m in task_modules], 'args': vars(args),
               'eval_log': [], 'loss_log': []}

    for step in range(args.steps + 1):
        if step % args.log_interval == 0 or step == args.steps:
            entry = {'step': step}
            for m in task_modules:
                ev = m.eval(model, shared_ctx)
                if ev: entry[m.name] = ev
            results['eval_log'].append(entry)
            print(f"  step {step:>6} | " + " | ".join(
                f"{m.name}={entry.get(m.name)}" for m in task_modules if m.name in entry))

        if step == args.steps:
            break

        if loss_modules:
            total_loss = 0.0
            log_row = {'step': step}
            batches = [m.step_batch(model, shared_ctx) for m in loss_modules]
            newly = sync_optimizer(model, opt)
            if newly and step % args.log_interval == 0:
                print(f"  [optimizer] synced {newly} newly-registered parameters")
            for m in loss_modules:
                m.zero_grad()
            opt.zero_grad()
            for m, batch in zip(loss_modules, batches):
                loss_val, weight, log = m.loss(model, shared_ctx, batch)
                total_loss = total_loss + weight * loss_val
                log_row[m.name] = log
            total_loss.backward()
            for m in loss_modules:
                m.post_backward(step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % args.log_interval == 0:
                results['loss_log'].append(log_row)

        for m in epoch_modules:
            if m.should_run(step):
                log = m.run_epoch(model, shared_ctx, step)
                if step % args.log_interval == 0:
                    results['loss_log'].append({'step': step, m.name: log})

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> {args.out}")

    task_states = {}
    for m in task_modules:
        s = m.state_dict()
        if s is not None:
            task_states[m.name] = s

    prior_steps = ckpt.get('total_steps_trained', 0) if ckpt else 0
    checkpoint_out = {
        'model_state': model.state_dict(),
        'vocab_chars': shared_ctx['vocab_chars'],
        'd': d, 'n': n, 'k': k, 'seq_len': seq_len,
        'pos_mode': pos, 'address_mode': address,
        'enc_hidden': args.enc_hidden, 'enc_ctx': args.enc_ctx,
        'hh_film': bool(args.hh_film), 'n_registers': n_registers,
        'n_routed': n_routed, 'top_k_active': top_k_active, 'grid_head': bool(grid_head),
        'task_names': list(model.hh_by_task.keys()),
        'task_states': task_states,
        'total_steps_trained': prior_steps + args.steps,
    }
    torch.save(checkpoint_out, args.save_to)
    print(f"Checkpoint -> {args.save_to} "
          f"(this pass: {args.steps}; lifetime: {checkpoint_out['total_steps_trained']})")


def build_model_from_checkpoint(ckpt, device='cpu'):
    """Shared helper for generate.py / find_address.py / diagnostics:
    reconstructs a SharedSwarm exactly as the checkpoint describes it and
    loads its state. Returns (model, vocab_chars, stoi)."""
    vocab_chars = ckpt['vocab_chars']
    model = SharedSwarm(
        len(vocab_chars), ckpt['d'], ckpt['n'], ckpt['k'],
        seq_len=ckpt.get('seq_len', 512),
        pos=ckpt.get('pos_mode', 'learned'),
        address=ckpt.get('address_mode', 'table'),
        enc_hidden=ckpt.get('enc_hidden', 64),
        enc_ctx=ckpt.get('enc_ctx', 256),
        hh_film=ckpt.get('hh_film', False),
        n_registers=ckpt.get('n_registers', 4),
        n_routed=ckpt.get('n_routed', 8),
        top_k_active=ckpt.get('top_k_active', 0),
        grid_head=ckpt.get('grid_head', False),
    ).to(device)
    for name in ckpt.get('task_names', []):
        model.add_task(name)
    state = dict(ckpt['model_state'])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [load] partial restore -- missing={missing} unexpected={unexpected}")
    model.eval()
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    return model, vocab_chars, stoi


if __name__ == '__main__':
    main()
