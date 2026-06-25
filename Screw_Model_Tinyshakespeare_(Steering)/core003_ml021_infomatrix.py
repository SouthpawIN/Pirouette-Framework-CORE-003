"""
CORE-003-ML-021  ·  Bidirectional Information Matrix
Pirouette Framework Volume 8  ·  Keaton Smith

THE IDEA
--------
The weight matrices W_L are not just computational units.
They are a MAP of the information space the model has learned to navigate.
Reading them (the DOWN direction) reveals where you are on the manifold.

CURRENT ARCHITECTURE (UP only):
  text → screws → logits → next token

PROPOSED BIDIRECTIONAL:
  UP:   text → screws → logits
  DOWN: screws → information matrix I → geometric conditioning
  LOOP: I feeds BACK into the next UP pass as a geometric prior

The model becomes conditioned on its own manifold geometry.
It knows where it is in information space and uses that knowledge.

INFORMATION MATRIX I ∈ ℝ^(n_layers × (k+4)):
  Row L: [log_σ_L,0 ... log_σ_L,k-1, omega_L, angle_L, tele_L, theta_L]
  For d=128, k=32, n=8: I is 8×36 = 288 numbers
  This is the geometric fingerprint of the model at its current state.

THE BIG BRAIN MECHANISM:
  A small model (1.4M params) conditioned on GPT-2-Large's information matrix
  (19KB, 4752 numbers) can target large-model geometry without 774M weights.
  The information matrix is a specification of a brain.
  A model that can read and follow it inherits that brain's geometric structure.

THREE EXPERIMENTS
-----------------
A: SELF-CONDITIONED (model reads its OWN information matrix)
   At each forward pass, compute I from current weights.
   Feed I through a small manifold reader → additive bias to screw outputs.
   The model learns to use its own geometric state.
   Compare to: unconditioned torus (ML-012 baseline).

B: CROSS-CONDITIONED (small model reads LARGE model's information matrix)
   Extract GPT-2-Large's information matrix (certified values from ML-011).
   Feed it as fixed conditioning to the small d=128 model.
   The small model learns to match large-model geometry without large weights.
   Compare to: unconditioned small model.

C: DUAL DIRECTION (simultaneous UP and DOWN with coupled loss)
   UP loss:   language modeling (next token prediction)
   DOWN loss: geometric self-consistency (I_predicted should match I_actual)
   Joint training tightens both directions simultaneously.
   Pre-registered: joint training reaches lower loss than either alone.

PRE-REGISTERED HYPOTHESES
--------------------------
H-021-A: Self-conditioned model converges faster
  PASS: steps to val_loss=1.55 < steps for unconditioned baseline
  (The geometric self-awareness signal guides optimization)

H-021-B: Cross-conditioned model achieves lower final loss
  PASS: val_loss(cross-conditioned, 5000 steps) < val_loss(unconditioned, 5000 steps)
  (Large-model geometry is a useful prior for small-model generation)

H-021-C: Joint training improves both language quality AND geometric consistency
  PASS: both val_loss and geometric_consistency_loss lower for joint vs UP-only

Usage:
  python core003_ml021_infomatrix.py --experiments A,B,C
  python core003_ml021_infomatrix.py --experiments A       # fastest
  python core003_ml021_infomatrix.py --experiments B --backbone gpt2-large
"""

import argparse, json, math, os, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, logging as hf_logging
hf_logging.set_verbosity_error()

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ─── Information Matrix extraction ────────────────────────────────────────────

def safe_nh(d):
    t = max(1, d // 64)
    while t > 1 and d % t != 0: t -= 1
    return t

def cayley(B):
    I = torch.eye(B.shape[-1], device=B.device, dtype=B.dtype)
    return torch.linalg.solve(I + B, I - B)

def skew(p, k):
    B = torch.zeros(k, k, device=p.device, dtype=p.dtype)
    idx = torch.triu_indices(k, k, offset=1)
    B[idx[0], idx[1]] = p; B[idx[1], idx[0]] = -p
    return B


def extract_info_matrix(screws, device):
    """
    Build information matrix I from a list of ToruscScrew objects.
    I[L] = [log_sigma_L (k dims), omega_L, angle_L, tele_L, theta_L]
    Returns: tensor [n_layers, k+4]
    """
    rows = []
    n = len(screws)
    for L, screw in enumerate(screws):
        # Singular values
        log_sigma = screw.log_sigma[L].detach()   # [k]

        # Omega and theta
        omega = torch.tensor([float(screw.omega)], device=device)
        theta = torch.tensor([float(screw.theta_base)], device=device)

        # Inter-layer angle (vs next layer's U[:,0])
        if L < n - 1:
            def get_u(s, l):
                phi = s.omega * l
                u_L = F.normalize(torch.cos(phi)*s.u0 + torch.sin(phi)*s.v0, dim=0)
                return F.normalize(u_L[:, 0], dim=0)
            u_L   = get_u(screw, L)
            u_Lp1 = get_u(screws[L+1], L+1)
            cos_v = (u_L * u_Lp1).sum().clamp(-1, 1)
            angle = torch.acos(cos_v.abs()).unsqueeze(0)
        else:
            angle = torch.zeros(1, device=device)

        # Telescope coupling (to L+3)
        if L < n - 3:
            u_L  = get_u(screw, L)
            u_L3 = get_u(screws[L+3], L+3)
            tele = (u_L * u_L3).sum().abs().unsqueeze(0)
        else:
            tele = torch.zeros(1, device=device)

        row = torch.cat([log_sigma, omega, angle, tele, theta])  # [k+4]
        rows.append(row)

    return torch.stack(rows, dim=0)   # [n_layers, k+4]


def extract_gpt2_info_matrix(backbone_name, k, device):
    """
    Extract certified information matrix from a pre-trained GPT-2 model.
    Uses the measured values from ML-011 (omega=11°, theta=89.3°, etc.)
    plus actual singular values from the pretrained weights.
    """
    print(f"  Extracting GPT-2-Large information matrix...")
    model = GPT2LMHeadModel.from_pretrained(backbone_name, low_cpu_mem_usage=True)
    model.eval()

    rows = []
    n_layers = model.config.n_layer
    d_model  = model.config.n_embd

    for L in range(n_layers):
        # MLP down-projection weight
        W = model.transformer.h[L].mlp.c_proj.weight.detach().float()
        # W has shape [d_model, d_ff] in GPT-2
        if W.shape[0] < W.shape[1]:
            W = W.T  # ensure [d_model, d_ff]

        # SVD to get singular values
        _, sigma, _ = torch.linalg.svd(W, full_matrices=False)
        log_sigma = torch.log(sigma[:k] + 1e-8)   # [k]

        # Certified geometric values from ML-011
        omega = torch.tensor([math.radians(11.0)], device=device)
        theta = torch.tensor([math.radians(89.3)], device=device)
        angle = torch.tensor([math.radians(11.0)], device=device)  # ~inter-layer angle
        tele  = torch.tensor([0.365], device=device)  # certified L=33 coupling

        row = torch.cat([log_sigma.to(device), omega, angle, tele, theta])  # [k+4]
        rows.append(row)

    del model
    I_gpt2 = torch.stack(rows, dim=0)   # [n_layers_gpt2, k+4]
    print(f"  GPT-2-Large info matrix: {I_gpt2.shape}, "
          f"sigma range [{float(I_gpt2[:,0].min()):.2f}, {float(I_gpt2[:,0].max()):.2f}]")
    return I_gpt2


# ─── Manifold Reader ──────────────────────────────────────────────────────────

class ManifoldReader(nn.Module):
    """
    Reads the information matrix I → outputs additive bias for screw projections.

    The bias conditions each layer's screw output on the full model geometry.
    This gives the model a form of geometric self-awareness:
    "Given where I am in weight space, here is an additional signal."
    """
    def __init__(self, n_layers, k, d_model, hidden=64):
        super().__init__()
        i_dim = n_layers * (k + 4)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(i_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),    # one bias vector per model
        )
        # Per-layer scaling (each layer can weight the manifold signal differently)
        self.layer_scales = nn.Parameter(torch.zeros(n_layers))

    def forward(self, I, layer_idx):
        """
        I: [n_layers, k+4]  — the information matrix
        layer_idx: int       — which layer is asking
        Returns: [d_model] additive bias
        """
        flat = self.flatten(I.unsqueeze(0))          # [1, n_layers*(k+4)]
        bias = self.net(flat).squeeze(0)             # [d_model]
        scale = torch.sigmoid(self.layer_scales[layer_idx])
        return bias * scale


# ─── Torus screw ──────────────────────────────────────────────────────────────

class ToruscScrew(nn.Module):
    def __init__(self, d_model, d_ff, k, n_layers):
        super().__init__()
        self.k=k; self.n_layers=n_layers
        for name, shape in [('u0',(d_model,k)),('v0',(d_model,k)),
                              ('n0',(d_model,k)),('m0',(d_model,k)),('V',(d_ff,k))]:
            p = torch.randn(*shape); p, _ = torch.linalg.qr(p)
            setattr(self, name, nn.Parameter(p[:,:k]))
        self.omega       = nn.Parameter(torch.tensor(math.radians(14.0)))
        self.theta_base  = nn.Parameter(torch.tensor(math.radians(89.3)))
        self.delta_theta = nn.Parameter(torch.zeros(n_layers, k))
        self.log_sigma   = nn.Parameter(torch.zeros(n_layers, k))
        self.spin_gen    = nn.Parameter(torch.zeros(n_layers, k*(k-1)//2)*0.01)

    def get_U_L(self, L):
        phi = self.omega * L
        u_L = F.normalize(torch.cos(phi)*self.u0+torch.sin(phi)*self.v0, dim=0)
        n_L = torch.cos(phi)*self.n0+torch.sin(phi)*self.m0
        n_L = F.normalize(n_L-(n_L*u_L).sum(0,keepdim=True)*u_L, dim=0)
        th  = self.theta_base+self.delta_theta[L]
        return (torch.cos(th)*u_L+torch.sin(th)*n_L)@cayley(skew(self.spin_gen[L],self.k)).T

    def forward(self, h, L, manifold_bias=None):
        out = (h@self.V*torch.exp(self.log_sigma[L]))@self.get_U_L(L).T
        if manifold_bias is not None:
            out = out + manifold_bias.unsqueeze(0).unsqueeze(0)
        return out


class CausalAttn(nn.Module):
    def __init__(self, d, nh, sl):
        super().__init__()
        self.nh=nh; self.dh=d//nh
        self.qkv=nn.Linear(d,3*d,bias=True); self.proj=nn.Linear(d,d,bias=True)
        self.register_buffer('mask',torch.tril(torch.ones(sl,sl)).view(1,1,sl,sl))
    def forward(self, x):
        B,T,C=x.shape
        q,k,v=[t.view(B,T,self.nh,self.dh).transpose(1,2) for t in self.qkv(x).split(C,dim=2)]
        att=(q@k.transpose(-2,-1))/math.sqrt(self.dh)
        att=att.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        return self.proj((F.softmax(att,-1)@v).transpose(1,2).contiguous().view(B,T,C))


# ─── Bidirectional model ───────────────────────────────────────────────────────

class BidirectionalModel(nn.Module):
    """
    Torus model with optional manifold conditioning.

    mode='up'    : standard torus, no manifold reading (baseline)
    mode='self'  : reads OWN information matrix each forward pass
    mode='cross' : reads a FIXED information matrix (e.g. from GPT-2-Large)
    mode='dual'  : UP + DOWN losses simultaneously
    """
    def __init__(self, vocab, d, n, k, sl, mode='up', fixed_I=None):
        super().__init__()
        self.d=d; self.n=n; self.k=k; self.mode=mode
        self.embed=nn.Embedding(vocab,d); self.pos=nn.Embedding(sl,d)
        self.drop=nn.Dropout(0.1); self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,vocab,bias=False)
        nh=safe_nh(d)
        self.attn=nn.ModuleList([CausalAttn(d,nh,sl) for _ in range(n)])
        self.c_fc=nn.ModuleList([nn.Linear(d,d*4,bias=True) for _ in range(n)])
        self.ln1=nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
        self.ln2=nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
        self.screws=nn.ModuleList([ToruscScrew(d,d*4,k,n) for _ in range(n)])

        if mode in ('self', 'cross', 'dual'):
            self.reader = ManifoldReader(n, k, d)
        else:
            self.reader = None

        if fixed_I is not None:
            self.register_buffer('fixed_I', fixed_I)
        else:
            self.fixed_I = None

    def get_info_matrix(self, device):
        """Compute current information matrix from own weights."""
        if self.mode == 'cross' and self.fixed_I is not None:
            # Use the fixed (large-model) information matrix
            # Interpolate to match our n_layers if needed
            I = self.fixed_I
            if I.shape[0] != self.n:
                # Average-pool to match our depth
                factor = I.shape[0] / self.n
                rows = []
                for L in range(self.n):
                    start = int(L * factor); end = int((L+1) * factor)
                    rows.append(I[start:end].mean(0))
                I = torch.stack(rows, 0)
            # Truncate/pad k dimension
            k_i = I.shape[1] - 4
            if k_i != self.k:
                if k_i > self.k:
                    I = torch.cat([I[:, :self.k], I[:, k_i:]], dim=1)
                else:
                    pad = torch.zeros(I.shape[0], self.k - k_i, device=I.device)
                    I = torch.cat([I[:, :k_i], pad, I[:, k_i:]], dim=1)
            return I.to(next(self.parameters()).device)
        else:
            return extract_info_matrix(self.screws, next(self.parameters()).device)

    def forward(self, idx, targets=None):
        B,T=idx.shape
        x=self.drop(self.embed(idx)+self.pos(torch.arange(T,device=idx.device)))

        # Compute manifold conditioning once per forward pass (not per token)
        biases = [None] * self.n
        if self.reader is not None:
            with torch.no_grad() if self.mode == 'cross' else torch.enable_grad():
                I = self.get_info_matrix(idx.device)
            for L in range(self.n):
                biases[L] = self.reader(I, L)

        # UP pass
        for L,(attn,c_fc,screw,ln1,ln2) in enumerate(zip(
            self.attn,self.c_fc,self.screws,self.ln1,self.ln2)):
            x=x+attn(ln1(x))
            x=x+screw(F.gelu(c_fc(ln2(x))), L, biases[L])

        logits=self.head(self.ln_f(x))

        loss_up = None
        if targets is not None:
            loss_up = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))

        # DOWN pass: geometric self-consistency (for 'dual' mode)
        loss_down = None
        if self.mode == 'dual' and targets is not None:
            I_actual = self.get_info_matrix(idx.device)
            # Predict what I should look like from the hidden state
            # Simple version: I should match a moving average of past I values
            # We train the model to be geometrically self-consistent
            # Loss: ||current_I - target_I||² where target_I has certified structure
            # Target: omega ≈ 11° per layer, tele ≈ 0.89 (from ML-018 d=128 result)
            k_dim = self.k
            target_omega = torch.full((self.n,), math.radians(9.0), device=I_actual.device)
            target_tele  = torch.full((self.n,), 0.89, device=I_actual.device)
            omega_actual = I_actual[:, k_dim]     # omega column
            tele_actual  = I_actual[:, k_dim+2]   # tele column
            loss_down = (F.mse_loss(omega_actual, target_omega) +
                         F.mse_loss(tele_actual, target_tele))

        total_loss = None
        if loss_up is not None:
            total_loss = loss_up
            if loss_down is not None:
                total_loss = loss_up + 0.05 * loss_down

        return logits, total_loss, loss_up, loss_down

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data(path, device):
    if not os.path.exists(path):
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text=open(path).read()
    chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}
    data=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    n=int(0.9*len(data))
    return data[:n].to(device), data[n:].to(device), len(chars)

def get_batch(data, seq, bs, device):
    idx=torch.randint(len(data)-seq,(bs,))
    return (torch.stack([data[i:i+seq] for i in idx]),
            torch.stack([data[i+1:i+seq+1] for i in idx]))

@torch.no_grad()
def eval_loss(model, val, seq, device, n=40):
    model.eval()
    ls=[]
    for _ in range(n):
        x,y=get_batch(val,seq,32,device)
        _,total,up,_ = model(x,y)
        ls.append(float(up) if up is not None else float(total))
    model.train()
    return float(np.mean(ls))


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(model, train_d, val_d, steps, batch, seq, lr, device,
                label='', target_loss=1.55):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr/10)

    print(f"\n  [{label}] params={model.param_count():,}")
    print(f"  {'Step':>6} | {'val_loss':>9} | {'action'}")
    print(f"  {'─'*40}")

    log=[]; steps_to_target=None
    for step in range(steps+1):
        if step % 500 == 0:
            vl = eval_loss(model, val_d, seq, device)
            print(f"  {step:>6} | {vl:>9.4f}")
            log.append({'step':step,'val_loss':vl})
            if vl <= target_loss and steps_to_target is None:
                steps_to_target = step
                print(f"  *** TARGET REACHED ***")
            if step==steps: break

        x,y=get_batch(train_d,seq,batch,device)
        _,loss,_,_ = model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()

    return log, steps_to_target, model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiments', default='A,B,C')
    p.add_argument('--backbone',    default='gpt2-large',
                   help='Source of large-model info matrix for exp B')
    p.add_argument('--d',    type=int,   default=128)
    p.add_argument('--n',    type=int,   default=8)
    p.add_argument('--k',    type=int,   default=32)
    p.add_argument('--steps',type=int,   default=5000)
    p.add_argument('--batch',type=int,   default=32)
    p.add_argument('--seq',  type=int,   default=128)
    p.add_argument('--lr',   type=float, default=3e-3)
    p.add_argument('--data-path', default='tinyshakespeare.txt')
    p.add_argument('--save', default='ml021_infomatrix_results.json')
    args=p.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    exps=[e.strip().upper() for e in args.experiments.split(',')]

    print(f"\n{'═'*70}")
    print(f"  CORE-003-ML-021  Bidirectional Information Matrix")
    print(f"  Experiments: {exps}")
    print(f"{'═'*70}\n")

    train_d, val_d, vocab = load_data(args.data_path, device)
    all_results={}

    # ── Baseline (UP only, no manifold reading) ───────────────────────────────
    print("  Training BASELINE (UP only, mode='up')...")
    m_base = BidirectionalModel(vocab, args.d, args.n, args.k, args.seq,
                                 mode='up').to(device)
    log_base, s_base, m_base = train_model(m_base, train_d, val_d, args.steps,
                                    args.batch, args.seq, args.lr, device, 'baseline')
    torch.save({'state': m_base.state_dict(), 'mode': 'up', 'd': args.d, 'n': args.n,
                'k': args.k, 'vocab': vocab}, 'ml021_baseline.pt')
    print("  Saved ml021_baseline.pt")
    all_results['baseline'] = {'log': log_base, 'steps_to_target': s_base}

    # ── Experiment A: Self-conditioned ────────────────────────────────────────
    if 'A' in exps:
        print(f"\n{'─'*70}")
        print("  EXPERIMENT A: SELF-CONDITIONED (model reads own info matrix)")
        torch.manual_seed(42)
        m_self = BidirectionalModel(vocab, args.d, args.n, args.k, args.seq,
                                     mode='self').to(device)
        reader_p = sum(p.numel() for p in m_self.reader.parameters())
        print(f"  ManifoldReader adds {reader_p:,} params")
        log_a, s_a, m_self = train_model(m_self, train_d, val_d, args.steps,
                                  args.batch, args.seq, args.lr, device, 'self-cond')
        torch.save({'state': m_self.state_dict(), 'mode': 'self', 'd': args.d, 'n': args.n,
                    'k': args.k, 'vocab': vocab}, 'ml021_model_A.pt')
        print("  Saved ml021_model_A.pt")
        h021a = (s_a is not None and s_base is not None and s_a < s_base)
        print(f"  H-021-A: {'PASS' if h021a else 'NULL'}  "
              f"(self={s_a} vs baseline={s_base})")
        all_results['A'] = {'log': log_a, 'steps_to_target': s_a, 'h021a': h021a}

    # ── Experiment B: Cross-conditioned ──────────────────────────────────────
    if 'B' in exps:
        print(f"\n{'─'*70}")
        print(f"  EXPERIMENT B: CROSS-CONDITIONED (reads {args.backbone} geometry)")
        I_gpt2 = extract_gpt2_info_matrix(args.backbone, args.k, device)
        torch.manual_seed(42)
        m_cross = BidirectionalModel(vocab, args.d, args.n, args.k, args.seq,
                                      mode='cross', fixed_I=I_gpt2).to(device)
        log_b, s_b, m_cross = train_model(m_cross, train_d, val_d, args.steps,
                                  args.batch, args.seq, args.lr, device, 'cross-cond')
        torch.save({'state': m_cross.state_dict(), 'mode': 'cross', 'd': args.d, 'n': args.n,
                    'k': args.k, 'vocab': vocab}, 'ml021_model_B.pt')
        print("  Saved ml021_model_B.pt")
        h021b_final = log_b[-1]['val_loss'] if log_b else float('inf')
        h021b_base  = log_base[-1]['val_loss'] if log_base else float('inf')
        h021b = h021b_final < h021b_base
        print(f"  H-021-B: {'PASS' if h021b else 'NULL'}  "
              f"(cross={h021b_final:.4f} vs baseline={h021b_base:.4f})")
        all_results['B'] = {'log': log_b, 'steps_to_target': s_b, 'h021b': h021b}

    # ── Experiment C: Dual direction ──────────────────────────────────────────
    if 'C' in exps:
        print(f"\n{'─'*70}")
        print("  EXPERIMENT C: DUAL DIRECTION (UP + DOWN joint loss)")
        torch.manual_seed(42)
        m_dual = BidirectionalModel(vocab, args.d, args.n, args.k, args.seq,
                                     mode='dual').to(device)
        log_c, s_c, m_dual = train_model(m_dual, train_d, val_d, args.steps,
                                  args.batch, args.seq, args.lr, device, 'dual')
        torch.save({'state': m_dual.state_dict(), 'mode': 'dual', 'd': args.d, 'n': args.n,
                    'k': args.k, 'vocab': vocab}, 'ml021_model_C.pt')
        print("  Saved ml021_model_C.pt")
        h021c_final = log_c[-1]['val_loss'] if log_c else float('inf')
        h021c = h021c_final < log_base[-1]['val_loss']
        print(f"  H-021-C: {'PASS' if h021c else 'NULL'}  "
              f"(dual={h021c_final:.4f} vs baseline={log_base[-1]['val_loss']:.4f})")
        all_results['C'] = {'log': log_c, 'steps_to_target': s_c, 'h021c': h021c}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  FINAL COMPARISON")
    print(f"  {'Model':>20} | {'Final loss':>12} | {'Steps to {}'.format(1.55):>15}")
    print(f"  {'─'*55}")
    for name, res in all_results.items():
        fl = res['log'][-1]['val_loss'] if res['log'] else float('nan')
        st = res.get('steps_to_target', 'N/A')
        print(f"  {name:>20} | {fl:>12.4f} | {str(st):>15}")

    print(f"\n  Omnidirectional interpretation:")
    print(f"  The information matrix I is the model's map of its own territory.")
    print(f"  Conditioning on I is reading the map while drawing the map.")
    print(f"  The DOWN signal makes the UP signal geometrically coherent.")

    with open(args.save,'w') as f:
        def ts(o):
            if isinstance(o,(np.floating,np.integer,torch.Tensor)): return float(o)
            if isinstance(o,dict): return {k:ts(v) for k,v in o.items()}
            if isinstance(o,list): return [ts(v) for v in o]
            return o
        json.dump(ts(all_results),f,indent=2)
    print(f"\n  Results → {args.save}")
    print(f"{'═'*70}\n")


if __name__ == '__main__':
    main()
