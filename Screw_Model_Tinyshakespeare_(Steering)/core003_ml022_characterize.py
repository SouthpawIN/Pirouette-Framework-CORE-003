"""
CORE-003-ML-022  ·  Model Characterization & Generation Suite
Pirouette Framework Volume 8  ·  Keaton Smith

Loads saved ML-021 models and runs four characterization experiments:

A: GENERATION COMPARISON
   Generate text from baseline, Model A (self-conditioned), Model B (cross-conditioned).
   Same prompt, same temperature — compare coherence, variety, and content.
   The "funkiness" test: does the limit-cycle attractor produce richer text?

B: OUT-OF-DOMAIN PERPLEXITY
   Evaluate all models on text that was NOT in training:
   - A paragraph from a science article
   - A legal contract snippet
   - A mathematical proof
   Hypothesis: Model A (wider basin, geometric attractor) generalizes better.

C: MANIFOLDREADER BIAS ANALYSIS
   Extract the bias vector the ManifoldReader adds at each layer.
   Compute alignment with:
   - The embedding matrix top singular vector (language prior direction)
   - The certified attractor direction (from ML-011-A)
   - Random baseline
   If the bias points toward the certified attractor: ManifoldReader discovered
   the geometry independently without being told about it.

D: EXTENDED LIMIT CYCLE
   Load Model A and continue training for 10,000 more steps.
   Plot loss trajectory to characterize:
   - Stable limit cycle (fixed period, stable amplitude)
   - Decaying oscillation (spiral into fixed point)
   - Growing oscillation (spiral away — instability)
   - Chaotic (non-repeating) 

Usage:
  python core003_ml022_characterize.py --experiments A,B,C,D
  python core003_ml022_characterize.py --experiments A        # just generation
  python core003_ml022_characterize.py --experiments D --extra-steps 10000
"""

import argparse, json, math, os, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Out-of-domain test passages (deliberately NOT Shakespeare)
OOD_PASSAGES = {
    'science': (
        "The mitochondria are membrane-bound organelles found in the cytoplasm of "
        "eukaryotic cells. They generate most of the cell's supply of adenosine "
        "triphosphate, used as a source of chemical energy. The organelle is often "
        "colloquially called the powerhouse of the cell."
    ),
    'legal': (
        "This Agreement is entered into as of the date last signed below between "
        "the parties hereto. Each party represents that it has full power and "
        "authority to enter into this Agreement and to perform its obligations "
        "hereunder. The terms and conditions set forth herein shall be binding."
    ),
    'mathematical': (
        "Let f be a continuous function on the closed interval from a to b. "
        "If f is differentiable on the open interval from a to b, then there "
        "exists a point c in that open interval such that the derivative of f "
        "at c equals the average rate of change of f over the entire interval."
    ),
}

# Prompts for generation comparison
PROMPTS = [
    "KING:\nWhat is the nature of",
    "First Citizen:\nWe are",
    "HAMLET:\nTo be or not",
]


# ─── Paste model classes from ML-021 (minimal versions for loading) ───────────

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

class ToruscScrew(nn.Module):
    def __init__(self, d_model, d_ff, k, n_layers):
        super().__init__()
        self.k=k; self.n_layers=n_layers
        for name, shape in [('u0',(d_model,k)),('v0',(d_model,k)),
                              ('n0',(d_model,k)),('m0',(d_model,k)),('V',(d_ff,k))]:
            p = torch.randn(*shape); p,_=torch.linalg.qr(p)
            setattr(self, name, nn.Parameter(p[:,:k]))
        self.omega=nn.Parameter(torch.tensor(math.radians(14.0)))
        self.theta_base=nn.Parameter(torch.tensor(math.radians(89.3)))
        self.delta_theta=nn.Parameter(torch.zeros(n_layers,k))
        self.log_sigma=nn.Parameter(torch.zeros(n_layers,k))
        self.spin_gen=nn.Parameter(torch.zeros(n_layers,k*(k-1)//2)*0.01)

    def get_U_L(self, L):
        phi=self.omega*L
        u_L=F.normalize(torch.cos(phi)*self.u0+torch.sin(phi)*self.v0,dim=0)
        n_L=torch.cos(phi)*self.n0+torch.sin(phi)*self.m0
        n_L=F.normalize(n_L-(n_L*u_L).sum(0,keepdim=True)*u_L,dim=0)
        th=self.theta_base+self.delta_theta[L]
        return (torch.cos(th)*u_L+torch.sin(th)*n_L)@cayley(skew(self.spin_gen[L],self.k)).T

    def forward(self, h, L, bias=None):
        out=(h@self.V*torch.exp(self.log_sigma[L]))@self.get_U_L(L).T
        if bias is not None: out=out+bias.unsqueeze(0).unsqueeze(0)
        return out

class ManifoldReader(nn.Module):
    def __init__(self, n_layers, k, d_model, hidden=64):
        super().__init__()
        i_dim=n_layers*(k+4)
        self.flatten=nn.Flatten()
        self.net=nn.Sequential(nn.Linear(i_dim,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),
                               nn.Linear(hidden,d_model))
        self.layer_scales=nn.Parameter(torch.zeros(n_layers))

    def forward(self, I, layer_idx):
        flat=self.flatten(I.unsqueeze(0))
        bias=self.net(flat).squeeze(0)
        return bias*torch.sigmoid(self.layer_scales[layer_idx])

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

def extract_info_matrix(screws, device):
    rows=[]
    n=len(screws)
    for L,screw in enumerate(screws):
        log_sigma=screw.log_sigma[L].detach()
        omega=torch.tensor([float(screw.omega)],device=device)
        theta=torch.tensor([float(screw.theta_base)],device=device)
        if L<n-1:
            def gu(s,l):
                phi=s.omega*l
                return F.normalize(F.normalize(torch.cos(phi)*s.u0+torch.sin(phi)*s.v0,dim=0)[:,0],dim=0)
            cos_v=(gu(screw,L)*gu(screws[L+1],L+1)).sum().clamp(-1,1)
            angle=torch.acos(cos_v.abs()).unsqueeze(0)
        else: angle=torch.zeros(1,device=device)
        if L<n-3:
            def gu2(s,l):
                phi=s.omega*l
                return F.normalize(F.normalize(torch.cos(phi)*s.u0+torch.sin(phi)*s.v0,dim=0)[:,0],dim=0)
            tele=(gu2(screw,L)*gu2(screws[L+3],L+3)).sum().abs().unsqueeze(0)
        else: tele=torch.zeros(1,device=device)
        rows.append(torch.cat([log_sigma,omega,angle,tele,theta]))
    return torch.stack(rows,dim=0)

class BidirectionalModel(nn.Module):
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
        if mode in ('self','cross','dual'): self.reader=ManifoldReader(n,k,d)
        else: self.reader=None
        if fixed_I is not None: self.register_buffer('fixed_I',fixed_I)
        else: self.fixed_I=None

    def get_info_matrix(self, device):
        if self.mode=='cross' and self.fixed_I is not None:
            I=self.fixed_I
            if I.shape[0]!=self.n:
                factor=I.shape[0]/self.n
                rows=[I[int(L*factor):int((L+1)*factor)].mean(0) for L in range(self.n)]
                I=torch.stack(rows,0)
            k_i=I.shape[1]-4
            if k_i>self.k: I=torch.cat([I[:,:self.k],I[:,k_i:]],dim=1)
            elif k_i<self.k:
                pad=torch.zeros(I.shape[0],self.k-k_i,device=I.device)
                I=torch.cat([I[:,:k_i],pad,I[:,k_i:]],dim=1)
            return I.to(next(self.parameters()).device)
        return extract_info_matrix(self.screws, next(self.parameters()).device)

    def forward(self, idx, targets=None):
        B,T=idx.shape
        x=self.drop(self.embed(idx)+self.pos(torch.arange(T,device=idx.device)))
        biases=[None]*self.n
        if self.reader is not None:
            I=self.get_info_matrix(idx.device)
            for L in range(self.n): biases[L]=self.reader(I,L)
        for L,(attn,c_fc,screw,ln1,ln2) in enumerate(zip(
            self.attn,self.c_fc,self.screws,self.ln1,self.ln2)):
            x=x+attn(ln1(x)); x=x+screw(F.gelu(c_fc(ln2(x))),L,biases[L])
        logits=self.head(self.ln_f(x))
        loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) \
             if targets is not None else None
        return logits,loss,loss,None

    def get_reader_biases(self):
        """Extract ManifoldReader bias vectors at current weight state."""
        if self.reader is None: return None
        device=next(self.parameters()).device
        I=self.get_info_matrix(device)
        with torch.no_grad():
            return [self.reader(I, L).cpu() for L in range(self.n)]


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_model(path, device, seq_len=128):
    if not os.path.exists(path):
        print(f"  {path} not found — run ml021 first")
        return None, None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    d, n, k = ckpt['d'], ckpt['n'], ckpt['k']
    vocab    = ckpt['vocab']
    mode     = ckpt.get('mode', 'up')
    
    # Extract the state dict
    state = ckpt['state']
    
    # FIX: Pull fixed_I out of the state dict if it exists, so the model can register it
    fixed_I = state.get('fixed_I', None)
    
    # Pass fixed_I into the instantiation
    model = BidirectionalModel(vocab, d, n, k, seq_len, mode=mode, fixed_I=fixed_I).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, vocab, mode


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_shakespeare(path, device):
    if not os.path.exists(path): urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text=open(path).read()
    chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}
    itos={i:c for c,i in stoi.items()}
    return stoi, itos

def get_batch_from_text(text, stoi, seq, bs, device):
    """Create batch from raw text string."""
    ids=[stoi.get(c,0) for c in text]
    if len(ids)<seq+1: return None,None
    toks=torch.tensor(ids,dtype=torch.long)
    chunks=[toks[i:i+seq+1] for i in range(0,len(toks)-seq,seq//2)]
    if not chunks: return None,None
    idx=torch.randint(len(chunks),(min(bs,len(chunks)),))
    b=torch.stack([chunks[i] for i in idx])
    return b[:,:-1].to(device), b[:,1:].to(device)

@torch.no_grad()
def compute_perplexity(model, text, stoi, seq, device, n_batches=20):
    """Compute perplexity on arbitrary text."""
    model.eval()
    losses=[]
    for _ in range(n_batches):
        x,y=get_batch_from_text(text, stoi, seq, 8, device)
        if x is None: break
        _,_,loss_up,_ = model(x,y)
        if loss_up is not None: losses.append(float(loss_up))
    return math.exp(np.mean(losses)) if losses else float('nan')


# ─── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, seed_ids, n, temperature, itos, seq_len, device):
    model.eval()
    ids=seed_ids.to(device)
    for _ in range(n):
        logits,_,_,_ = model(ids[:,-seq_len:])
        next_id=torch.multinomial(F.softmax(logits[:,-1,:]/temperature,-1),1)
        ids=torch.cat([ids,next_id],dim=1)
    return ''.join(itos[int(i)] for i in ids[0][seed_ids.shape[1]:])


# ─── Sub-experiments ──────────────────────────────────────────────────────────

def exp_A_generation(models_dict, stoi, itos, device, seq_len, n_tokens=300):
    """Generate text from all models with same prompts. The funkiness test."""
    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT A: GENERATION COMPARISON")
    print(f"  {n_tokens} tokens per model per prompt, T=0.8")
    print(f"{'═'*70}")

    results={}
    for temp in [0.7, 1.0]:
        print(f"\n  ── Temperature = {temp} ──────────────────────────────────────")
        prompt = PROMPTS[0]  # "KING:\nWhat is the nature of"

        print(f"  Prompt: \"{prompt}\"")
        print()

        for model_name, (model, vocab, mode) in models_dict.items():
            if model is None: continue
            seed_ids=torch.tensor([[stoi.get(c,0) for c in prompt]],dtype=torch.long)
            text=generate(model, seed_ids, n_tokens, temp, itos, seq_len, device)
            results[f'{model_name}_T{temp}']=text
            print(f"  ┌─ {model_name} (mode={mode}) T={temp} ────────────────────────────────")
            for i in range(0,min(len(text),200),65):
                print(f"  │ {text[i:i+65]}")
            print(f"  └──────────────────────────────────────────────────────────────")
            print()

    return results


def exp_B_ood(models_dict, stoi, device, seq_len):
    """Out-of-domain perplexity. Does the limit-cycle model generalize better?"""
    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT B: OUT-OF-DOMAIN PERPLEXITY")
    print(f"  (Lower = model handles text outside Shakespeare training better)")
    print(f"{'═'*70}")

    results={}
    # Build extended charset from OOD passages
    all_chars=set()
    for text in OOD_PASSAGES.values():
        all_chars.update(text)

    print(f"\n  {'Model':>20} | ", end='')
    print(' | '.join(f'{k:>12}' for k in OOD_PASSAGES.keys()))
    print(f"  {'─'*70}")

    for model_name, (model, vocab, mode) in models_dict.items():
        if model is None: continue
        row=[f'{model_name:>20}']
        ppls={}
        for domain, text in OOD_PASSAGES.items():
            ppl = compute_perplexity(model, text, stoi, seq_len, device)
            ppls[domain]=ppl
            row.append(f'{ppl:>12.1f}')
        print(f"  {'  |  '.join(row)}")
        results[model_name]=ppls

    return results


def exp_C_bias_analysis(models_dict, device):
    """What direction is the ManifoldReader pushing?"""
    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT C: MANIFOLDREADER BIAS ANALYSIS")
    print(f"  Does the bias point toward the certified attractor direction?")
    print(f"{'═'*70}")

    results={}
    for model_name, (model, vocab, mode) in models_dict.items():
        if model is None or model.reader is None:
            print(f"\n  {model_name}: no reader (mode={mode})")
            continue

        biases = model.get_reader_biases()
        if biases is None: continue

        print(f"\n  {model_name} (mode={mode})")
        # Bias magnitude per layer
        mags=[float(b.norm()) for b in biases]
        print(f"  Bias magnitudes: {[f'{m:.3f}' for m in mags]}")

        # Bias direction consistency: do consecutive layers agree?
        cosines=[]
        for i in range(len(biases)-1):
            b1=F.normalize(biases[i], dim=0)
            b2=F.normalize(biases[i+1], dim=0)
            cosines.append(float((b1*b2).sum()))
        print(f"  Inter-layer cosines: {[f'{c:.3f}' for c in cosines]}")
        print(f"  Mean: {np.mean(cosines):.3f} (1.0=all pointing same direction)")

        # Does bias point toward embedding dominant direction?
        embed_weight = model.embed.weight.detach().cpu()
        _,_,Vh = torch.linalg.svd(embed_weight, full_matrices=False)
        dominant = F.normalize(Vh[0], dim=0)

        align=[float((F.normalize(b,dim=0)*dominant).sum()) for b in biases]
        print(f"  Alignment with embedding top SV: {[f'{a:.3f}' for a in align]}")
        print(f"  Mean alignment: {np.mean(align):.3f}")
        print(f"  (>0.1 = bias consistently points toward language prior direction)")

        results[model_name]={'magnitudes': mags, 'inter_cosines': cosines,
                             'embed_alignment': align}

    return results


def exp_D_extended(model_path, train_d_path, device, extra_steps, seq_len=128, batch=32, lr=3e-3):
    """Extend Model A training to characterize the limit cycle."""
    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT D: EXTENDED LIMIT CYCLE ({extra_steps} more steps)")
    print(f"{'═'*70}")

    model, vocab, mode = load_model(model_path, device, seq_len)
    if model is None: return {}

    stoi, itos = load_shakespeare(train_d_path, device)
    text=open(train_d_path).read()
    chars=sorted(set(text)); stoi2={c:i for i,c in enumerate(chars)}
    data=torch.tensor([stoi2[c] for c in text],dtype=torch.long)
    n=int(0.9*len(data))
    train_d2, val_d2 = data[:n].to(device), data[n:].to(device)

    def get_b(d):
        idx=torch.randint(len(d)-seq_len,(batch,))
        return (torch.stack([d[i:i+seq_len] for i in idx]),
                torch.stack([d[i+1:i+seq_len+1] for i in idx]))

    opt=torch.optim.AdamW(model.parameters(), lr=lr*0.3, weight_decay=0.1)  # lower lr for extension

    print(f"  {'Step':>8} | {'val_loss':>9}")
    print(f"  {'─'*25}")
    log=[]
    model.train()
    for step in range(extra_steps+1):
        if step%500==0:
            model.eval()
            with torch.no_grad():
                ls=[float(model(*get_b(val_d2))[:2][1]) for _ in range(40)]
            vl=float(np.mean(ls))
            print(f"  {step:>8} | {vl:>9.4f}")
            log.append({'step':step,'val_loss':vl})
            model.train()
            if step==extra_steps: break
        x,y=get_b(train_d2)
        _,loss,_,_=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

    # Characterize: is the late-stage oscillation periodic?
    late=[e['val_loss'] for e in log[-10:]]
    print(f"\n  Late trajectory (last 5000 steps):")
    print(f"  {[f'{v:.4f}' for v in late]}")
    late_std=float(np.std(late))
    late_range=max(late)-min(late)
    verdict=('STABLE LIMIT CYCLE' if late_std>0.005 else
             'DECAYING (spiraling in)' if late_std<0.003 else 'MARGINAL')
    print(f"  Std: {late_std:.5f}  Range: {late_range:.4f}")
    print(f"  Verdict: {verdict}")

    return {'log': log, 'late_std': late_std, 'verdict': verdict}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiments',  default='A,B,C,D')
    p.add_argument('--baseline',     default='ml021_baseline.pt')
    p.add_argument('--model-a',      default='ml021_model_A.pt')
    p.add_argument('--model-b',      default='ml021_model_B.pt')
    p.add_argument('--extra-steps',  type=int, default=5000)
    p.add_argument('--seq',          type=int, default=128)
    p.add_argument('--data-path',    default='tinyshakespeare.txt')
    p.add_argument('--save',         default='ml022_characterize_results_2.json')
    args=p.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    exps=[e.strip().upper() for e in args.experiments.split(',')]

    print(f"\n{'═'*70}")
    print(f"  CORE-003-ML-022  Model Characterization & Generation Suite")
    print(f"  Experiments: {exps}")
    print(f"{'═'*70}\n")

    # Load vocabulary
    stoi, itos = load_shakespeare(args.data_path, device)

    # Load models
    print("  Loading models...")
    models={}
    for name, path in [('baseline', args.baseline),
                       ('model_A',  args.model_a),
                       ('model_B',  args.model_b)]:
        m, vocab, mode = load_model(path, device, args.seq)
        if m: print(f"  {name}: {path} ({mode} mode)")
        models[name]=(m, vocab, mode)

    all_results={}

    if 'A' in exps:
        all_results['A']=exp_A_generation(models, stoi, itos, device, args.seq)

    if 'B' in exps:
        all_results['B']=exp_B_ood(models, stoi, device, args.seq)

    if 'C' in exps:
        all_results['C']=exp_C_bias_analysis(models, device)

    if 'D' in exps:
        all_results['D']=exp_D_extended(args.model_a, args.data_path,
                                         device, args.extra_steps, args.seq)

    # Final generation with altruism prompt
    print(f"\n{'═'*70}")
    print(f"  ALTRUISM SIGNAL (the original intelligence test)")
    print(f"{'═'*70}")
    altruism_prompt = "What is altruism?"
    for name,(model,vocab,mode) in models.items():
        if model is None: continue
        seed=torch.tensor([[stoi.get(c,0) for c in altruism_prompt]],dtype=torch.long)
        text=generate(model, seed, 200, 0.8, itos, args.seq, device)
        field={'love','give','other','sacrifice','duty','honour','soul','heart',
               'good','mercy','kind','virtue','god','country','nature'}
        found=sorted({w.strip('.,;:\'"').lower() for w in text.split()}&field)
        print(f"\n  [{name}] field_words({len(found)}): {found}")
        print(f"  ┌────────────────────────────────────────────────────────────")
        for i in range(0,min(len(text),200),65):
            print(f"  │ {text[i:i+65]}")
        print(f"  └────────────────────────────────────────────────────────────")

    with open(args.save,'w') as f:
        def ts(o):
            if isinstance(o,(float,int,bool)): return o
            if isinstance(o,dict): return {k:ts(v) for k,v in o.items()}
            if isinstance(o,list): return [ts(v) for v in o]
            if isinstance(o,torch.Tensor): return o.tolist()
            if isinstance(o,np.ndarray): return o.tolist()
            return str(o)
        json.dump(ts(all_results),f,indent=2)
    print(f"\n  Results → {args.save}")


if __name__=='__main__':
    main()
