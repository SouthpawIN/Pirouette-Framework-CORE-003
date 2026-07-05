"""
generate.py -- autoregressive generation for core v3 checkpoints, with
the address supplied any of three ways:

  --prompt "..."                    encoder mode: address computed FROM the
                                    prompt (exactly the training condition)
  --task pirouette                  table/blend mode: a registered address
  --hh J1,J2,Phi,E,Ksi              ANY explicit point on the manifold --
                                    including one found by find_address.py

Under --pos alibi/rope there is no positional ceiling; generation past
the training window is architecturally legal (quality still degrades
gracefully with distance under rope; alibi extrapolates best). --window
caps the attention context per step (memory/speed), it is not a
coherence cliff the way learned positions were.

Usage:
  python generate.py --ckpt model_checkpoint_v3.pt --prompt "The manifold" --n_chars 800
  python generate.py --ckpt model_checkpoint_v3.pt --task pirouette --n_chars 800
  python generate.py --ckpt model_checkpoint_v3.pt --prompt "..." --hh 0.244,0.0,1.558,0.05,0.1
  python generate.py --ckpt ... --prompt "..." --readdress_every 128   # let the address drift with the text
"""
import argparse
import torch
import torch.nn.functional as F
from core_runner_v3 import build_model_from_checkpoint, HH_DIMS


def sample_next(logits, temperature=0.8, top_k=40, top_p=0.0):
    logits = logits / max(temperature, 1e-6)
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cut = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(cut, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


def resolve_generation_hh(model, args, prompt_ids):
    if args.hh:
        vals = [float(v) for v in args.hh.split(',')]
        if len(vals) != 5:
            raise SystemExit("--hh needs 5 comma-separated values: J1,J2,Phi,E,Ksi")
        print(f"  address: explicit HH {dict(zip(HH_DIMS, vals))}")
        return torch.tensor(vals)
    if args.task:
        if args.task not in model.hh_by_task:
            raise SystemExit(f"--task {args.task} not registered in this checkpoint "
                             f"({len(model.hh_by_task)} known tasks)")
        hh = torch.stack([v.detach() for v in model.hh_by_task[args.task]()])
        print(f"  address: table['{args.task}'] = "
              f"{dict(zip(HH_DIMS, [round(float(v), 4) for v in hh]))}")
        return hh
    if model.moe is not None:
        if prompt_ids.numel() < 2:
            raise SystemExit("MoE routing needs a --prompt of at least a few characters.")
        with torch.no_grad():
            addrs = model.moe.encode_text(prompt_ids)   # [n_layers, 5] per-layer address
        mean_addr = addrs.mean(dim=0)
        print(f"  address: per-layer MoE mixture, {addrs.shape[0]} layers "
              f"(mean = {dict(zip(HH_DIMS, [round(float(v), 4) for v in mean_addr]))})")
        return addrs.detach()     # per-layer matrix; forward_with_hh applies row-per-layer
    if model.registers is not None:
        with torch.no_grad():
            pts = model.registers.register_points()
            if args.register is not None:
                if not (0 <= args.register < model.registers.n_registers):
                    raise SystemExit(f"--register must be in [0, {model.registers.n_registers})")
                hh = pts[args.register].detach()
                print(f"  address: register R{args.register} = "
                      f"{dict(zip(HH_DIMS, [round(float(v), 4) for v in hh]))}")
                return hh
            if prompt_ids.numel() < 2:
                raise SystemExit("Register routing needs a --prompt of at least a few characters.")
            hh, logits = model.registers(prompt_ids.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
        print(f"  address: routed mixture "
              f"{[round(float(p), 3) for p in probs]} over {len(pts)} registers = "
              f"{dict(zip(HH_DIMS, [round(float(v), 4) for v in hh]))}")
        return hh.detach()
    if model.encoder is None:
        raise SystemExit("This checkpoint has no encoder or registers -- supply --task or --hh.")
    if prompt_ids.numel() < 2:
        raise SystemExit("Encoder addressing needs a --prompt of at least a few characters.")
    with torch.no_grad():
        hh = model.encoder.encode_text(prompt_ids)
    print(f"  address: encoder(prompt) = "
          f"{dict(zip(HH_DIMS, [round(float(v), 4) for v in hh]))}")
    return hh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--prompt', default='\n')
    p.add_argument('--task', default=None)
    p.add_argument('--hh', default=None, help="J1,J2,Phi,E,Ksi -- explicit manifold point")
    p.add_argument('--register', type=int, default=None,
                   help="registers mode: generate from register k's own point "
                        "instead of routing the prompt")
    p.add_argument('--n_chars', type=int, default=600)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=40)
    p.add_argument('--top_p', type=float, default=0.0)
    p.add_argument('--window', type=int, default=1024,
                   help="attention context cap per step (memory/speed knob)")
    p.add_argument('--readdress_every', type=int, default=0,
                   help="encoder mode only: re-encode the address from the last "
                        "window every N generated chars (0 = fix it from the prompt)")
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, vocab_chars, stoi = build_model_from_checkpoint(ckpt, device)
    print(f"Checkpoint: {args.ckpt}  pos={model.pos_mode} address={model.address_mode} "
          f"d={model.d} n={model.n} k={model.k}")
    if model.pos_mode == 'learned':
        print(f"  NOTE: learned positions -- expect coherence to degrade near "
              f"seq_len={model.seq_len} (the certified v2 ceiling). alibi/rope checkpoints "
              f"don't have this cliff.")

    unknown = [c for c in args.prompt if c not in stoi]
    if unknown:
        print(f"  WARNING: prompt characters not in vocab, dropped: {sorted(set(unknown))!r}")
    ids = torch.tensor([stoi[c] for c in args.prompt if c in stoi],
                       dtype=torch.long, device=device)
    if ids.numel() == 0:
        ids = torch.tensor([stoi[vocab_chars[0]]], dtype=torch.long, device=device)

    hh = resolve_generation_hh(model, args, ids).to(device)

    out = ids.tolist()
    with torch.no_grad():
        for i in range(args.n_chars):
            ctx = torch.tensor(out[-args.window:], dtype=torch.long,
                               device=device).unsqueeze(0)
            if model.pos_mode == 'learned':
                ctx = ctx[:, -model.seq_len:]
            logits, _ = model.forward_with_hh(ctx, hh)
            nxt = sample_next(logits[0, -1], args.temperature, args.top_k, args.top_p)
            out.append(int(nxt))
            if (args.readdress_every and model.encoder is not None
                    and (i + 1) % args.readdress_every == 0):
                recent = torch.tensor(out[-model.encoder.enc_ctx:], dtype=torch.long,
                                      device=device)
                hh = model.encoder.encode_text(recent).to(device)

    text = ''.join(vocab_chars[i] for i in out)
    print("\n" + "─" * 70)
    print(text)
    print("─" * 70)


if __name__ == '__main__':
    main()
