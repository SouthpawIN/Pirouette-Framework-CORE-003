"""
map_manifold.py -- the ATLAS: idea 1 in its brute-force form. Instead of
asking "what is THE best address" (find_address.py), chart the whole box:
sample many HH points, score EVERY point against EVERY domain, and report
each point's full profile. From the atlas you get, for free:

  - SPECIALISTS: for each domain, the sampled point with the best
    'specialization score' -- own-domain CE minus mean other-domain CE.
    This is idea 2's "explains the content" as a search criterion: a
    register isn't just low-loss, it's DIAGNOSTICALLY low-loss on its
    domain (comparative advantage), which is what makes it worth routing
    to. A point that's equally good everywhere is a generalist, not a
    register.
  - GENERALISTS: the point with the best mean CE across the whole basket
    (find_address.py's objective, now with per-domain visibility).
  - THE SHAPE OF THE MANIFOLD: dump the whole (point, profile) table and
    you have exactly the dataset "forward every weight address to learn
    where it belongs" asks for -- suitable for plotting, clustering, or
    seeding a RegisterBank (see tasks_registers.py; init-from-atlas is a
    natural follow-up once atlas structure looks real).

Sampled points: a quasi-random scrambled-Sobol fill of the bounded box
(much better 5-dim coverage than uniform random at small n), plus every
registered table address, every register-bank point, and encoder outputs
on a few windows per domain if the checkpoint has those components.

COST: n_points x n_domains x score_batches forward passes, forward-only,
frozen backbone. 256 points x 3 domains x 4 batches at seq_len 256 /
batch 16 is a couple of minutes on a 4070 Ti.

USAGE:
  python map_manifold.py --ckpt gen_v5_contrastive.pt \
      --texts pirouette_corpus_clean.txt+combined_classics.txt \
      --n_points 256 --out atlas_v5.json

  # then generate from any specialist/generalist it prints:
  python generate.py --ckpt gen_v5_contrastive.pt --prompt "..." --hh <point>
"""
import argparse, json
import torch
import torch.nn.functional as F

from core_runner_v3 import (build_model_from_checkpoint, HH_DEFAULT, HH_RANGE,
                            HH_DIMS)
from find_address import load_texts, minibatch


@torch.no_grad()
def score_point_per_domain(model, hh, texts, seq_len, batch, n_batches, device, seed):
    g = torch.Generator(device=device); g.manual_seed(seed)
    out = {}
    for name, ids in texts:
        total = 0.0
        for _ in range(n_batches):
            x, y = minibatch([(name, ids)], seq_len, batch, device, generator=g)
            logits, _ = model.forward_with_hh(x, hh)
            total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           y.reshape(-1)))
        out[name] = total / n_batches
    return out


def sobol_points(n, seed):
    eng = torch.quasirandom.SobolEngine(dimension=5, scramble=True, seed=seed)
    u = eng.draw(n)                                   # [n, 5] in [0,1)
    return HH_DEFAULT + (2.0 * u - 1.0) * HH_RANGE    # fill the bounded box


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--texts', required=True, help="path1.txt+path2.txt -- the domains")
    p.add_argument('--n_points', type=int, default=256)
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--score_batches', type=int, default=4)
    p.add_argument('--enc_samples', type=int, default=3,
                   help="encoder/register-routed points sampled per domain, if available")
    p.add_argument('--top', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='atlas.json')
    args = p.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, vocab_chars, stoi = build_model_from_checkpoint(ckpt, device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    texts = load_texts(args.texts, stoi, device, args.seq_len)
    domains = [t for t, _ in texts]
    print(f"Checkpoint: {args.ckpt}  domains: {domains}")

    # ---- assemble the point set ----
    points = [('default', HH_DEFAULT.clone().to(device))]
    for name, mod in model.hh_by_task.items():
        points.append((f'table:{name}', torch.stack([v.detach() for v in mod()]).to(device)))
    if model.registers is not None:
        for i, pt in enumerate(model.registers.register_points()):
            points.append((f'register:R{i}', pt.detach().to(device)))
    if model.encoder is not None:
        for name, ids in texts:
            for s in range(args.enc_samples):
                i = torch.randint(0, max(1, len(ids) - args.seq_len), (1,)).item()
                hh = model.encoder.encode_text(ids[i:i + args.seq_len]).detach()
                points.append((f'encoder:{name}#{s}', hh.to(device)))
    for i, pt in enumerate(sobol_points(args.n_points, args.seed)):
        points.append((f'sobol{i}', pt.to(device)))
    print(f"scoring {len(points)} points x {len(domains)} domains "
          f"x {args.score_batches} batches...")

    # ---- score ----
    rows = []
    for j, (label, hh) in enumerate(points):
        profile = score_point_per_domain(model, hh, texts, args.seq_len,
                                         args.batch, args.score_batches,
                                         device, args.seed)
        mean_ce = sum(profile.values()) / len(profile)
        row = {'label': label, 'hh': [float(v) for v in hh],
               'profile': {k: round(v, 4) for k, v in profile.items()},
               'mean_ce': round(mean_ce, 4)}
        # specialization score per domain: how much BETTER this point is on
        # that domain than its own average elsewhere (positive = specialist)
        if len(domains) > 1:
            row['specialization'] = {
                d: round((sum(v for k, v in profile.items() if k != d)
                          / (len(profile) - 1)) - profile[d], 4)
                for d in domains}
        rows.append(row)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(points)}")

    # ---- report ----
    print("\n" + "═" * 74)
    by_mean = sorted(rows, key=lambda r: r['mean_ce'])
    print(f"GENERALISTS (best mean CE across {domains}):")
    for r in by_mean[:args.top]:
        print(f"  {r['mean_ce']:.4f}  {r['label']:<24} "
              f"{dict(zip(HH_DIMS, [round(v, 3) for v in r['hh']]))}")
        print(f"          profile: {r['profile']}")

    if len(domains) > 1:
        for d in domains:
            by_spec = sorted(rows, key=lambda r: -r['specialization'][d])
            print(f"\nSPECIALISTS for {d} (own-domain advantage, idea-2 criterion):")
            for r in by_spec[:args.top]:
                print(f"  adv {r['specialization'][d]:+.4f}  own {r['profile'][d]:.4f}  "
                      f"{r['label']:<24} "
                      f"{dict(zip(HH_DIMS, [round(v, 3) for v in r['hh']]))}")

    best = by_mean[0]
    hh_str = ','.join(f"{v:.6f}" for v in best['hh'])
    print(f"\ngenerate from the best generalist:")
    print(f"  python generate.py --ckpt {args.ckpt} --prompt \"...\" --hh {hh_str}")

    with open(args.out, 'w') as f:
        json.dump({'ckpt': args.ckpt, 'domains': domains,
                   'seq_len': args.seq_len, 'atlas': rows}, f, indent=2)
    print(f"\nFull atlas ({len(rows)} points) -> {args.out}")


if __name__ == '__main__':
    main()
