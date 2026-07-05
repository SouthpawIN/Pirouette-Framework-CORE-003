"""
check_encoder_diversity.py -- automates the readdress_every diagnosis:
does model.encoder.encode_text() actually vary with content, or has it
collapsed to a near-constant output?

For each source in --texts, samples --n_samples random windows and encodes
each independently. Reports:
  - within-source mean pairwise distance (should be small-ish: same source,
    different windows -- some spread is fine, this isn't the interesting
    number)
  - cross-source mean pairwise distance (samples from DIFFERENT sources --
    THIS is the number that matters)
  - the ratio cross/within: near 1.0 means the encoder can't tell sources
    apart at all (collapse); notably > 1 means real content-dependence.

Run this on a baseline checkpoint (e.g. gen_v4.pt) and again on a
contrastive-trained checkpoint (tasks_lm_contrastive.py) with identical
--texts/--n_samples/--seed, and compare the ratio directly -- that's the
pre-registered pass/fail number for whether the contrastive term worked.

USAGE:

  python check_encoder_diversity.py --ckpt gen_v4.pt \
      --texts pirouette_corpus_clean.txt+combined_classics.txt \
      --n_samples 15 --out diversity_gen_v4.json

  python check_encoder_diversity.py --ckpt gen_v5_contrastive.pt \
      --texts pirouette_corpus_clean.txt+combined_classics.txt \
      --n_samples 15 --out diversity_gen_v5.json
"""
import argparse
import itertools
import json
import torch

from core_runner_v3 import build_model_from_checkpoint, HH_DIMS
from find_address import load_texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--texts', required=True, help="path1.txt+path2.txt")
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--n_samples', type=int, default=15, help="random windows per source")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='encoder_diversity.json')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, vocab_chars, stoi = build_model_from_checkpoint(ckpt, device)
    model.eval()
    print(f"Checkpoint: {args.ckpt}  address={model.address_mode}")
    if model.encoder is None:
        raise SystemExit("This checkpoint has no encoder -- nothing to measure.")

    texts = load_texts(args.texts, stoi, device, args.seq_len)
    print(f"Sources: {[t for t, _ in texts]}")

    hh_by_source = {}
    for name, ids in texts:
        hhs = []
        for _ in range(args.n_samples):
            i = torch.randint(0, len(ids) - args.seq_len, (1,)).item()
            window = ids[i:i + args.seq_len]
            with torch.no_grad():
                hh = model.encoder.encode_text(window)
            hhs.append(hh)
        hh_by_source[name] = hhs
        mean_hh = torch.stack(hhs).mean(dim=0)
        std_hh = torch.stack(hhs).std(dim=0)
        print(f"  {name}: mean={dict(zip(HH_DIMS, [round(float(v),4) for v in mean_hh]))}")
        print(f"    within-source std per dim={dict(zip(HH_DIMS, [round(float(v),4) for v in std_hh]))}")

    within_dists, cross_dists = [], []
    for name, hhs in hh_by_source.items():
        for a, b in itertools.combinations(hhs, 2):
            within_dists.append(float(torch.norm(a - b)))
    names = list(hh_by_source.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a in hh_by_source[names[i]]:
                for b in hh_by_source[names[j]]:
                    cross_dists.append(float(torch.norm(a - b)))

    mean_within = sum(within_dists) / len(within_dists) if within_dists else 0.0
    mean_cross = sum(cross_dists) / len(cross_dists) if cross_dists else 0.0
    ratio = mean_cross / mean_within if mean_within > 1e-9 else float('inf')

    summary = {
        'mean_within_source_distance': round(mean_within, 5),
        'mean_cross_source_distance': round(mean_cross, 5),
        'cross_to_within_ratio': round(ratio, 3) if ratio != float('inf') else 'inf (within-source distance ~0)',
        'note': ("ratio near 1.0 means the encoder cannot distinguish sources at all -- "
                "collapse. Notably above 1 (e.g. >1.5-2x) means real, usable "
                "content-dependence. Compare this ratio directly against a baseline "
                "checkpoint run with identical --texts/--n_samples/--seed."),
    }

    out = {
        'ckpt': args.ckpt, 'sources': names, 'n_samples_per_source': args.n_samples,
        'per_source_hh': {
            name: [dict(zip(HH_DIMS, [round(float(v), 6) for v in hh])) for hh in hhs]
            for name, hhs in hh_by_source.items()
        },
        'summary': summary,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nFull results -> {args.out}")


if __name__ == '__main__':
    main()
