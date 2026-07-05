# The Addressed-Component Model — Registers, Atlas, and the Governed Colosseum

This layer builds on core v3 and turns your three ideas into four components that share one premise: the HH box is a weight pool, and the system's job is to *chart it, name regions of it, and compose from it*.

## How the ideas map to components

**Idea 1 — "forward every address to learn where it belongs" — has a brute-force form and a learned form.** The brute-force form is `map_manifold.py`, the atlas: a scrambled-Sobol fill of the bounded box plus every named point (table addresses, register points, encoder outputs), each scored against *every* domain, producing a full (address → per-domain CE profile) table. The learned form is the redesigned colosseum: a Solver that must identify an address from the behavior it produces has, when it wins, internalized exactly that map — behavior → address — over the whole box, not just the points training data visits. `infer_address()` exposes it directly.

**Idea 2 — "loss down AND explains the content" — becomes an identifiability objective, in two guises.** In search form (the atlas), it's the specialization score: a register-worthy point isn't just low-loss on its domain, it has *comparative advantage* there — own-domain CE minus mean other-domain CE. In training form (`tasks_registers.py`), it's a classification pressure on the router: every training batch comes from a known source, so the router's own logits are trained to name the source. An address that predicts well but says nothing about what produced it gets pushed toward one that does. This is the supervised scaffold; the unsupervised generalization (InfoNCE — identify which window among distractors produced this address) is a drop-in later step, cheapest test first.

**Idea 3 — the agent register — is the `RegisterBank` addressing mode** (`--address registers --n_registers K`). K explicit HH points plus a router; the effective address is a *convex* combination, so composition can never leave the hull of the registers — however wrong the router is, the address stays inside territory the registers themselves occupy. Registers 0..K-2 are the supervisable per-domain slots. Register K-1 is the AGENT slot: never supervised, its coordinates tracked separately in the checkpoint like you specified, kept alive by label smoothing that routes a little mass its way on every labeled batch ("recruiting the agent is never wrong") and free for the router to lean on for mixed or unseen content. Initialization follows the same philosophy as the encoder: registers start clustered near HH_DEFAULT and the router starts uniform, so step 0 is the certified-good single-shared-address regime, with structure growing only as gradient justifies it.

**The colosseum — "the model sets the game, and the game gets harder when it's too easy" — now has a thermostat instead of an arms race.** The certified v2 null (mean_error → 19.9B, unbounded) is fixed by two independent governors. First, the identification error is range-normalized and saturated (`e/(1+e)`, always in [0,1)), so no address choice can blow up the shared loss — the Architect can make the game *hard* but never make the loss *infinite*. Second, the Architect only proposes addresses inside a trust region of radius r around HH_DEFAULT, and r is controlled by the Solver's recent solve rate: above the band → too easy → r grows; below → too hard → r shrinks. The Architect's own objective is band-seeking rather than maximal — it's rewarded for challenges at the *edge* of the Solver's ability, not beyond it. An arms race has no equilibrium; a thermostat does. In the smoke run this behaved exactly as intended: solve rate climbed and radius expanded 0.15 → 0.19 across passes, with the difficulty state surviving checkpoint continuity.

## What was observed in the smoke tests

At toy scale (d=64, n=2, 150 steps, two synthetic corpora): routing went diagonal (pirouette text → its register at 0.93 mass), register separation grew monotonically (0.15 → 0.25), and — the detail worth noticing — **the AGENT register emerged as the best generalist point in the atlas**, beating both domain registers on mean basket CE. That is the intended role appearing without being directly supervised into it. Treat it as an encouraging anecdote, not a result; the pre-registered version is below.

Also from your v5 diversity data, one thing to watch on real runs: **J2 saturated its tanh bound** (pinned at 0.4999+ across nearly all pirouette samples) — the contrastive push found the J2 rail as the cheapest escape direction, and most of your usable separation actually lives in Ksi and Phi. A saturated dimension is a dead gradient direction. If register points start pinning against a rail, either widen that dimension's HH_RANGE or treat the pinning itself as a finding about which coordinates carry content.

## Run recipes

The full addressed-component training run, all pieces together:

```
python core_runner_v3.py \
  --modules tasks_registers:tasks=pirouette+classics_1,eval_only=classics_3,seq_len=256,batch=16,ident_weight=0.1 \
            tasks_colosseum:weight=0.05,run_every=2 \
  --pirouette_path pirouette_corpus_clean.txt --classics_path combined_classics.txt \
  --skip_shakespeare --pos alibi --address registers --n_registers 3 \
  --steps 30000 --save_to gen_v6_registers.pt
```

Chart the manifold on any checkpoint, then generate from what the atlas finds:

```
python map_manifold.py --ckpt gen_v6_registers.pt \
  --texts pirouette_corpus_clean.txt+combined_classics.txt --n_points 256 --out atlas_v6.json
python generate.py --ckpt gen_v6_registers.pt --prompt "..." --hh <any atlas point>
```

Generation in registers mode: `--prompt` alone routes the prompt (the printed mixture *is* the interpretability payoff — which domains the model thinks your prompt belongs to), `--register k` forces one register's own point, `--hh` remains the universal override.

## Pre-registered checks (write down before running, per house style)

For the registers run: PASS if held-out routing is diagonal-dominant (each named source routes ≥0.6 mass to its own register) AND per-source held-out CE is within 5% of a matched `--address encoder` contrastive run at identical steps. Diagonal routing with degraded CE means ident_weight is taxing the LM — lower it. Flat routing with fine CE means the router found the mixture unnecessary — itself informative: it says the backbone would rather absorb the variation than express it through the address, and `--hh_film` is the widening lever.

For the agent slot: track `agent_mass` on eval_only (never-trained) text vs. trained-source text. PASS for the agent-register hypothesis if eval_only text recruits the agent slot at meaningfully higher mass than trained text does. That's "the aggregator picks up what the specialists don't cover" as a measurable claim.

For the colosseum: PASS if solve_rate holds inside [band_lo, band_hi] while radius climbs and the run's total loss stays stable (no v2-style divergence). Radius pinned at r_min means the game is too hard at its easiest (raise solver capacity or weight); radius pinned at 1.0 with high solve_rate means the box is mastered — a real result meaning the inverse map covers the whole manifold.

## Honest caveats

The identifiability supervision uses source labels — a scaffold, not the final form; it can only teach the router distinctions you already named. The atlas's specialization score needs an absolute-quality gate when picking registers: a point can have the best own-domain *advantage* while being bad everywhere (the toy atlas showed exactly this — a "specialist" with own-CE 2.01 vs. the generalist's 1.24); pick registers from points that clear both bars. `infer_address()` under a neutral probe address answers "which address's behavior does this text's activations resemble," and its resolution is limited to the radius the game has explored so far — early in training, when r is small, expect its outputs to be compressed toward HH_DEFAULT (the smoke test showed a domain separation of only 0.06 at r≈0.17; that should widen as the governor expands the region). And addressing remains per-batch/per-window; per-token routing (the address varying along a sequence) is the natural next granularity, at the cost of per-position weight generation in the screw — worth its own isolated thread before merging.
