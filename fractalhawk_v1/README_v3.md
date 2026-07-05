# CORE-003 v3 — Redesign Notes

Every structural change here maps to a certified finding in the consolidated checkpoint. The theme of the redesign is a single inversion: **stop storing addresses and start computing them.**

## The diagnosis, restated

Your certified findings describe three independent failure sources for generation, and only one of them is about training quality. First, learned absolute positional embeddings guarantee a coherence collapse at the training seq_len, because `torch.arange(T)` starts every training window at position 0 and rows beyond it never see gradient. Second, the per-task HH lookup table makes untrained and diluted addresses *structurally possible* — the optimizer-registration bug, the garbled-generation-under-default-init failure, and the ~2700-way capacity dilution are all symptoms of addresses being free parameters that can exist without being trained. Third, the table gives the manifold no reason to organize; the +0.306 correlation was an accident of shared-backbone pressure, and the text→HH regression was fighting targets that are part signal, part per-document hash noise.

## What changed

**Positions** (`--pos {alibi, rope, learned}`, default `alibi`). ALiBi has no positional parameters at all — position enters as a linear attention bias — so there is nothing to leave untrained and the collapse ceiling is removed by construction rather than pushed out. RoPE is included as the standard alternative, and `learned` is kept as the reproducible v2 control. This is your Open Question 3, now runnable as a pre-registered three-arm A/B on identical data. The fixed-size causal mask buffer is also gone (replaced with `scaled_dot_product_attention` plus a length-cached ALiBi bias), so no buffer bakes in a sequence ceiling.

**Addressing** (`--address {encoder, table, blend}`, default `encoder`). The `AddressEncoder` reads the input window (shared character embeddings → mean/max pool → 2-layer MLP) and emits HH coordinates bounded to `HH_DEFAULT ± HH_RANGE` via tanh, so every produced address lives in the numerically sane box the screw math was designed around. Its last layer is zero-initialized, which means step 0 of an encoder run is *exactly* a single-shared-address run — the regime your direct_lm result certified as good — and content-dependence grows outward only as gradient justifies it. Consequences: the lazy-parameter bug class cannot recur (the encoder exists before the optimizer is built; `sync_optimizer` is retained as a tripwire, and if it ever reports synced parameters in an encoder run, something is wrong); an untrained address cannot exist; similar text maps to nearby coordinates because the encoder is a smooth function, so manifold organization is built in rather than hoped for; and generation-time conditioning for unseen text — your deferred Q7 — becomes trivially safe, because encoding the prompt is precisely the operation training optimized, not a post-hoc regressor bolted onto noisy targets. `blend` interpolates table and encoder outputs and is the recommended bridge when continuing from a table checkpoint.

**Task modules.** `tasks_lm.py` replaces both `tasks_direct_lm` and `tasks_microtraining`. Under table addressing those two had to be different things (one address trained heavily vs. thousands trained thinly); under encoder addressing there is no per-source address parameter to dilute, so both collapse into "sample windows from sources, priority-weighted by recent loss." The balancer's retirement rule is kept but off by default (`floor_patience=0`), since a floored source still feeds useful gradient to the shared pathway. `eval_only` is kept and, in encoder mode, upgraded: unseen text is evaluated under *its own* encoder-produced address — the honest generation-time condition — so the generalization gap you measure is the one generation will actually experience. Table/blend sources are registered in `setup()`, before the optimizer exists, eliminating the lazy `add_task` path entirely. The HUD prototype machinery is not carried forward: the encoder subsumes it (`encoder(text)` *is* the address-inference the HUD's nearest-centroid classifier was approximating at ~10% top-3).

**Generation** (`generate.py`). Sliding-window sampling with temperature/top-k/top-p, address supplied by prompt encoding, by table lookup, or as an explicit `--hh J1,J2,Phi,E,Ksi` literal. The `--window` flag is a memory knob, not a coherence cliff. `--readdress_every N` optionally re-encodes the address from recent output as generation proceeds.

**Manifold search** (`find_address.py`). This is your generalist-point idea implemented directly. Because the address is 5 scalars steering a frozen backbone, "find an address that is better for generation overall" is a 5-dimensional optimization: multi-start (the historical default, every trained table point, encoder outputs on samples of the eval basket, plus random points in the bounded box) followed by Adam on raw coordinates against mean cross-entropy over a *basket* of texts, then a fixed-seed comparable rescore. Note why this sidesteps the Q2 null: the regression needed a dataset of (text, address) pairs and died at n=226 against noisy targets; the search needs no dataset at all — it optimizes against the actual LM loss. Trained table points are among the starts, so the search can only match-or-beat your best existing address on the basket. The winner prints as a ready-to-run `generate.py --hh ...` command.

**Migration** (`migrate_checkpoint.py`). Ports a v2 checkpoint into v3: embed/head, attention, gates, layernorms, and all hh_by_task coordinates transfer cleanly; `pos.weight` and the old mask buffers are dropped when moving to alibi/rope. The warm start is approximate — the old attention weights expected position vectors added to the residual stream, and under a relative scheme that signal arrives through attention geometry instead — so expect an initial loss spike and run at least one finetuning pass before judging generation. At d=128/n=8 char scale this still beats from-scratch.

**Optional widening** (`--hh_film 1`). The screw only touches the MLP down-projection, so the 5-dim address has one narrow channel of influence. FiLM adds a tiny shared MLP mapping hh → per-layer residual gains on the attention and screw branches, zero-initialized so it is exactly identity at step 0. Off by default for comparability; it is the next lever if encoder mode trains well but coherence still under-delivers — widen the address's *influence* before blaming training.

## Recommended first runs

Fresh encoder run on your real corpora, the new default configuration:

```
python core_runner_v3.py --modules tasks_lm:tasks=pirouette,eval_only=classics_3,seq_len=256,batch=16 \
  --pirouette_path pirouette.txt --classics_path classics.txt --skip_shakespeare \
  --pos alibi --address encoder --steps 30000 --save_to gen_v3.pt
python generate.py --ckpt gen_v3.pt --prompt "your prompt here" --n_chars 1500
```

The pre-registered Q3 test (does relative positioning remove the ceiling or push it out): three runs identical except `--pos alibi`, `--pos rope`, `--pos learned --seq_len 256`; generate 4–8× the training window from each and measure where coherence degrades. The learned arm should reproduce the certified ~seq_len collapse; the interesting comparison is alibi vs. rope beyond ~2× the window.

Warm start from your existing checkpoint instead of from scratch:

```
python migrate_checkpoint.py --old gen_v2.pt --new gen_v3_warm.pt --pos alibi --address blend
python core_runner_v3.py --modules tasks_lm:tasks=pirouette,seq_len=256 \
  --init_from gen_v3_warm.pt --address blend --steps 10000 --save_to gen_v3_ft.pt
```

Manifold search for a generalist point, then generate from it:

```
python find_address.py --ckpt gen_v3.pt --texts pirouette.txt+classics.txt --iters 300
python generate.py --ckpt gen_v3.pt --prompt "..." --hh <the printed winner>
```

## What to watch for

The interesting open empirical question this design creates: in encoder mode, does the encoder actually *use* its 5 dimensions, or does it collapse to emitting one point for everything? Check by encoding snippets from different corpora and looking at the spread of coordinates (the search tool's encoder starts print these). Collapse-to-a-point would mean the shared backbone prefers to absorb all content variation itself — a real finding either way, and the `--hh_film` widening plus a heavier encoder (`--enc_hidden`) are the levers if you want to force more of the variation through the address bottleneck. Also note the batch-pooled design choice: the encoder emits one address per batch (the analogue of table mode's one-address-per-task-batch); per-example addressing is a one-line change in `AddressEncoder.forward` if you want to test it, at the cost of a per-example weight compute in the screw.
