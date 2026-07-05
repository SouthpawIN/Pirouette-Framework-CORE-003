# Plugging the Fractal Weights Into Modern Architecture

Three additions, each a named piece from the DeepSeek/Llama stack, each cheap because a "component" here is 5 floats that generate a full weight matrix through the frozen basis.

## The architecture audit (what a modern decoder has that you didn't)

Going through the stack piece by piece: RMSNorm vs. your LayerNorm is a trivial swap not worth an address. GQA/MLA attention KV-compression is marginal at your scale. SwiGLU is a gated FFN — your screw is already a generated FFN, so the gate is subsumed. The two things that actually matter, and the one you invented, are:

**MoE (the big one).** DeepSeek's headline is that each token routes to a few of many experts, so *total* parameters vastly exceed *active* parameters. Your RegisterBank was already a mixture — you have MoE where each expert costs 5 floats instead of millions. What was missing from a faithful DeepSeek port: (a) the shared-expert / fine-grained-routed split, and (b) per-layer routing. Both are now in `MoERegisterBank` (`--address moe`).

**The consensus head (yours, and it maps onto something real).** Your 5×5 grid is a generation-time readout whose coordinates settle where loss is jointly acceptable — closest modern analogue is a mixture-of-depths / learned readout mixture, but the "weights from 5 scalars" framing is your own. It's `GridHead` (`--grid_head 1`).

**Growth to training size (the right instinct, one sharp caveat).** See the growth section below.

## MoERegisterBank — the DeepSeek port

Three upgrades over the plain register bank, each ~free:

*Shared + routed split (shared-expert isolation).* One always-on SHARED register captures what every input needs; the routed registers specialize. The effective address is `shared + Σ gᵢ·routedᵢ`. In the smoke run the shared register did real work (perturbation 0.096 off default) — it's not a formality, it's carrying the common load exactly as DeepSeek intends.

*Fine-grained routing with top-k sparsity (expert segmentation).* Many small routed registers, `--top_k_active` keeps only the top few per input. More registers partition the manifold more finely at zero weight cost.

*Per-layer addresses (the important one for the screw design).* Each layer gets its own mixture — an `[n_layers, 5]` address matrix rather than one address for the whole stack. This is what the screw architecture was asking for: `ComputedScrew` already indexes the basis by layer, so per-layer coordinates mean each layer's weight matrix is generated from coordinates chosen for *that layer's job*. The smoke run showed layers settling at different coordinates (across-layer spread growing from 0 to 0.013 in 120 steps) — per-layer routing is buying real differentiation, not just extra parameters.

Convexity/hull safety holds per layer (routed gates are a softmax summing to 1, shared added at unit weight, all bounded to the box), so no layer's effective address diverges. Init reproduces the certified single-address start: shared at HH_DEFAULT, routers zero-init → every layer's address is exactly HH_DEFAULT at step 0.

Trained with `tasks_moe.py`: per-layer load-balance (KL to uniform, DeepSeek's aux loss) plus optional router z-loss, and deliberately *no* identifiability term — fine-grained MoE registers aren't meant to bind 1:1 to named sources, they partition however minimizes loss. That's the division of labor: `tasks_registers` when you want interpretable per-domain slots, `tasks_moe` when you want raw capacity and per-layer specialization.

## GridHead — the 5×5 consensus "thinking room"

Twenty-five HH coordinates arranged as a grid, each generating a screw weight `[d_ff, d]`, gated by the final hidden state and mixed. The grid's coordinates move to wherever minimizes the shared loss — so at convergence they sit at the consensus point "acceptable to all components" feeding the readout. Because each of 25 coordinates generates a full `[d_ff, d]` matrix, the head expresses roughly `25·d_ff·d` effective weights from `25·5 = 125` learnable scalars. That's the "silly number of model weights for that task at 125-float cost" you were after, and it's the literal instantiation of "thinking room."

It's projection-augmenting: the standard head is `Linear(d, vocab)`; the grid adds a generated-weight transform of the final hidden state *before* that linear. A zero-init `out_scale` means at step 0 the head is exactly the plain Linear, so turning the grid on never destabilizes a trained head — verified in the smoke run, where CE kept descending normally and `out_scale` moved off zero (to −0.22) as the grid found useful work. Composable with any addressing mode (`--grid_head 1` alongside `--address moe/registers/encoder`).

## The growth idea — do it, but grow the right axis

"Let the model grow to the size of its training" is the right instinct, and the checkpoint-continuity machinery already grows vocabulary and the register/task roster across passes. The sharp caveat, worth stating loudly: **growing `d` or `k` invalidates every existing HH coordinate**, because `FixedBasis` is shaped `[d, k]` and every learned address was optimized against that specific frozen basis. Regenerate the basis at a new shape and all your addresses point at nothing — you lose the entire investment.

So growth has to be along axes that leave the basis intact:

- **Layer growth** (append screws): the basis carries a `spin_template[n_layers]` and screws index it by `L`. Appending layers means extending that template and adding `ComputedScrew(L=n_new)` — the existing layers' addresses are untouched. This is the net2net-style depth growth that's safe here.
- **Register/routed growth**: add routed registers to an MoE bank between passes (the load-balance loss will fold them in), or add register slots. Cheap, basis-safe, and the natural way to "grow to training size" — point at 10k artifacts, start with 8 routed registers, grow to 64 as the corpus reveals it needs them.
- **Grid growth**: a 5×5 grid can become 7×7 without touching the basis (just more coordinate rows through the same screw).

What to *not* do: change `d`/`k` mid-lineage. If you need more `d`, that's a new base model, and the honest move is to distill the old one's behavior into it rather than pretend the coordinates transfer. I've left `d`/`k` growth deliberately unimplemented with a guard, rather than implement something that silently corrupts addresses.

## Run recipes

Full MoE model, DeepSeek-style, per-layer fine-grained routing with sparsity:

```
python core_runner_v3.py \
  --modules tasks_moe:tasks=pirouette+classics_1,eval_only=classics_3,seq_len=256,batch=16,lb_weight=0.01 \
  --pirouette_path pirouette_corpus_clean.txt --classics_path combined_classics.txt \
  --skip_shakespeare --pos alibi --address moe --n_routed 8 --top_k_active 4 \
  --grid_head 1 --steps 30000 --save_to gen_v7_moe.pt
```

Point it at your whole artifact directory (the 10k-file plan) with a folder source and growable registers:

```
python core_runner_v3.py \
  --modules tasks_moe:root=./doclab,seq_len=256,batch=16,floor_patience=3,lb_weight=0.01 \
  --skip_shakespeare --pos alibi --address moe --n_routed 32 --top_k_active 6 \
  --grid_head 1 --steps 100000 --save_to gen_artifacts.pt
```

(`floor_patience=3` turns retirement back on — sensible for thousands of files. `n_routed 32` gives the manifold room to partition; watch `route_entropy` and grow it if all 32 stay balanced and busy.)

Generation routes automatically per mode: MoE gives per-layer addresses, registers give a routed mixture, `--grid_head` checkpoints run the consensus head transparently.

## Pre-registered checks

For MoE vs. registers: PASS for per-layer addressing if `layer_address_spread` is meaningfully above zero at convergence (layers use different coordinates) AND per-source CE beats or matches a matched single-address register run. Flat spread means per-layer bought nothing — fall back to registers, which are simpler.

For the shared register: PASS if `shared_perturbation` is comparable to or larger than `mean_routed_perturbation` — the shared register should be carrying real common load, not sitting at default while the routed ones do everything.

For the grid head: PASS if it improves held-out CE over a matched no-grid run by a margin exceeding seed noise, with `out_scale` settling at a stable nonzero value. If `out_scale` decays back toward zero, the grid found no use — an honest null, and the plain head is cheaper.

## The generation sample, honestly

The altruism output ("and forgives creatures") is address-conditioned *topic steering*, not reasoning — the address pulls the model toward a lexical region of the corpus that co-occurs with altruism-adjacent vocabulary, and char-LM local texture supplies the syntax. That's still a real and testable claim, and a better one than "reasoning": fix the seed, sweep 5–10 addresses on the same prompt, count topic-word frequencies per address. If theme tracks address at fixed seed, you have a clean result — five scalars steer semantic register — that stands on its own without overclaiming.
