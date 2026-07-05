"""
tasks_moe_contrastive.py — MoE with cross-source address repulsion.

Subclasses tasks_moe.MoETask and adds a margin-based repulsion term on the
per-layer addresses produced by the MoERegisterBank. This exists because
pure MoE routing (load-balance only) gives no explicit incentive to route
DIFFERENTLY for different content — the registers can find a partition that
minimizes CE while all layers route similarly, defeating the point of
per-layer addressing.

MECHANISM: every step, alongside the normal single-source CE batch, sample
one window from a SECOND source. Compute per-layer addresses for both:

    addrs_A = moe(text_A)  →  [n_layers, 5]
    addrs_B = moe(text_B)  →  [n_layers, 5]

For each layer L, penalize when the distance is under a margin:

    contrastive = Σ_L relu(margin - ||addrs_A[L] - addrs_B[L]||) / n_layers

This is a floor, not an unbounded attractor — once addresses are margin apart,
the term is zero and stops pushing. Unlike the encoder contrastive module
(tasks_lm_contrastive.py), this operates on the per-layer MoE addresses, not
a single encoder output, so it directly encourages per-layer routing diversity.

MARGIN CHOICE: default 0.10 per layer (smaller than the 0.15 used for the
global encoder, because per-layer addresses have 5× more total dimensions
and the MoE's own load-balance loss is already preventing collapse). The
term sums over layers, so total repulsion scales with depth — the margin is
per-layer to keep it comparable across different model sizes.

WEIGHT CHOICE: default 0.02, deliberately very small — this is a nudge to
prevent routing monoculture, not a dominant objective. Watch route_entropy
per layer in eval: it should stay high and the cross-source address distance
(reported as `moe_contrastive_dist`) should grow from near-zero to a stable
nonzero value.

USAGE (drop-in for tasks_moe, adds contrastive_weight + contrastive_margin):

  python core_runner_v3.py \
    --modules tasks_moe_contrastive:tasks=pirouette+classics_1,eval_only=classics_3,seq_len=256,batch=16,lb_weight=0.01,contrastive_weight=0.02,contrastive_margin=0.10 \
    --pirouette_path ... --classics_path ... --skip_shakespeare \
    --pos alibi --address moe --n_routed 8 --top_k_active 4 \
    --hh_film 1 --grid_head 1 --steps 30000 --save_to gen_v8_moe_contrastive.pt
"""
import random
import torch
import torch.nn.functional as F

from tasks_moe import MoETask


class MoEContrastiveTask(MoETask):
    def __init__(self, contrastive_weight=0.02, contrastive_margin=0.10, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin
        self.name += f"+contrastive(w={contrastive_weight},m={contrastive_margin})"

    def setup(self, model, shared_ctx):
        super().setup(model, shared_ctx)
        self._contrastive_active = (
            self.contrastive_weight > 0
            and getattr(model, 'address_mode', None) == 'moe'
            and model.moe is not None
            and len(self.source_ids) >= 2
        )
        if self.contrastive_weight > 0 and not self._contrastive_active:
            print("  [moe_contrastive] WARNING: contrastive_weight > 0 but "
                  "need moe mode + >=2 sources — term disabled.")
        else:
            print(f"  [moe_contrastive] active: weight={self.contrastive_weight} "
                  f"margin={self.contrastive_margin} over {self.n_layers} layers, "
                  f"sources={self.source_ids}")

    def step_batch(self, model, shared_ctx):
        batch = super().step_batch(model, shared_ctx)
        if not self._contrastive_active:
            return batch

        src1 = batch['source']
        other_sources = [s for s in self.source_ids if s != src1]
        src2 = random.choice(other_sources) if other_sources else src1

        # One window from the second source
        probe2 = self._random_window(self.sources[src2][0])

        batch['probe2'] = probe2
        batch['source1'] = src1
        batch['source2'] = src2
        return batch

    def _random_window(self, ids):
        if len(ids) <= self.seq_len + 1:
            return ids[:self.seq_len]
        i = torch.randint(len(ids) - self.seq_len - 1, (1,)).item()
        return ids[i:i + self.seq_len]

    def loss(self, model, shared_ctx, batch):
        ce_loss, weight, log = super().loss(model, shared_ctx, batch)

        if not self._contrastive_active or 'probe2' not in batch:
            return ce_loss, weight, log

        # Get per-layer addresses for both sources.
        # We use the MoERegisterBank directly on the probe windows.
        # probe2 is a 1D tensor [T] — need to batch it.
        with torch.no_grad():
            # Address for source 1: reuse the stashed moe logits/address from the
            # CE forward pass. But the MoERegisterBank.forward is called inside
            # resolve_hh which happens before loss(). We can re-compute cheaply.
            addrs1, _ = model.moe(batch['x'][:1])     # [n_layers, 5] from moe.forward

        probe2_batch = batch['probe2'].unsqueeze(0).to(batch['x'].device)
        addrs2, _ = model.moe(probe2_batch)            # [n_layers, 5]

        # Per-layer L2 distance, meaned over layers
        per_layer_dist = torch.norm(addrs1 - addrs2, dim=1)  # [n_layers]
        mean_dist = per_layer_dist.mean()

        # Margin-based repulsion per layer: Σ_L relu(margin - dist_L) / n
        contrastive_term = F.relu(
            self.contrastive_margin - per_layer_dist
        ).mean()

        combined = ce_loss + self.contrastive_weight * contrastive_term

        log = dict(log)
        log['moe_contrastive_dist'] = round(float(mean_dist.detach()), 4)
        log['moe_contrastive_term'] = round(float(contrastive_term.detach()), 4)
        log['moe_contrastive_pair'] = f"{batch['source1']}~{batch['source2']}"
        log['moe_contrastive_min_layer'] = round(float(per_layer_dist.min().detach()), 4)
        log['moe_contrastive_max_layer'] = round(float(per_layer_dist.max().detach()), 4)
        return combined, weight, log


def build(config):
    """Drop-in replacement for tasks_moe.build with two extra contrastive params."""
    if 'tasks' not in config and 'root' not in config:
        raise SystemExit("tasks_moe_contrastive requires tasks=<name>[+...] and/or root=<folder>")
    return MoEContrastiveTask(
        tasks=str(config.get('tasks', '')),
        root=str(config['root']) if 'root' in config else None,
        weight=float(config.get('weight', 1.0)),
        seq_len=int(config.get('seq_len', 256)),
        batch=int(config.get('batch', 16)),
        eval_only=str(config.get('eval_only', '')),
        priority_temp=float(config.get('priority_temp', 1.0)),
        floor_threshold=float(config.get('floor_threshold', 1.5)),
        floor_patience=int(config.get('floor_patience', 0)),
        max_files=int(config['max_files']) if 'max_files' in config else None,
        eval_batches=int(config.get('eval_batches', 8)),
        lb_weight=float(config.get('lb_weight', 0.01)),
        z_weight=float(config.get('z_weight', 0.0)),
        contrastive_weight=float(config.get('contrastive_weight', 0.02)),
        contrastive_margin=float(config.get('contrastive_margin', 0.10)),
    )
