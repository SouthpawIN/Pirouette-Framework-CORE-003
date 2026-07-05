"""
tasks_lm_contrastive.py -- subclasses tasks_lm.LMTask to add a margin-based
repulsion term on the AddressEncoder's output, targeting the collapse
confirmed by the readdress_every diagnostic (encoder output moved by
<0.0002 across ~600 characters of genuinely different, increasingly
self-generated context).

WHY THIS EXISTS: under pure CE, the encoder has very little incentive to
differentiate by content. The certified generalist-point result (a single
jointly-optimized address landing within ~1% of each corpus's own CE
optimum) means there's almost no loss on the table for content-dependent
addressing to capture. A zero-initialized last layer plus that shallow
loss landscape means gradient descent has no strong reason to move the
encoder away from a near-constant output. This adds an explicit reason.

MECHANISM: every step, in addition to the normal single-source CE batch,
grab one representative window from a SECOND, different source. Compute
hh1 = encoder(window from source A), hh2 = encoder(window from source B).
Penalize only while ||hh1 - hh2|| is under a margin:

    contrastive_term = relu(margin - ||hh1 - hh2||)

This is a floor, not an unbounded attractor -- once the two addresses are
margin apart, the term is exactly zero and stops pushing. That keeps the
encoder from being dragged toward the unstable, garbage-generation corners
of the address box the way an unconstrained repulsion term could.

MARGIN CHOICE: pirouette's and classics' own single-corpus HH optima are
~0.41 apart in raw Euclidean distance (a real, already-measured, achievable
separation for this backbone). Default margin here is 0.15 -- well inside
that, so the term asks the encoder to recover a fraction of a distance the
loss landscape has already been shown to tolerate, not to invent a new one.

WEIGHT CHOICE: default 0.05, deliberately small relative to typical CE
values (~1.0-2.0 in your logs) so this nudges rather than dominates. Watch
eval_only CE closely -- if it degrades more than a few percent relative to
a matched non-contrastive run, the weight is too high.

PRE-REGISTRATION (write this down before running, not after):
  PASS: mean cross-source encoder distance (measured by
  check_encoder_diversity.py) increases by >= 2x over a matched
  non-contrastive baseline checkpoint, AND eval_only CE on classics_3
  degrades by < 5% relative to that same baseline.
  FAIL: distance does not move, or CE degrades beyond that threshold.
  Either result is informative -- a FAIL on distance with stable CE means
  the contrastive term itself needs a larger weight or margin, not that
  the encoder can't differentiate; a FAIL on CE means the shared backbone
  is fighting the differentiation and the margin/weight need to come down.

USAGE (drop-in replacement for tasks_lm, same required args plus two new
optional ones):

  python core_runner_v3.py \
      --modules tasks_lm_contrastive:tasks=pirouette,eval_only=classics_3,seq_len=256,batch=16,contrastive_weight=0.05,contrastive_margin=0.15 \
      --pirouette_path pirouette_corpus_clean.txt --classics_path combined_classics.txt \
      --skip_shakespeare --pos alibi --address encoder --steps 30000 \
      --save_to gen_v5_contrastive.pt --hh_film 1

Then compare against gen_v4.pt (or a freshly matched non-contrastive
baseline at identical steps/pos/hh_film) with check_encoder_diversity.py.
"""
import random
import torch
import torch.nn.functional as F

from tasks_lm import LMTask


class LMContrastiveTask(LMTask):
    def __init__(self, contrastive_weight=0.05, contrastive_margin=0.65, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin
        self.name += f"+contrastive(w={contrastive_weight},m={contrastive_margin})"

    def setup(self, model, shared_ctx):
        super().setup(model, shared_ctx)
        self._contrastive_active = (
            self.contrastive_weight > 0
            and getattr(model, 'address_mode', 'table') == 'encoder'
            and len(self.source_ids) >= 2
        )
        if self.contrastive_weight > 0 and not self._contrastive_active:
            print("  [lm_contrastive] WARNING: contrastive_weight > 0 but address_mode "
                 "!= 'encoder' or fewer than 2 sources -- contrastive term will be a no-op.")
        else:
            print(f"  [lm_contrastive] active: weight={self.contrastive_weight} "
                 f"margin={self.contrastive_margin} sources={self.source_ids}")

    def step_batch(self, model, shared_ctx):
        batch = super().step_batch(model, shared_ctx)
        if not self._contrastive_active:
            return batch

        src1 = batch['source']
        other_sources = [s for s in self.source_ids if s != src1]
        src2 = random.choice(other_sources) if other_sources else src1

        # one representative window per source -- cheap, matches the pattern
        # find_address.py/generate.py already use for encode_text() calls
        probe1 = batch['x'][0]  # already-sampled window from src1, reuse it
        probe2 = self._random_window(self.sources[src2][0])

        batch['probe1'] = probe1
        batch['probe2'] = probe2
        batch['source1'] = src1
        batch['source2'] = src2
        return batch

    def _random_window(self, ids):
        i = torch.randint(len(ids) - self.seq_len - 1, (1,)).item()
        return ids[i:i + self.seq_len]

    def loss(self, model, shared_ctx, batch):
        ce_loss, weight, log = super().loss(model, shared_ctx, batch)

        if not self._contrastive_active or 'probe1' not in batch:
            return ce_loss, weight, log

        hh1 = model.encoder.encode_text(batch['probe1'])
        hh2 = model.encoder.encode_text(batch['probe2'])
        dist = torch.norm(hh1 - hh2)
        contrastive_term = F.relu(self.contrastive_margin - dist)
        combined = ce_loss + self.contrastive_weight * contrastive_term

        log = dict(log)
        log['contrastive_dist'] = round(float(dist.detach()), 4)
        log['contrastive_term'] = round(float(contrastive_term.detach()), 4)
        log['contrastive_pair'] = f"{batch['source1']}~{batch['source2']}"
        return combined, weight, log


def build(config):
    from tasks_lm import build as base_build
    base = base_build(config)  # validates tasks=/root=, reuses all existing defaults/parsing
    return LMContrastiveTask(
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
        contrastive_weight=float(config.get('contrastive_weight', 0.05)),
        contrastive_margin=float(config.get('contrastive_margin', 0.15)),
    )
