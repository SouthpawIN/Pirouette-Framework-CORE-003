"""
hh_orbital_generator_v24r6.py
Orbital Generator v24r6 — Cone Illumination Architecture
Pirouette Framework Volume 8 · CORE-003 · ML-099

TWO ARCHITECTURAL CHANGES
==========================

CHANGE 1: CONE ILLUMINATION (replaces moving arc particle)
───────────────────────────────────────────────────────────
Old architecture: A separate arc particle phi_arc evolves continuously via
HH transport equations. Token selection looks near phi_arc. The arc and
the hidden state j1_h are two coupled but distinct objects — the arc can
drift away from the actual token positions.

New architecture: The arc IS the token sequence. After generating each
token, the generator teleports to that token's (j1, ksi) address and stops.
There is no continuous arc motion between tokens.

From that stopped position, a DIRECTIONAL CONE illuminates candidates:

  cone_axis   = arc direction from last K token steps (bivector R_bigram)
  cone_aperture θ = base_angle × (1 + winding_bonus)
                  where winding_bonus expands the cone in high-winding zones
                  (frontier vocabulary-rich regions need wider search)
                  and contracts it in low-winding zones (settled basin = tight focus)

Gain function inside the cone:
  For each candidate token at angular offset δ from cone axis:

  direction_weight = exp(-δ² / (2σ²))        σ = θ/2 (beam std)
  distance_weight  = exp(-dist / CONE_REACH)  (tokens farther away get falloff)
  amplification    = (1 + AMP × mass_lift)    (strong tokens amplified over weak)

  cone_score = direction_weight × distance_weight × amplification

  This is the "zooming outward and amplifying strong weights over weak ones"
  mechanism: the cone concentrates energy along the forward direction, and
  within that beam, high-mass tokens (content words, rare words) outcompete
  low-mass tokens more strongly at distance than they do at close range.

The cone pre-filters candidates before TPCI scoring. Tokens outside the
cone are not scored at all — this is a geometric gate, stronger than a
penalty. TPCI then scores within-cone candidates as before.

Cone aperture by zone (from certified inner/outer orbital radii, ML-OG):
  Zone A center (j1=80-110°): θ = 20° (inner orbit, function words)
  Zone A edge   (j1=110-140°): θ = 35° (transition zone, mixed)
  Zone B        (j1=265-340°): θ = 50° (outer orbit, sparse content words)
  Winding bonus: multiply θ by (1 + winding_norm × WINDING_SCALE)

CHANGE 2: DYNAMIC KSI TARGET + SEMANTIC ANCHOR
────────────────────────────────────────────────
Dynamic Ksi target: A Ksi preference that changes per sentence phase,
controlling register without changing token addresses.

  Phase 0 (steps 0-25% of sentence): KSI_TARGET = ksi_init = 0.50
    Register: concrete/factual (Gold basin)
  Phase 1 (steps 25-60%): KSI_TARGET rises to KSI_MID = 0.68
    Register: analytical/relational (Teal basin)
  Phase 2 (steps 60-85%): KSI_TARGET = KSI_MID = 0.68
    Register: elaboration maintained
  Phase 3 (steps 85-100%): KSI_TARGET falls back to ksi_init
    Register: grounded conclusion

  Scoring bonus: exp(-((tok_ksi - KSI_TARGET) / KSI_SIGMA)²) × KSI_WEIGHT

Semantic anchor: Recency-weighted running mean of last K token (j1, ksi)
positions. Tokens near the anchor (where generation has been coherent)
get a proximity bonus. This is semantic gravity — the context pulls toward
the neighborhood that has been producing output.

  anchor_j1  = recency-weighted mean of last K tok_j1 values
  anchor_ksi = recency-weighted mean of last K tok_ksi values
  anchor_bonus = exp(-dist(tok, anchor)² / ANCHOR_SIGMA²) × ANCHOR_WEIGHT

PRE-REGISTERED (ML-099)
========================
H-V24-001: CONE REDUCES BURST dj1 BELOW 15°
  The directional cone, by pre-filtering candidates to forward-aligned tokens,
  should reduce J1 variance within bursts.
  PASS: mean_burst_dj1 < 15.0°

H-V24-002: DYNAMIC KSI TARGET SHIFTS VOCABULARY REGISTER
  At KSI_TARGET=0.5 (sentence start): mean tok_ksi of selected tokens < 0.60
  At KSI_TARGET=0.68 (sentence middle): mean tok_ksi > 0.60
  PASS: ksi_start_mean < 0.60 AND ksi_mid_mean > 0.60

H-V24-003: SEMANTIC ANCHOR INCREASES LOCK_FRAC
  By pulling token selection toward the neighborhood where spin-locked
  tokens have been generated, anchor increases lock_frac.
  PASS: lock_frac > 0.85

H-V24-004: CONE BEATS ARC (head-to-head)
  burst_fisher(v24_cone) max_burst > burst_fisher(v24_arc) max_burst × 1.0
  Run both modes in ablation. Cone should match or exceed arc.
  (Not demanding strict improvement — testing that cone is not worse)
  PASS: cone_max >= arc_max

H-V24-PHASE: PHASE NULL RETEST
  Structured cone direction (real bigram history) must outperform
  shuffled cone direction (randomized bigram history).
  PASS: real_mean_burst > shuffled_mean_burst × 1.10
  This is the critical structural test for whether cone direction is doing
  real work. In v23, this failed at W_BIGRAM=0.15 and 0.06. With the cone
  pre-filtering (not just reweighting), the directional signal is stronger.

Run:
  python hh_orbital_generator_v24.py ablation ^
    --engram engram_curve.json ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --n_tokens 80 --output v24_ablation.json

  python hh_orbital_generator_v24.py fish ^
    --engram engram_curve.json ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --n_tokens 120 --n_fish 20 --output v24_fish.json

  python hh_orbital_generator_v24.py phase_null ^
    --engram engram_curve.json ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --n_tokens 80 --output v24_phase_null.json
"""

import argparse, json, math, random, sys
from pathlib import Path
from collections import deque
import numpy as np

try:
    from transformers import GPT2TokenizerFast
    HAS_TOK = True
except ImportError:
    HAS_TOK = False

# ══════════════════════════════════════════════════════════════════════════════
# CERTIFIED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
T_A2B           = 0.1697
THETA_LOCK      = -130.0
DEPOL_J1        = 65.0
DEPOL_WIDTH     = 15.0
G_TIME          = 0.40
G_PULL          = 2.00
OMEGA_BASE      = 8.0
MIN_FRAC        = 0.05
DEPOSIT_STD     = 18.0
DEPOSIT_HALF    = 12
PENUMBRA_R      = 45.0
WADA_KSI        = 0.879
WADA_J1         = 90.0
K_POL           = 0.25
XI_DEP          = 20.0
VEL_LOW         = 5.0

# Calibrated geometry (ML-097)
PHI_INIT        = 16.0
KSI_INIT        = 0.50
SENT_MAX        = 82

# Z3 skeleton
Z3_GOLD, Z3_TEAL, Z3_RED = 30., 150., 270.
CLOSURE_J1      = 30.0

# ── Cone architecture parameters ─────────────────────────────────────────────
K_HIST          = 5          # arc history length for cone direction
CONE_BASE       = 25.0       # base half-angle (degrees)
CONE_REACH      = 60.0       # exponential falloff scale (degrees)
CONE_AMP        = 1.5        # amplification factor for mass lift at distance
WINDING_SCALE   = 0.8        # how much winding bonus expands aperture
CONE_STRICT     = False       # if True: hard gate (exclude outside cone)

# ── Spin gate ─────────────────────────────────────────────────────────────────
# Per-token spin deviation from THETA_LOCK classifies tokens as field particles.
# Hard gate: anti-aligned tokens (spin_dev > 90°) are virtual particles — skip.
# Soft weight: smoothly rewards field-coherent tokens (spin_dev < 45°).
SPIN_GATE_HARD = 90.0
SPIN_GATE_SOFT = 45.0
SPIN_GATE_W    = 0.30

# ── Cadence: burst-triggered ksi lift ────────────────────────────────────────
# When arc is in a sustained burst (burst_len >= KSI_BURST_MIN), ksi_target
# jumps immediately to KSI_BURST (content word territory) rather than waiting
# for the slow sentence-level arc. Fast oscillation inside slow arc.
#   burst open  (len 0-2):  ksi ~ 0.50  (function words)
#   burst peak  (len 3+):   ksi ~ 0.67  (content words)  ← THIS LIFT
#   burst close (break):    ksi returns to sentence phase
KSI_BURST     = 0.67
KSI_BURST_MIN = 3

# ── Cadence: dynamic mass + length schedule ───────────────────────────────────
# English phrase structure: DET(short,low-mass) -> NOUN(long,high-mass) -> VERB
# Mass and length schedules encode this as burst-position functions.
# The length schedule is the HH word-length predictor — implemented as a
# deterministic function of character count rather than a trained model.
# (Training a network to predict length from HH position would rediscover
# exactly this mapping: mass × ksi × length are jointly correlated in the
# engram geometry via Zipf's law.)
MASS_SCHED = {'open': 0.5, 'peak': 2.0, 'close': 1.0}
LEN_SCHED  = {'open': 2,   'peak': 7,   'close': 4  }
W_LEN      = 0.20

                              # if False: soft gate (penalize outside cone)

# Zone-specific aperture (from certified inner/outer orbital radii)
ZONE_APERTURE = {
    'A':     20.,   # inner orbit (function words, tight)
    'B':     50.,   # outer orbit (sparse content words, wide)
    'DEAD':  35.,   # gap zone — use moderate aperture
    'OTHER': 30.,   # default
}

# ── Dynamic Ksi target ────────────────────────────────────────────────────────
KSI_MID         = 0.68       # peak Ksi target (analytical/Teal register)
KSI_SIGMA       = 0.15       # Ksi preference width
KSI_WEIGHT      = 0.35       # strength of Ksi preference in scoring

# ── Semantic anchor ───────────────────────────────────────────────────────────
ANCHOR_K        = 8          # recency window for semantic anchor
ANCHOR_DECAY    = 0.85       # recency decay (recent tokens weighted more)
ANCHOR_SIGMA    = 25.0       # angular width of anchor proximity bonus
ANCHOR_KSI_SIG  = 0.12       # Ksi width of anchor proximity bonus
ANCHOR_WEIGHT   = 0.25       # strength of anchor in scoring

# ── Bivector weights (v23r3 calibrated) ──────────────────────────────────────
W_BIGRAM        = 0.06       # cone direction weight from arc history
W_SPIN          = 0.25       # spin correction weight (negated = restoring)
W_CLOSURE       = 0.10       # closure tether weight

NOISE_BREAK_N   = 3

STOPWORDS = {
    'the','a','an','in','of','to','is','was','are','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should','may',
    'might','shall','can','each','its','this','that','these','those','it','he',
    'she','they','we','you','i','my','your','his','her','their','our','who',
    'which','what','when','where','how','and','or','but','if','not','on','at',
    'by','for','with','from','as','all','said','some','re','un','two','one',
    'three','after','also','most','any','other','just','more','very','up','out',
    'about','into','than','so','no','only','new','old','over','such','then',
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def ang_diff(a, b): return float(((float(b)-float(a)+180)%360)-180)
def ang_dist(a, b): return abs(ang_diff(a, b))

def j1_zone(j1):
    j = float(j1) % 360
    if 80 <= j <= 140:   return 'A'
    if 265 <= j <= 340:  return 'B'
    if 140 < j < 265:    return 'DEAD'
    return 'OTHER'

def is_dead(j1): return 140 < float(j1)%360 < 265

def lut_get(phi, lut):
    phi = float(phi) % 360.
    i = phi / 360. * len(lut)
    lo = int(i) % len(lut); hi = (lo+1) % len(lut); t = i - int(i)
    return float(lut[lo]*(1-t) + lut[hi]*t)

def lut_grad(phi, lut):
    return (lut_get(phi+1, lut) - lut_get(phi-1, lut)) / 2.

def is_real_word(w):
    ws = w.strip().strip("'\".,;:!?-—()[]{}*#@%^&+=|\\/<>~`")
    return len(ws) > 2 and any(c.isalpha() for c in ws) and not any(c.isdigit() for c in ws)

def is_function(w):
    return w.strip().lower().strip("'\".,;:!?") in STOPWORDS

def is_alpha_token(w):
    return any(c.isalpha() for c in w)

# Carrier tokens: syntactic scaffolding, neutral to burst counting
CARRIERS = {
    'a','A','an','An','the','The','in','In','of','Of','to','To','at','At',
    'on','On','by','By','or','Or','as','As','is','Is','it','It','be','Be',
    'do','Do','if','If','no','No','so','So','up','Up','us','Us','we','We',
    'my','My','he','He','me','Me','go','Go','I','s','n','m','B','N','S',
    'M','D','C','E','F','G','H','J','K','L','P','Q','R','T','V','W',
    'Y','Z','re','un','ex','co','de','pro','pre','non','anti',
}

def token_charge(tok_text):
    """
    CONTENT  — meaningful word, extends burst
    CARRIER  — syntactic scaffolding, neutral (neither extends nor breaks burst)
    NOISE    — BPE fragment / symbol, breaks burst
    """
    ws = tok_text.strip()
    if not any(c.isalpha() for c in ws): return 'NOISE'
    if '<|' in ws: return 'NOISE'
    ws_clean = ws.lower().strip("'\".,;:!?-—")
    if ws_clean in CARRIERS or len(ws_clean) <= 2: return 'CARRIER'
    if any(c.isalpha() for c in ws) and not any(c.isdigit() for c in ws):
        return 'CONTENT'
    return 'CARRIER'


def is_sent_end(w):
    ws = w.strip()
    return any(ws.endswith(p) for p in ['.','!','?','."','!"','?"']) or ws in ['.','!','?']

def load_engram(path):
    with open(path, encoding='utf-8') as f: e = json.load(f)
    out = {
        'ksi_v': np.array(e['ksi_vals'], dtype=np.float32),
        'j1_v':  np.array(e['j1_pca'],  dtype=np.float32) % 360.,
        'n':     len(e['ksi_vals']),
        'has_spin': 'spin_angle' in e,
    }
    if out['has_spin']:
        out['spin_angle'] = np.array(e['spin_angle'], dtype=np.float32)
    else:
        out['spin_angle'] = np.full(out['n'], math.radians(THETA_LOCK), np.float32)
    return out

def load_lut(path, n=360):
    if path and Path(path).exists():
        arr = np.load(path)
        return (arr[:,1] if arr.ndim == 2 else arr).astype(np.float32)
    return np.ones(n, np.float32) * 0.5

def load_mass(path):
    if path and Path(path).exists():
        with open(path) as f: d = json.load(f)
        return np.array(d['masses_norm'], dtype=np.float32)
    return None

# ══════════════════════════════════════════════════════════════════════════════
# v25 ADDITIONS — ML-100
# ══════════════════════════════════════════════════════════════════════════════
#
# Two independent, separately-ablatable knobs added on top of v24r6:
#
#   KNOB 1 — FLOW FIELD (direction)
#     A transition bivector field on the (j1, ksi) manifold. For each manifold
#     cell, the empirical mean outgoing displacement (Δj1, Δksi) of real English
#     bigrams whose source token lands in that cell. This is the geometric
#     successor model: "from where I am, where does English actually go next."
#     It is PURE GEOMETRY — a flow on the manifold, not a weight matrix, not the
#     transition GRAPH of VEMAW (which stays in the null). It only sets the cone
#     AXIS direction. Built once from a corpus run through the tokenizer.
#
#   KNOB 2 — INVERTED-VEMAW LAMBDA CONTROLLER (commitment entropy)
#     The EntropyController math from VEMAW_Core (λ → temperature → softmax),
#     lifted verbatim and fed CONE SCORES instead of transition counts. It carries
#     NO linguistic structure — it only modulates how decisively we commit to
#     whatever the geometry already scored. Crucially the λ-from-roughness map is
#     INVERTED relative to the null's instinct:
#         null VEMAW:  melt (high entropy) when structure is present
#         v25 generator: explore (high λ) when SMOOTH / over-visited / stuck
#                        commit  (low  λ) when ROUGH / structured / pointing
#     The smoother and more repetitive the local basin, the harder the controller
#     is forced to jump out — directly targeting the function-word-swamp loop.
#
# FALSIFIABILITY GUARD: the null operator remains VEMAW-the-surrogate. The
# generator is "flow-cone + lambda-anneal", a DIFFERENT object that must beat it.
# Each knob is ablatable alone so we can tell which (if either) is load-bearing.
# Kill-criterion pre-registered in H-V25-003 below.
# ══════════════════════════════════════════════════════════════════════════════

# ── Flow field parameters ─────────────────────────────────────────────────────
FLOW_NJ          = 24        # j1 cells (15° each)
FLOW_NK          = 16        # ksi cells over [0,1]
FLOW_MIN_COUNT   = 4         # cells with fewer bigrams than this are "unobserved"
FLOW_WEIGHT      = 1.0       # blend weight of flow axis vs momentum/tpci (set per-arm)

# ── Inverted-VEMAW lambda controller parameters ──────────────────────────────
LAM_MIN          = 0.05      # floor: most-committed (near-argmax) when very rough
LAM_MAX          = 0.85      # ceiling: most-exploratory when very smooth/stuck
LAM_SCHEDULE     = 'log'     # reuse VEMAW temperature schedule
# Roughness inputs are normalized to [0,1]; smooth=0 -> lam high, rough=1 -> lam low.
# We also fold in a "stuck" term: recent low-displacement / high-revisit raises lam.
LAM_STUCK_GAIN   = 0.50      # how much recent stuckness pushes toward exploration


class FlowField:
    """
    KNOB 1. Transition bivector field on the (j1, ksi) manifold.

    Built from a corpus: tokenize -> map each token id to its engram (j1, ksi)
    address -> bin into FLOW_NJ x FLOW_NK cells -> accumulate the mean outgoing
    displacement (circular mean for j1, linear mean for ksi) per source cell.

    query(j1, ksi) -> (dj1, dksi, confidence)
      dj1, dksi : preferred geometric displacement to the successor neighborhood
      confidence: in [0,1], low where the cell is sparsely observed (back-off)

    This is the geometric successor model. It carries direction only.
    """
    def __init__(self, nj=FLOW_NJ, nk=FLOW_NK):
        self.nj = nj; self.nk = nk
        self.sin = np.zeros((nj, nk), np.float64)   # for circular mean of dj1
        self.cos = np.zeros((nj, nk), np.float64)
        self.dk  = np.zeros((nj, nk), np.float64)    # accumulate dksi
        self.cnt = np.zeros((nj, nk), np.int64)
        self.built = False

    def _cell(self, j1, ksi):
        cj = int((float(j1) % 360.) / 360. * self.nj) % self.nj
        ck = int(max(0., min(0.999999, float(ksi))) * self.nk)
        return cj, ck

    def add_bigram(self, j1a, ksia, j1b, ksib):
        cj, ck = self._cell(j1a, ksia)
        dj = math.radians(ang_diff(j1a, j1b))   # signed angular step a->b
        self.sin[cj, ck] += math.sin(dj)
        self.cos[cj, ck] += math.cos(dj)
        self.dk[cj, ck]  += (float(ksib) - float(ksia))
        self.cnt[cj, ck] += 1

    def finalize(self):
        self.built = True

    def query(self, j1, ksi):
        cj, ck = self._cell(j1, ksi)
        c = self.cnt[cj, ck]
        if c < FLOW_MIN_COUNT:
            return 0.0, 0.0, 0.0
        mean_dj1 = math.degrees(math.atan2(self.sin[cj, ck], self.cos[cj, ck]))
        mean_dk  = self.dk[cj, ck] / c
        # confidence: saturating in count, plus resultant-length sharpness for dj1
        R = math.hypot(self.sin[cj, ck], self.cos[cj, ck]) / c   # [0,1] directional concentration
        conf = (1. - math.exp(-c / 25.)) * (0.5 + 0.5 * R)
        return mean_dj1, mean_dk, float(conf)

    def coverage(self):
        return float((self.cnt >= FLOW_MIN_COUNT).mean())

    def velocity_grids(self):
        """
        Return the flow as explicit velocity-component grids on the manifold:
          u_j1[cj,ck] : mean j1-displacement (degrees) at each cell   (phase axis)
          u_k [cj,ck] : mean ksi-displacement at each cell            (advance axis)
          conf[cj,ck] : per-cell confidence (count + directional concentration)
        Cells below FLOW_MIN_COUNT report zero velocity and zero confidence.
        This is u(j1, ksi) — the vector field whose curl/helicity v26 computes.
        """
        nj, nk = self.nj, self.nk
        u_j1 = np.zeros((nj, nk)); u_k = np.zeros((nj, nk)); conf = np.zeros((nj, nk))
        for cj in range(nj):
            for ck in range(nk):
                c = self.cnt[cj, ck]
                if c < FLOW_MIN_COUNT:
                    continue
                u_j1[cj, ck] = math.degrees(math.atan2(self.sin[cj, ck], self.cos[cj, ck]))
                u_k[cj, ck]  = self.dk[cj, ck] / c
                R = math.hypot(self.sin[cj, ck], self.cos[cj, ck]) / c
                conf[cj, ck] = (1. - math.exp(-c / 25.)) * (0.5 + 0.5 * R)
        return u_j1, u_k, conf


def build_flow_field(corpus_path, tokenizer, eng, shuffled=False):
    """
    Build a FlowField from a text corpus.

    Each consecutive token pair (a, b) in the corpus contributes one bigram,
    placed at token a's engram (j1, ksi) address with displacement to token b's.

    shuffled=True : DESTROYS the successor relation by permuting the b-addresses
                    across all bigrams. This is the flow-field null arm — it keeps
                    the marginal displacement statistics but breaks which source
                    actually leads to which target. Used by H-V25-PHASE.
    """
    ff = FlowField()
    if tokenizer is None or not Path(corpus_path).exists():
        ff.finalize(); return ff
    text = Path(corpus_path).read_text(encoding='utf-8', errors='ignore')
    ids = tokenizer(text)['input_ids']
    j1v = eng['j1_v']; ksv = eng['ksi_v']; n = eng['n']
    pairs = []
    for t in range(len(ids) - 1):
        a, b = ids[t], ids[t+1]
        if a >= n or b >= n: continue
        pairs.append((float(j1v[a]), float(ksv[a]), float(j1v[b]), float(ksv[b])))
    if shuffled and pairs:
        srcs = [(p[0], p[1]) for p in pairs]
        tgts = [(p[2], p[3]) for p in pairs]
        random.shuffle(tgts)
        pairs = [(s[0], s[1], t[0], t[1]) for s, t in zip(srcs, tgts)]
    for j1a, ksia, j1b, ksib in pairs:
        ff.add_bigram(j1a, ksia, j1b, ksib)
    ff.finalize()
    return ff


# ══════════════════════════════════════════════════════════════════════════════
# v26 ADDITIONS — ML-101  : HELICITY FLOW MECHANISM
# ══════════════════════════════════════════════════════════════════════════════
#
# KNOB 3 — HELICITY (the braid-structured flow interface)
#
# The FlowField gives a velocity field u(j1, ksi) on the manifold. v25 used only
# its DIRECTION. v26 uses its ROTATION:
#
#   vorticity      omega(j1,ksi) = d(u_j1)/d(ksi) - d(u_k)/d(j1)   [scalar curl in 2D]
#   helicity dens. h(j1,ksi)     = u_j1 * omega                    [velocity·rotation]
#
# Helicity is a SIGNED pseudoscalar. Its sign labels the winding handedness of the
# local flow channel. Same-sign channels co-flow; opposite-sign channels are
# separated by a sign wall (h crosses zero) that behaves as a topological barrier.
# Steering ALONG the helicity gradient (toward stronger same-sign channel, away
# from a sign-flip) follows the braid rather than the marginal displacement.
#
# This is the distinction from the plain flow field: a marginal-displacement
# average (what NULLed on Laws / Mutual Aid) destroys braid structure, because
# averaging displacements washes out the curl. If helicity is load-bearing, the
# helicity computed from the REAL flow must beat helicity computed from a SHUFFLED
# flow (shuffle-then-curl) — H-V26-PHASE.
#
# COUPLING kappa: from CORE-003 Helical Calculus, kappa is the dimensionless
# winding parameter (arc-length excess). It is domain-specific. We default to the
# certified manifold constant alpha_physical = 0.4784; kappa_crit = 0.3 (the "10%
# excess" reference from the calculus) is exposed as an alternative. kappa enters
# via the helical-derivative normalization 1/sqrt(1 + kappa^2 * omega^2): it damps
# the steering where vorticity is large, exactly the calculus's covariant scaling.
#
# ATTRIBUTION (your trichotomy, frozen into the verdict — see run_ablation):
#   GEOMETRIC : real helicity beats shuffled-flow helicity AND lifts content where
#               the plain flow field failed -> kappa links the fractal geometrically.
#   MAP-ONLY  : helicity ~ flow field, both > their shuffles, but helicity adds
#               nothing over plain flow -> the engram is a record of where things
#               have been (history map), not live geometry.
#   ADDRESSING: helicity ~ its own shuffle (curl is noise) -> not an engram of
#               geometry; a digital fractal whose 2^160 address space is incidental.
# ══════════════════════════════════════════════════════════════════════════════

ALPHA_PHYSICAL   = 0.4784    # certified manifold constant (default kappa)
KAPPA_CRIT       = 0.30      # helical-calculus 10%-arc-excess reference
HELI_WEIGHT      = 1.0       # blend weight of helicity axis into cone axis
HELI_WALL_GAIN   = 0.60      # how hard a same-sign preference repels sign-flips


class HelicityField:
    """
    KNOB 3. Computes vorticity and helicity from a FlowField's velocity grids,
    then exposes a steering direction (helicity gradient, sign-walled) per query.

    Pure geometry: everything here is derived from u(j1,ksi) by finite-difference
    curl on the manifold grid. No corpus counts beyond what FlowField already held.
    """
    def __init__(self, flow_field, kappa=ALPHA_PHYSICAL):
        self.kappa = float(kappa)
        self.nj = flow_field.nj; self.nk = flow_field.nk
        self.built = False
        if flow_field is None or not flow_field.built:
            return
        u_j1, u_k, conf = flow_field.velocity_grids()
        self.u_j1 = u_j1; self.u_k = u_k; self.conf = conf
        self._compute_curl_and_helicity()
        self.built = True

    def _compute_curl_and_helicity(self):
        nj, nk = self.nj, self.nk
        # Finite-difference curl on the (j1 periodic, ksi non-periodic) grid.
        # d(u_j1)/d(ksi): central diff along ksi (axis 1, clamped at edges)
        # d(u_k)/d(j1):   central diff along j1   (axis 0, wraps — j1 is periodic)
        dphi = 360.0 / nj          # degrees per j1 cell
        dksi = 1.0 / nk            # ksi units per cell
        du_j1_dksi = np.zeros((nj, nk))
        du_k_dj1   = np.zeros((nj, nk))
        for cj in range(nj):
            for ck in range(nk):
                ck_p = min(ck + 1, nk - 1); ck_m = max(ck - 1, 0)
                span_k = (ck_p - ck_m) * dksi or dksi
                du_j1_dksi[cj, ck] = (self.u_j1[cj, ck_p] - self.u_j1[cj, ck_m]) / span_k
                cj_p = (cj + 1) % nj; cj_m = (cj - 1) % nj
                # u_k difference across wrapped j1 neighbors
                du_k_dj1[cj, ck] = (self.u_k[cj_p, ck] - self.u_k[cj_m, ck]) / (2 * dphi)
        self.omega = du_j1_dksi - du_k_dj1                 # scalar vorticity
        self.helicity = self.u_j1 * self.omega             # velocity·rotation
        # Helicity gradient (steering): central diff of helicity over the grid
        self.h_grad_j1 = np.zeros((nj, nk))
        self.h_grad_k  = np.zeros((nj, nk))
        for cj in range(nj):
            for ck in range(nk):
                cj_p = (cj + 1) % nj; cj_m = (cj - 1) % nj
                self.h_grad_j1[cj, ck] = (self.helicity[cj_p, ck] - self.helicity[cj_m, ck]) / (2 * dphi)
                ck_p = min(ck + 1, nk - 1); ck_m = max(ck - 1, 0)
                span_k = (ck_p - ck_m) * dksi or dksi
                self.h_grad_k[cj, ck] = (self.helicity[cj, ck_p] - self.helicity[cj, ck_m]) / span_k

    def _cell(self, j1, ksi):
        cj = int((float(j1) % 360.) / 360. * self.nj) % self.nj
        ck = int(max(0., min(0.999999, float(ksi))) * self.nk)
        return cj, ck

    def query(self, j1, ksi):
        """
        Returns (steer_dj1, steer_dksi, helicity_sign, confidence):
          steer_dj1/dksi : direction up the helicity gradient (follow the channel),
                           normalized by the helical-calculus factor
                           1/sqrt(1 + kappa^2 * omega^2).
          helicity_sign  : sign of local helicity (+1/-1/0) — channel handedness.
          confidence     : flow confidence at this cell.
        """
        if not self.built:
            return 0.0, 0.0, 0.0, 0.0
        cj, ck = self._cell(j1, ksi)
        conf = float(self.conf[cj, ck])
        if conf <= 0.:
            return 0.0, 0.0, 0.0, 0.0
        omega = float(self.omega[cj, ck])
        norm = 1.0 / math.sqrt(1.0 + (self.kappa ** 2) * (omega ** 2))   # helical-calculus scaling
        gj = float(self.h_grad_j1[cj, ck]) * norm
        gk = float(self.h_grad_k[cj, ck])  * norm
        # Convert the j1-gradient into a steering angle offset (degrees), bounded.
        steer_dj1 = max(-90., min(90., gj))
        steer_dk  = max(-0.2, min(0.2, gk))
        h_sign = float(np.sign(self.helicity[cj, ck]))
        return steer_dj1, steer_dk, h_sign, conf

    # ── Trough / wound-channel diagnostic ─────────────────────────────────────
    def trough_report(self, shuffled_helicity=None):
        """
        H-V26-TROUGH. Are high-|omega| cells organized as CONNECTED channels
        (criss-crossed wound channels scratched into the substrate) or scattered
        specks (binning noise)?

        Method: threshold |vorticity| at its 75th percentile over covered cells.
        Count connected components among the supra-threshold cells (4-neighbour,
        j1 wrapping). A connectivity ratio = (cells in components of size>=3) /
        (total supra-threshold cells). High ratio => channels; low => specks.
        Compare against a shuffled-helicity baseline if provided.
        """
        mask_cov = self.conf > 0.
        vort = np.abs(self.omega)[mask_cov]
        if vort.size == 0:
            return {'status': 'EMPTY'}
        thr = float(np.percentile(vort, 75))
        supra = (np.abs(self.omega) >= thr) & mask_cov
        ratio = self._connectivity_ratio(supra)
        out = {
            'threshold_75pct_vort': round(thr, 4),
            'n_supra': int(supra.sum()),
            'connectivity_ratio': round(ratio, 4),
        }
        if shuffled_helicity is not None and shuffled_helicity.built:
            smask = shuffled_helicity.conf > 0.
            sthr = float(np.percentile(np.abs(shuffled_helicity.omega)[smask], 75)) if smask.any() else 0.
            ssupra = (np.abs(shuffled_helicity.omega) >= sthr) & smask
            sratio = shuffled_helicity._connectivity_ratio(ssupra)
            out['shuffled_connectivity_ratio'] = round(sratio, 4)
            out['status'] = 'CHANNELS' if ratio > sratio * 1.10 else 'SPECKS'
        else:
            out['status'] = 'CHANNELS' if ratio > 0.5 else 'SPECKS'
        return out

    def _connectivity_ratio(self, supra):
        nj, nk = supra.shape
        seen = np.zeros_like(supra, dtype=bool)
        comp_sizes = []
        for cj in range(nj):
            for ck in range(nk):
                if supra[cj, ck] and not seen[cj, ck]:
                    # BFS flood fill, j1 wraps (axis 0), ksi clamps (axis 1)
                    stack = [(cj, ck)]; seen[cj, ck] = True; size = 0
                    while stack:
                        aj, ak = stack.pop(); size += 1
                        nbrs = [((aj+1) % nj, ak), ((aj-1) % nj, ak)]
                        if ak+1 < nk: nbrs.append((aj, ak+1))
                        if ak-1 >= 0: nbrs.append((aj, ak-1))
                        for bj, bk in nbrs:
                            if supra[bj, bk] and not seen[bj, bk]:
                                seen[bj, bk] = True; stack.append((bj, bk))
                    comp_sizes.append(size)
        total = int(supra.sum())
        if total == 0:
            return 0.0
        in_channels = sum(s for s in comp_sizes if s >= 3)
        return in_channels / total

    def _label_components(self, supra):
        """Return (label_grid, sizes) — component id per supra cell (0=none), and
        a dict {label: size}. j1 wraps, ksi clamps. Same connectivity as ratio."""
        nj, nk = supra.shape
        labels = np.zeros((nj, nk), dtype=int)
        sizes = {}
        cur = 0
        for cj in range(nj):
            for ck in range(nk):
                if supra[cj, ck] and labels[cj, ck] == 0:
                    cur += 1
                    stack = [(cj, ck)]; labels[cj, ck] = cur; size = 0
                    while stack:
                        aj, ak = stack.pop(); size += 1
                        nbrs = [((aj+1) % nj, ak), ((aj-1) % nj, ak)]
                        if ak+1 < nk: nbrs.append((aj, ak+1))
                        if ak-1 >= 0: nbrs.append((aj, ak-1))
                        for bj, bk in nbrs:
                            if supra[bj, bk] and labels[bj, bk] == 0:
                                labels[bj, bk] = cur; stack.append((bj, bk))
                    sizes[cur] = size
        return labels, sizes

    def scrapes_report(self, flow_field=None, shuffled_helicity=None):
        """
        H-SKIP — the 'scrapes' analysis. For every covered cell, relate:
          speed   |u| = hypot(u_j1_deg_as_frac, u_k)   (flow magnitude)
          vort    |omega|
          density cell bigram count (if flow_field given)
          channel membership: in a connected supra-|omega| component of size>=3

        Pre-registered:
          H-SKIP-001 (skipping): channel cells have LOWER speed than speck cells.
            metric: mean_speed_channel < mean_speed_speck, and point-biserial
            r(speed, in_channel) < 0. Reported with a 200x label-shuffle null band.
          H-SKIP-002 (density confound): r(count, in_channel). If count explains
            membership and speed does not, skipping is rejected for data-density.
          H-SKIP-003 (size classes): histogram of component sizes — trimodal
            (troughs / channels / specks) vs smooth continuum.
        """
        nj, nk = self.nj, self.nk
        mask = self.conf > 0.
        if mask.sum() == 0:
            return {'status': 'EMPTY'}

        # Speed: convert u_j1 (degrees) to a fraction of a full turn so it's
        # commensurate with u_k (ksi units in [0,1]); then magnitude.
        speed = np.hypot(self.u_j1 / 360.0, self.u_k)

        # Channel membership from supra-threshold vorticity components (>=3)
        vort = np.abs(self.omega)
        thr = float(np.percentile(vort[mask], 75))
        supra = (vort >= thr) & mask
        labels, sizes = self._label_components(supra)
        in_channel = np.zeros((nj, nk), dtype=bool)
        for lab, sz in sizes.items():
            if sz >= 3:
                in_channel |= (labels == lab)

        cov = mask
        sp = speed[cov]
        ch = in_channel[cov].astype(float)
        vo = vort[cov]

        def pb_corr(x, y):
            # point-biserial = pearson with a 0/1 variable
            if x.std() < 1e-12 or y.std() < 1e-12: return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        r_speed = pb_corr(sp, ch)
        # H-SKIP-001 null: shuffle channel labels across covered cells 200x
        rng = np.random.default_rng(0)
        null_rs = []
        for _ in range(200):
            null_rs.append(pb_corr(sp, rng.permutation(ch)))
        null_lo, null_hi = float(np.percentile(null_rs, 2.5)), float(np.percentile(null_rs, 97.5))

        mean_sp_ch  = float(sp[ch == 1].mean()) if (ch == 1).any() else float('nan')
        mean_sp_spk = float(sp[ch == 0].mean()) if (ch == 0).any() else float('nan')

        # H-SKIP-002 density confound
        r_count = None; mean_ct_ch = None; mean_ct_spk = None
        if flow_field is not None:
            cnt = flow_field.cnt.astype(float)
            ct = cnt[cov]
            r_count = pb_corr(ct, ch)
            mean_ct_ch  = float(ct[ch == 1].mean()) if (ch == 1).any() else float('nan')
            mean_ct_spk = float(ct[ch == 0].mean()) if (ch == 0).any() else float('nan')

        # H-SKIP-003 size classes
        size_list = sorted(sizes.values(), reverse=True)
        big      = [s for s in size_list if s >= 6]      # big troughs
        channels = [s for s in size_list if 3 <= s < 6]  # shallow channels
        specks   = [s for s in size_list if s < 3]       # specks

        # Verdicts
        skip_pass = (not math.isnan(mean_sp_ch) and not math.isnan(mean_sp_spk)
                     and mean_sp_ch < mean_sp_spk and r_speed < null_lo)
        density_explains = (r_count is not None and abs(r_count) > abs(r_speed)
                            and (r_count < null_lo or r_count > null_hi))

        if skip_pass and not density_explains:
            verdict = 'SKIPPING'          # channels are slow, specks are fast — confirmed
        elif density_explains and not skip_pass:
            verdict = 'DENSITY'           # split is data-count, not speed
        elif skip_pass and density_explains:
            verdict = 'SPEED+DENSITY'     # both move; report both
        else:
            verdict = 'NEITHER'           # no clean relation

        out = {
            'n_covered': int(cov.sum()),
            'n_supra': int(supra.sum()),
            'r_speed_channel': round(r_speed, 4),
            'speed_null_95ci': [round(null_lo, 4), round(null_hi, 4)],
            'mean_speed_channel': round(mean_sp_ch, 5) if not math.isnan(mean_sp_ch) else None,
            'mean_speed_speck':   round(mean_sp_spk, 5) if not math.isnan(mean_sp_spk) else None,
            'r_count_channel': round(r_count, 4) if r_count is not None else None,
            'mean_count_channel': round(mean_ct_ch, 2) if mean_ct_ch is not None else None,
            'mean_count_speck':   round(mean_ct_spk, 2) if mean_ct_spk is not None else None,
            'size_classes': {'big_troughs(>=6)': len(big),
                             'shallow_channels(3-5)': len(channels),
                             'specks(<3)': len(specks)},
            'component_sizes_sorted': size_list[:20],
            'H_SKIP_001_skipping': 'PASS' if skip_pass else 'NULL',
            'H_SKIP_002_density':  'EXPLAINS' if density_explains else 'no',
            'VERDICT': verdict,
        }
        return out


def build_helicity_field(flow_field, kappa=ALPHA_PHYSICAL):
    return HelicityField(flow_field, kappa=kappa)


class LambdaController:
    """
    KNOB 2. Inverted-VEMAW commitment-entropy controller.


    Wraps VEMAW's exact lambda->temperature math (log/linear/sigmoid schedules,
    copied from VEMAW_Core.EntropyController) but:
      (a) is fed CONE SCORES, not transition counts -> carries no linguistic info;
      (b) maps roughness -> lambda with the null's instinct INVERTED:
            smooth / stuck  -> high lambda -> explore (flatten the softmax)
            rough  / structured -> low lambda -> commit (sharpen toward argmax)

    softmax_select(scores, rng) returns an index into `scores`.
    """
    def __init__(self, schedule=LAM_SCHEDULE):
        self.schedule = schedule

    # --- VEMAW temperature math, lifted verbatim (do not edit: shared with null) ---
    def _temperature(self, lam):
        if self.schedule == 'log':
            if lam < 1e-10:   return 1e-10
            elif lam > 0.9999: return 1e10
            else:              return -1.0 / np.log(lam)
        elif self.schedule == 'linear':
            return lam / (1.0 - lam + 1e-10)
        elif self.schedule == 'sigmoid':
            x = (lam - 0.5) * 10
            return 1.0 / (1.0 + np.exp(-x))
        raise ValueError(f"Unknown schedule: {self.schedule}")

    def lambda_from_roughness(self, roughness, stuckness):
        """
        roughness in [0,1]: 0 = smooth basin, 1 = rough frontier.
        stuckness in [0,1]: 0 = moving/diverse, 1 = looping/over-visited.
        INVERTED map: smooth and/or stuck -> push lambda UP (explore).
        """
        smooth = 1. - max(0., min(1., roughness))
        stuck  = max(0., min(1., stuckness))
        drive  = max(smooth, LAM_STUCK_GAIN * stuck + (1. - LAM_STUCK_GAIN) * smooth)
        return LAM_MIN + (LAM_MAX - LAM_MIN) * max(0., min(1., drive))

    def softmax_select(self, scores, lam, rng):
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 1:
            return 0
        if lam <= LAM_MIN + 1e-6:
            return int(np.argmax(scores))      # MIMAW-like: commit
        T = self._temperature(lam)
        z = scores / max(T, 1e-9)
        z = z - z.max()
        p = np.exp(z); p = p / p.sum()
        return int(rng.choice(len(scores), p=p))


# ══════════════════════════════════════════════════════════════════════════════
# DEPOSIT FIELD
# ══════════════════════════════════════════════════════════════════════════════

class Deposits:
    def __init__(self):
        self.d = []; self.step = 0

    def add(self, j1, ksi, s=1.):
        self.d.append((float(j1)%360., float(ksi), float(s), self.step))

    def penalty(self, j1, ksi):
        p = 0.
        for dj, dk, s, t in self.d:
            dec = 0.5**((self.step-t)/DEPOSIT_HALF)
            dist = math.sqrt((ang_dist(j1,dj)/DEPOSIT_STD)**2+((ksi-dk)/0.15)**2)
            p += s * dec * math.exp(-0.5*dist**2)
        return p

    def penumbra(self, j1):
        b = 0.
        for dj, _, s, t in self.d:
            age = self.step - t
            if age > DEPOSIT_HALF*3: continue
            dj_ = ang_dist(j1, dj)
            if PENUMBRA_R*0.5 < dj_ < PENUMBRA_R*1.5:
                b += 0.3 * s * 0.5**(age/DEPOSIT_HALF)
        return b

    def tick(self):
        self.step += 1
        self.d = [(j,k,s,t) for j,k,s,t in self.d if self.step-t < DEPOSIT_HALF*4]

class BandDeposits:
    """
    Manifold-level angular band deposits.
    After selecting a token at j1_tok, deposits a Gaussian repulsion
    centered at j1_tok with width BAND_WIDTH degrees.
    This penalizes entire j1 neighborhoods, forcing the cone to rotate
    away and explore adjacent vocabulary before returning.

    Halflife controls exploration horizon:
      Short (3-5 steps): immediate word-level anti-repetition
      Long (10-15 steps): sentence-level register exploration
    """
    BAND_WIDTH   = 15.0   # degrees, Gaussian sigma of band repulsion
    BAND_HALF    = 5      # steps halflife (short: pushes arc away quickly)
    BAND_STRENGTH = 0.70  # max penalty [0,1] on cone_score at band center
    BAND_FLOOR   = 0.10   # minimum allowed cone_score multiplier (never fully blocked)

    def __init__(self):
        self.bands = []   # (j1_center, ksi_center, strength, step)
        self.step  = 0

    def add(self, j1, ksi, strength=1.0):
        self.bands.append((float(j1)%360., float(ksi), float(strength), self.step))

    def penalty_multiplier(self, tok_j1, tok_ksi):
        """
        Returns a multiplier in [BAND_FLOOR, 1.0].
        1.0 = no penalty (token far from all band centers).
        BAND_FLOOR = maximum penalty (token exactly at recently visited j1).
        """
        total_pen = 0.
        for bj1, bksi, strength, t in self.bands:
            age  = self.step - t
            dec  = 0.5**(age / self.BAND_HALF)
            j_dist = min(abs(tok_j1 - bj1), 360. - abs(tok_j1 - bj1))
            k_dist = abs(tok_ksi - bksi)
            gauss = math.exp(-0.5*(j_dist/self.BAND_WIDTH)**2
                             -0.5*(k_dist/0.12)**2)
            total_pen += strength * dec * gauss * self.BAND_STRENGTH
        return max(self.BAND_FLOOR, 1. - min(total_pen, 1. - self.BAND_FLOOR))

    def tick(self):
        self.step += 1
        self.bands = [(j,k,s,t) for j,k,s,t in self.bands
                      if self.step - t < self.BAND_HALF * 5]


def wada_seed(eng, phi_arc, dep, n=5):
    ksi_v = eng['ksi_v']; j1_v = eng['j1_v']
    km = (ksi_v >= 0.82) & (ksi_v <= 0.95)
    jm = np.array([ang_dist(float(j), phi_arc) for j in j1_v]) < 40.
    idx = np.where(km & jm)[0]
    if len(idx) == 0:
        c = np.abs(ksi_v-WADA_KSI)/0.1 + np.array([ang_dist(float(j),phi_arc) for j in j1_v])/30.
        idx = np.argsort(c)[:max(n*3,20)]
    dc = np.abs(ksi_v[idx]-WADA_KSI)/0.05 + np.array([ang_dist(float(j1_v[i]),phi_arc) for i in idx])/20.
    for i in np.argsort(dc)[:n]:
        dep.add(float(j1_v[idx[i]]), float(ksi_v[idx[i]]), 0.3)

# ══════════════════════════════════════════════════════════════════════════════
# SPIN DYNAMICS (directional gate, v20 certified; stays for spin tracking)
# ══════════════════════════════════════════════════════════════════════════════

def update_spin(spin, j1_current, j1_prev):
    spin += -K_POL * ang_diff(spin, THETA_LOCK)
    j_eff = float(j1_current) % 360.
    dj = ang_diff(j1_prev, j1_current)
    if ang_dist(j_eff, DEPOL_J1) < DEPOL_WIDTH and dj > 0 and j_eff < 140.:
        spin += XI_DEP * (1. - ang_dist(j_eff, DEPOL_J1) / DEPOL_WIDTH)
    return float(((spin+180) % 360) - 180)

# ══════════════════════════════════════════════════════════════════════════════
# ARC HISTORY & CONE DIRECTION (bivector, adapted from v23)
# ══════════════════════════════════════════════════════════════════════════════

class ArcHistory:
    """
    Tracks the sequence of token positions and computes the cone direction.
    In v24 this IS the arc — the cone direction is derived entirely from
    the history of token (j1, ksi) positions, no separate arc particle.
    """
    def __init__(self, shuffled=False):
        self.j1s  = deque(maxlen=K_HIST+1)
        self.ksis = deque(maxlen=K_HIST+1)
        self.spin_last = THETA_LOCK
        self.shuffled  = shuffled

    def push(self, j1, ksi, spin):
        self.j1s.append(float(j1))
        self.ksis.append(float(ksi))
        self.spin_last = float(spin)

    def dominant_j1s(self, n=2):
        """Most-visited j1 neighborhoods for Lagrange anti-attractor seeding."""
        if not self.j1s: return []
        from collections import Counter
        bins = [int(j//15)*15 for j in self.j1s]
        ctr = Counter(bins)
        return [b + 7.5 for b, _ in ctr.most_common(n)]

    def cone_direction(self, j1_current):
        """
        Returns (cone_axis_j1, R_spin, R_closure) where:
          cone_axis_j1 = j1 offset from forward arc direction
          R_spin       = spin correction (restoring force)
          R_closure    = closure pull toward Gold basin
        """
        # Arc direction from history
        if len(self.j1s) >= 2:
            diffs = [ang_diff(self.j1s[i], self.j1s[i+1])
                     for i in range(len(self.j1s)-1)]
            if self.shuffled:
                random.shuffle(diffs)
            mean_step = float(np.mean(diffs)) if diffs else 0.
            R_bigram = mean_step * W_BIGRAM
        else:
            R_bigram = 0.

        # Spin correction (negated — restoring force)
        spin_dev = ang_diff(THETA_LOCK, self.spin_last)
        R_spin   = -spin_dev * W_SPIN

        # Closure tether
        R_closure = ang_diff(j1_current, CLOSURE_J1) * W_CLOSURE

        cone_axis = (j1_current + R_bigram + R_spin + R_closure) % 360.
        return cone_axis, R_bigram, R_spin, R_closure

    def anchor(self):
        """
        Recency-weighted mean of last ANCHOR_K positions.
        Returns (mean_j1, mean_ksi) as the semantic anchor.
        """
        if len(self.j1s) == 0:
            return PHI_INIT, KSI_INIT
        # Weights: most recent = weight 1.0, older decay by ANCHOR_DECAY
        n = min(len(self.j1s), ANCHOR_K)
        weights = [ANCHOR_DECAY**(n-1-i) for i in range(n)]
        j1s  = list(self.j1s)[-n:]
        ksis = list(self.ksis)[-n:]
        # Circular mean for j1
        sins = sum(w*math.sin(math.radians(j)) for w,j in zip(weights,j1s))
        coss = sum(w*math.cos(math.radians(j)) for w,j in zip(weights,j1s))
        anch_j1  = math.degrees(math.atan2(sins, coss)) % 360.
        anch_ksi = sum(w*k for w,k in zip(weights,ksis)) / sum(weights)
        return anch_j1, anch_ksi

# ══════════════════════════════════════════════════════════════════════════════
# CONE ILLUMINATION + TPCI SCORING
# ══════════════════════════════════════════════════════════════════════════════

def ksi_target_at_phase(frac):
    """
    Dynamic Ksi target that varies with sentence phase.
    frac = step / sent_max (within current sentence)
    """
    if frac < 0.25:
        return KSI_INIT                                        # concrete opening
    elif frac < 0.85:
        # Linear rise to KSI_MID
        t = (frac - 0.25) / 0.60
        return KSI_INIT + t * (KSI_MID - KSI_INIT)            # elaboration
    else:
        # Fall back to concrete
        t = (frac - 0.85) / 0.15
        return KSI_MID - t * (KSI_MID - KSI_INIT)             # grounded close

def tpci_score(j1_h, ksi_h, h_norm, wp, ws, wf, wn, tj1, tksi):
    """Standard TPCI without the cone direction offset (cone pre-filters)."""
    def ca(t, a): return (1. + math.cos(math.radians(ang_diff(t, a)))) / 2.
    ka = ksi_h * 360.; sa = h_norm * 360.
    c = (wp*ca((j1_h+j1_h)%360, tj1) +    # pos triad (no phi_ticker in v24)
         ws*ca((j1_h+ka)%360, tksi*360.) +
         wf*ca((j1_h+ka)%360, tj1) +
         wn*ca((j1_h+sa)%360, tj1))
    return (1. + c/(wp+ws+wf+wn+1e-9)) / 2.

def get_cone_candidates(eng, j1_h, ksi_h, h_norm,
                         cone_axis, cone_aperture,
                         wp, ws, wf, wn,
                         dep, mass, spin,
                         ksi_target, anchor_j1, anchor_ksi,
                         burst_len=0, len_lut=None,
                         nc=300):
    """
    Cone-illumination candidate selection.

    1. Gather nc nearest candidates by j1 distance from j1_h.
    2. For each candidate, compute cone_score = direction × distance × amplification.
    3. TPCI × mass × deposit as secondary score.
    4. Combine: final = (cone_score + tpci_score + ksi_bonus + anchor_bonus) - deposit_penalty.

    Tokens outside cone_aperture are soft-penalized (not hard-excluded)
    unless CONE_STRICT=True.
    """
    j1v = eng['j1_v']; ksv = eng['ksi_v']

    # Gather broader candidate pool than before (cone will narrow it)
    dists = np.array([ang_dist(float(j), j1_h) for j in j1v])
    cands = np.argsort(dists)[:nc]

    res = []
    for idx in cands:
        tj1 = float(j1v[idx]); tksi = float(ksv[idx])
        if is_dead(tj1): continue

        # ── Cone direction score ───────────────────────────────────────────
        delta = ang_dist(tj1, cone_axis)       # angular offset from cone axis
        sigma = cone_aperture / 2.             # beam std

        if CONE_STRICT and delta > cone_aperture:
            continue                            # hard gate

        sigma_safe  = max(abs(sigma), 1.)   # guard against negative aperture
        direction_w = math.exp(-0.5 * (delta/sigma_safe)**2)

        # Distance from current position (tokens farther away decay)
        d_from_pos = ang_dist(tj1, j1_h)
        distance_w = math.exp(-d_from_pos / max(CONE_REACH, 1.))

        # Dynamic mass amplification (burst-position schedule)
        phase = burst_phase(burst_len)
        mass_w = MASS_SCHED[phase]
        ml = 1. + mass_w * (float(mass[idx]) if mass is not None else 0.5)
        amp = 1. + CONE_AMP * (ml - 1.) * (d_from_pos / max(CONE_REACH, 1.))

        # Length score (HH word-length schedule)
        if len_lut is not None and idx < len(len_lut):
            tok_len = int(len_lut[idx])
            target_len = LEN_SCHED[phase]
            len_score = math.exp(-abs(tok_len - target_len) / 2.5)
        else:
            len_score = 0.5

        cone_score = direction_w * distance_w * amp

        # ── TPCI score ────────────────────────────────────────────────────
        t = tpci_score(j1_h, ksi_h, h_norm, wp, ws, wf, wn, tj1, tksi)

        # ── Deposit field ─────────────────────────────────────────────────
        dp = dep.penalty(tj1, tksi)
        pb = dep.penumbra(tj1)

        # ── Spin gate: field-coherent particles only ─────────────────────
        if eng['has_spin']:
            tok_spin = math.degrees(float(eng['spin_angle'][idx]))
            spin_dev = abs(((tok_spin - THETA_LOCK + 180) % 360) - 180)
            # Hard gate: anti-aligned tokens are virtual particles — skip
            if spin_dev > SPIN_GATE_HARD:
                continue
            # Soft alignment score: smoothly rewards field-coherent tokens
            spin_align = max(0., 1. - spin_dev / SPIN_GATE_HARD)
        else:
            spin_dev   = 0.
            spin_align = 0.5

        # ── Dynamic Ksi target bonus ──────────────────────────────────────
        ksi_bonus = KSI_WEIGHT * math.exp(-0.5 * ((tksi-ksi_target)/KSI_SIGMA)**2)

        # ── Semantic anchor bonus ─────────────────────────────────────────
        j_dist = ang_dist(tj1, anchor_j1)
        k_dist = abs(tksi - anchor_ksi)
        anchor_bonus = ANCHOR_WEIGHT * math.exp(
            -0.5*(j_dist/ANCHOR_SIGMA)**2 - 0.5*(k_dist/ANCHOR_KSI_SIG)**2)

        # ── Final score ───────────────────────────────────────────────────
        final = (cone_score * 2.0 +     # cone is primary
                 t +                              # TPCI secondary (unpenalized)
                 SPIN_GATE_W * spin_align +        # spin coherence bonus
                 W_LEN * len_score +              # length schedule (cadence)
                 ksi_bonus +
                 anchor_bonus +
                 pb -
                 dp * 0.6)

        res.append((final, int(idx), tj1, tksi))

    res.sort(reverse=True)
    return res

def build_length_lut(tokenizer, n=50257):
    """
    Precompute character length for all token ids.
    This is the HH word-length predictor — deterministic, no training needed.
    A trained network predicting length from HH basin would rediscover this
    mapping, since mass × ksi × character-length are jointly correlated via Zipf.
    """
    lut = np.zeros(n, dtype=np.int32)
    for i in range(n):
        try:
            lut[i] = len(tokenizer.decode([i]).strip())
        except Exception:
            lut[i] = 0
    return lut

def burst_phase(burst_len):
    """Maps burst position to phrase-structure phase."""
    if burst_len <= 2: return 'open'
    if burst_len <= 6: return 'peak'
    return 'close'

def hebbian(wp, ws, wf, wn, j1h, ksih, frac):
    from_gold = ang_dist(j1h, Z3_GOLD)
    from_teal = ang_dist(j1h, Z3_TEAL)
    basin = 'gold' if from_gold < from_teal else 'teal'
    lr = 0.05
    z  = j1_zone(j1h)
    if z in ('A','B'):
        wp = min(2., wp + lr)
    elif z == 'DEAD':
        wn = min(2., wn + lr)
    else:
        ws = min(2., ws + lr)
    # Decay all toward 1.0 slowly
    for attr, val in [('wp',wp),('ws',ws),('wf',wf),('wn',wn)]:
        pass
    return wp, ws, wf, wn

# ══════════════════════════════════════════════════════════════════════════════
# CORE GENERATION v24
# ══════════════════════════════════════════════════════════════════════════════


def tpci_steered_cone_axis(j1_h, ksi_h, h_norm, wp, ws, wf, wn, dep, mass, spin,
                            n_probes=16, probe_radius=25.):
    """
    Compute the TPCI gradient direction at current position.
    Probe N angles at probe_radius degrees from j1_h.
    The probe angle with highest mean TPCI score becomes the cone axis.
    This lets the manifold's own structure steer the cone — no arc history needed.
    Combined with arc history for a weighted blend.
    """
    # Quick probe: evaluate mean TPCI for a small set of candidate tokens
    # at each probe direction. Use a coarse ring (fewer candidates per probe).
    best_angle = j1_h; best_score = -1.
    for i in range(n_probes):
        probe_angle = (j1_h + probe_radius * math.cos(2*math.pi*i/n_probes)) % 360.
        # Simple TPCI at the probe angle (using probe as the "token" address)
        t = tpci_score(j1_h, ksi_h, h_norm, wp, ws, wf, wn, probe_angle, ksi_h)
        if t > best_score:
            best_score = t; best_angle = probe_angle
    return best_angle

def tpci_lookahead(eng, j1_candidate, ksi_candidate, h_norm_cand,
                   wp, ws, wf, wn, dep, mass, spin,
                   aperture, n_probes=8):
    """
    THREE-STEP CYCLE — Step 2: TPCI Steering from candidate position.

    Given T_candidate at (j1_candidate, ksi_candidate), this function
    asks: if we were AT T_candidate, where would the TPCI gradient point?

    That answer is T_anchor — the position the manifold wants to go to
    FROM the candidate token.

    Returns (anchor_j1, anchor_ksi, anchor_score):
      anchor_j1/ksi = where TPCI points from T_candidate
      anchor_score  = strength of that coupling (how good the next step would be)

    Tokens whose look-ahead leads back to already-visited territory
    (penalized by deposit field) naturally score lower, because the
    deposit field is active at anchor evaluation time.

    This is self-regulating repetition control:
      - "each" at j1=130° → look-ahead from j1=130° finds best direction
      - If deposit already covers j1=130° neighborhood → anchor_score is low
      - "each" is downranked without any explicit repetition rule
      - No halflife parameter, no strength parameter — emerges from geometry
    """
    best_j1 = j1_candidate
    best_ksi = ksi_candidate
    best_score = -1.

    for i in range(n_probes):
        probe_j1 = (j1_candidate + aperture * math.cos(2*math.pi*i/n_probes)) % 360.
        probe_ksi = max(0., min(1., ksi_candidate + 0.15*math.sin(2*math.pi*i/n_probes)))

        # TPCI at probe point (what would the manifold couple to from here?)
        t = tpci_score(j1_candidate, ksi_candidate, h_norm_cand,
                       wp, ws, wf, wn, probe_j1, probe_ksi)

        # Apply deposit penalty — probes in visited territory score lower
        dep_pen = dep.penalty(probe_j1, probe_ksi)
        t_adjusted = t - dep_pen * 0.4

        if t_adjusted > best_score:
            best_score = t_adjusted
            best_j1 = probe_j1
            best_ksi = probe_ksi

    return best_j1, best_ksi, best_score



def run_v24(eng, stiff_lut, helic_lut, mass, tokenizer,
            prompt_ids, n_tokens,
            j1_init=PHI_INIT, ksi_init=KSI_INIT, sent_max=SENT_MAX,
            shuffled=False, mode='cone',
            use_flow=False, use_lambda=False, flow_field=None,
            use_helicity=False, helicity_field=None,
            rng=None,
            verbose=True):
    """
    v24/v25: cone illumination architecture.

    mode='cone': cone-stopped architecture
    v25 knobs (independently ablatable):
      use_flow   : direction from the FlowField geometric successor model
      use_lambda : commitment entropy from the inverted-VEMAW controller
      flow_field : a built FlowField (required if use_flow=True)
    """
    if rng is None:
        rng = np.random.default_rng()
    lam_ctrl = LambdaController() if use_lambda else None
    last_lam = None
    lam_trace = []

    # Position = last token's address (start at init)
    j1_h  = float(j1_init)
    ksi_h = float(ksi_init)
    spin  = THETA_LOCK
    j1_prev = j1_h

    dep  = Deposits()
    wada_seed(eng, j1_h, dep)

    # Precompute length LUT (HH word-length schedule)
    len_lut = build_length_lut(tokenizer) if tokenizer else None

    hist = ArcHistory(shuffled=shuffled)
    hist.push(j1_h, ksi_h, spin)

    wp = ws = wf = wn = 1.

    # Tracking
    burst_len = 0; noise_run = 0; sent_tok = 0; n_sent = 0
    in_burst = False; burst_zone = 'OTHER'
    burst_dj1_vals = []; current_burst_j1s = []
    ksi_phase_records = {'start': [], 'mid': []}  # for H-V24-002
    n_noise = 0; n_total = 0

    tokens_out = []

    if verbose:
        print(f"\n  mode={mode}  j1_init={j1_init}°  ksi_init={ksi_init}  shuffled={shuffled}", flush=True)
        print(f"  {'Stp':>3} {'j1':>7} {'ksi':>5} {'Zone':>5} {'θ':>5} "
              f"{'Ksi_T':>6} {'Spin':>7}  Token", flush=True)
        print("  "+"-"*65, flush=True)

    for step in range(n_tokens):
        frac_total = step / max(n_tokens, 1)
        frac_sent  = sent_tok / max(sent_max, 1)

        zone_now   = j1_zone(j1_h)
        base_ap    = ZONE_APERTURE.get(zone_now, CONE_BASE)
        stiff_here = max(0., min(1., lut_get(j1_h, stiff_lut)))
        aperture   = max(8., base_ap * (1. + WINDING_SCALE * (1. - stiff_here)))

        # ── Cone direction: blend TPCI gradient + arc history ────────────
        # TPCI gradient: where does the manifold want to go from here?
        tpci_axis = tpci_steered_cone_axis(
            j1_h, ksi_h, lut_get(j1_h, helic_lut),
            wp, ws, wf, wn, dep, mass, spin,
            n_probes=12, probe_radius=aperture)
        # Arc history direction (bivector from past tokens)
        arc_axis, R_big, R_spn, R_clos = hist.cone_direction(j1_h)
        # Blend: 60% TPCI gradient, 40% arc history
        # TPCI gives manifold structure; arc gives temporal direction
        tpci_w = 0.60; arc_w = 0.40
        # Circular blend via mean of unit vectors
        sin_c = tpci_w*math.sin(math.radians(tpci_axis)) + arc_w*math.sin(math.radians(arc_axis))
        cos_c = tpci_w*math.cos(math.radians(tpci_axis)) + arc_w*math.cos(math.radians(arc_axis))
        cone_axis = math.degrees(math.atan2(sin_c, cos_c)) % 360.

        # ── KNOB 1: FLOW FIELD direction (geometric successor model) ─────
        flow_conf = 0.0; flow_dj1 = 0.0; flow_dk = 0.0
        if use_flow and flow_field is not None and flow_field.built:
            flow_dj1, flow_dk, flow_conf = flow_field.query(j1_h, ksi_h)
            if flow_conf > 0.:
                # Flow proposes an absolute target direction from current position.
                flow_axis = (j1_h + flow_dj1) % 360.
                # Confidence-weighted circular blend into the existing cone axis.
                w_flow = FLOW_WEIGHT * flow_conf
                s = (1. - w_flow)*math.sin(math.radians(cone_axis)) + w_flow*math.sin(math.radians(flow_axis))
                c = (1. - w_flow)*math.cos(math.radians(cone_axis)) + w_flow*math.cos(math.radians(flow_axis))
                cone_axis = math.degrees(math.atan2(s, c)) % 360.

        # ── KNOB 3: HELICITY steering (braid-structured flow) ────────────
        heli_conf = 0.0; heli_sign = 0.0
        if use_helicity and helicity_field is not None and helicity_field.built:
            steer_dj1, steer_dk, heli_sign, heli_conf = helicity_field.query(j1_h, ksi_h)
            if heli_conf > 0.:
                # Steer along the helicity gradient: follow the wound channel.
                heli_axis = (j1_h + steer_dj1) % 360.
                w_heli = HELI_WEIGHT * heli_conf
                s = (1. - w_heli)*math.sin(math.radians(cone_axis)) + w_heli*math.sin(math.radians(heli_axis))
                c = (1. - w_heli)*math.cos(math.radians(cone_axis)) + w_heli*math.cos(math.radians(heli_axis))
                cone_axis = math.degrees(math.atan2(s, c)) % 360.
                # Sign-wall: record current channel handedness so candidate scoring
                # can prefer same-sign cells (set on eng-side via heli_sign below).
                heli_steer_dk = steer_dk
            else:
                heli_steer_dk = 0.0
        else:
            heli_steer_dk = 0.0

        # ── Cone aperture from zone + winding ─────────────────────────────


        # ── Small stochastic perturbation on cone axis ────────────────────
        # Without this, system is fully deterministic (same run = same output).
        # Epsilon perturbation breaks symmetry while staying near the manifold.
        # Scale: 3-5° is small enough not to disrupt cone direction,
        # large enough to create variation across runs.
        epsilon = random.gauss(0., 3.0)   # 3° std perturbation
        cone_axis = (cone_axis + epsilon) % 360.

        # ── Dynamic Ksi target (two-timescale cadence) ──────────────────
        # Slow arc: sentence-level ksi rise over 82 steps
        sent_ksi = ksi_target_at_phase(frac_sent)
        # Fast arc: burst-triggered lift into content word territory
        if in_burst and burst_len >= KSI_BURST_MIN:
            ksi_target = max(sent_ksi, KSI_BURST)  # lift, never lower than slow arc
        else:
            ksi_target = sent_ksi

        # KNOB 1 (cont.): let the flow field nudge the ksi register target too.
        if use_flow and flow_conf > 0.:
            ksi_target = max(0., min(1., ksi_target + flow_conf * flow_dk))
        # KNOB 3 (cont.): helicity gradient nudges ksi along the channel too.
        if use_helicity and heli_conf > 0.:
            ksi_target = max(0., min(1., ksi_target + heli_conf * heli_steer_dk))

        # ── Semantic anchor ────────────────────────────────────────────────
        anch_j1, anch_ksi = hist.anchor()

        # ── Spin update (using token-to-token jumps, no arc particle) ─────
        spin = update_spin(spin, j1_h, j1_prev)

        # ── Select token ──────────────────────────────────────────────────
        h_norm = lut_get(j1_h, helic_lut)
        cands  = get_cone_candidates(
            eng, j1_h, ksi_h, h_norm,
            cone_axis, aperture,
            wp, ws, wf, wn,
            dep, mass, spin,
            ksi_target, anch_j1, anch_ksi,
            burst_len=burst_len, len_lut=len_lut)

        # ── THREE-STEP CYCLE ──────────────────────────────────────────────
        # Step 1 (GUESS): cone candidates — top N by cone+TPCI score
        # Step 2 (STEER): from each candidate, run TPCI look-ahead
        #                 find where the manifold would point NEXT
        # Step 3 (SELECT): pick candidate whose look-ahead anchor is best
        #                  (good now AND opens good territory for next step)

        # Gather top candidates (more than we need — look-ahead narrows them)
        N_LOOKAHEAD = 8   # how many candidates to evaluate with look-ahead
        alpha_cands = []
        if tokenizer:
            for sc, ci, tj1, tksi in cands:
                tok_txt = tokenizer.decode([ci])
                if not is_alpha_token(tok_txt): continue
                alpha_cands.append((sc, ci, tj1, tksi, tok_txt))
                if len(alpha_cands) >= N_LOOKAHEAD: break
        else:
            for sc, ci, tj1, tksi in cands[:N_LOOKAHEAD]:
                alpha_cands.append((sc, ci, tj1, tksi, f"[{ci}]"))

        if not alpha_cands: continue

        # Step 2+3: score each candidate by look-ahead quality
        h_norm_here = lut_get(j1_h, helic_lut)
        scored = []   # (combined, cone_sc, ci, tj1, tksi, tok_txt)
        for (cone_sc, ci, tj1, tksi, tok_txt) in alpha_cands:
            # From this candidate, where does the manifold want to go?
            anch_j1, anch_ksi, anch_score = tpci_lookahead(
                eng, tj1, tksi, lut_get(tj1, helic_lut),
                wp, ws, wf, wn, dep, mass, spin,
                aperture, n_probes=8)
            # Combined score: cone quality NOW + look-ahead quality NEXT
            combined = cone_sc * 0.60 + anch_score * 0.40
            # KNOB 3 sign-wall: prefer candidates on the SAME helicity channel as
            # the current position; repel candidates across a sign-flip (the wall).
            if use_helicity and helicity_field is not None and helicity_field.built and heli_conf > 0.:
                _, _, cand_sign, cand_conf = helicity_field.query(tj1, tksi)
                if cand_conf > 0. and heli_sign != 0.:
                    if cand_sign == heli_sign:
                        combined *= (1. + HELI_WALL_GAIN * cand_conf)      # co-flow
                    elif cand_sign == -heli_sign:
                        combined *= (1. - HELI_WALL_GAIN * cand_conf)      # wall
            scored.append((combined, cone_sc, ci, tj1, tksi, tok_txt))

        if not scored: continue

        if use_lambda:
            # ── KNOB 2: inverted-VEMAW commitment-entropy selection ───────
            # Roughness in [0,1]: high stiffness + tight aperture = SMOOTH (->0),
            # low stiffness / wide aperture / dead-zone edge = ROUGH (->1).
            # We also treat HIGH flow confidence as "structured" (rough side),
            # so a confident successor signal makes us commit, not wander.
            rough_geom = 1. - max(0., min(1., stiff_here))
            rough = max(rough_geom, flow_conf)   # confident flow => commit
            # Stuckness in [0,1]: recent j1 revisit density near current position.
            recent = list(hist.j1s)[-ANCHOR_K:]
            if len(recent) >= 3:
                near = sum(1 for j in recent if ang_dist(j, j1_h) < BandDeposits.BAND_WIDTH)
                stuck = near / len(recent)
            else:
                stuck = 0.0
            lam = lam_ctrl.lambda_from_roughness(rough, stuck)
            last_lam = lam; lam_trace.append(lam)
            sel = lam_ctrl.softmax_select([s[0] for s in scored], lam, rng)
            _, cone_sc, tok_id, tok_j1, tok_ksi, tok_text = scored[sel]
        else:
            # v24 behavior: deterministic argmax over combined score.
            best = max(scored, key=lambda x: x[0])
            _, cone_sc, tok_id, tok_j1, tok_ksi, tok_text = best

        sc = cone_sc

        # ── v24 CORE: stop at token position ─────────────────────────────
        j1_prev = j1_h
        j1_h    = tok_j1   # TELEPORT — arc is now the token sequence
        ksi_h   = tok_ksi

        # Update history
        hist.push(j1_h, ksi_h, spin)

        # ── Burst / zone-pure tracking (carrier-aware) ──────────────────
        zone = j1_zone(tok_j1)
        charge = token_charge(tok_text) if tokenizer else 'CONTENT'
        zone_changed = in_burst and (zone != burst_zone)

        n_total += 1
        if not is_alpha_token(tok_text): n_noise += 1

        if charge == 'CONTENT' and not zone_changed:
            burst_len += 1; noise_run = 0; in_burst = True
            burst_zone = zone if burst_len == 1 else burst_zone
            current_burst_j1s.append(tok_j1)
        elif charge == 'CARRIER':
            pass   # NEUTRAL: transparent to burst counting
        else:      # NOISE or zone change
            if in_burst and burst_len >= 3 and len(current_burst_j1s) >= 2:
                diffs = [ang_dist(current_burst_j1s[i], current_burst_j1s[i+1])
                         for i in range(len(current_burst_j1s)-1)]
                burst_dj1_vals.extend(diffs)
            current_burst_j1s = []
            burst_len = 0
            noise_run += (0 if zone_changed else 1)
            in_burst = False; burst_zone = zone

        # Ksi phase tracking for H-V24-002
        if frac_sent < 0.25:
            ksi_phase_records['start'].append(tok_ksi)
        elif frac_sent < 0.85:
            ksi_phase_records['mid'].append(tok_ksi)

        sent_tok += 1

        # ── Sentence rhythm ────────────────────────────────────────────────
        eff_max = sent_max + (15 if aperture < CONE_BASE * 0.8 else 0)
        sent_done = ((noise_run >= NOISE_BREAK_N and sent_tok > 5)
                     or sent_tok >= eff_max
                     or is_sent_end(tok_text))
        if sent_done and sent_tok >= 5:
            n_sent += 1; sent_tok = 0; noise_run = 0
            burst_len = 0; in_burst = False; current_burst_j1s = []
            burst_zone = 'OTHER'
            # Reset position to Zone A edge (calibrated)
            j1_h = 125.; ksi_h = max(0., min(1., ksi_h + 0.05))
            j1_prev = j1_h; spin = THETA_LOCK
            hist.push(j1_h, ksi_h, spin)
            wada_seed(eng, j1_h, dep, n=3)
            # Lagrange: repel from dominant j1 attractors
            for dom_j1 in hist.dominant_j1s():
                dep.add(dom_j1, ksi_h, 0.20)

        dep.add(tok_j1, tok_ksi); dep.tick()
        wp, ws, wf, wn = hebbian(wp, ws, wf, wn, tok_j1, tok_ksi, frac_total)

        if verbose:
            print(f"  {step:>3} {tok_j1:>7.1f}° {tok_ksi:>5.3f} {zone:>5} "
                  f"{aperture:>5.1f}° {ksi_target:>6.3f} {spin:>6.1f}°  "
                  f"'{tok_text.strip()[:28]}'", flush=True)

        tokens_out.append({
            'step': step, 'token_id': tok_id, 'token': tok_text,
            'j1': round(tok_j1, 2), 'ksi': round(tok_ksi, 4), 'zone': zone,
            'burst_len': burst_len, 'spin': round(spin, 2),
            'cone_axis': round(cone_axis, 2), 'aperture': round(aperture, 2),
            'ksi_target': round(ksi_target, 4),
            'anchor_j1': round(anch_j1, 2), 'anchor_ksi': round(anch_ksi, 4),
            'R_bigram': round(R_big, 3), 'R_spin': round(R_spn, 3),
            'R_closure': round(R_clos, 3),
            'sent_n': n_sent,
        })

    # ── POST ──────────────────────────────────────────────────────────────
    text   = ''.join(t['token'] for t in tokens_out)
    zones  = [t['zone'] for t in tokens_out]
    zone_c = {z: zones.count(z) for z in set(zones)}
    spins  = [t['spin'] for t in tokens_out]
    lock_f = sum(1 for s in spins if abs(ang_diff(s,THETA_LOCK))<30)/max(len(spins),1)

    words = text.split()
    bursts = []; in_b = False; bs = 0
    for i, w in enumerate(words):
        rw = is_real_word(w)
        if rw and not in_b: in_b = True; bs = i
        elif not rw and in_b:
            if i-bs >= 2: bursts.append(' '.join(words[bs:i]))
            in_b = False
    if in_b and len(words)-bs >= 2: bursts.append(' '.join(words[bs:]))

    mean_burst = float(np.mean([len(b.split()) for b in bursts])) if bursts else 0.
    max_burst  = max((len(b.split()) for b in bursts), default=0)
    ttr = len(set(t['token_id'] for t in tokens_out)) / max(len(tokens_out),1)
    noise_frac = n_noise / max(n_total, 1)
    mean_dj1   = float(np.mean(burst_dj1_vals)) if burst_dj1_vals else float('nan')
    # content fraction: share of emitted tokens that are CONTENT (not carrier/noise)
    n_content = sum(1 for t in tokens_out if token_charge(t['token']) == 'CONTENT')
    content_frac = n_content / max(len(tokens_out), 1)

    # H-V24-001
    h001 = 'PASS' if (not math.isnan(mean_dj1) and mean_dj1 < 15.) else 'NULL'
    # H-V24-002
    ks  = float(np.mean(ksi_phase_records['start'])) if ksi_phase_records['start'] else 0.
    km  = float(np.mean(ksi_phase_records['mid']))   if ksi_phase_records['mid']   else 0.
    h002 = 'PASS' if (ks < 0.60 and km > 0.60) else 'NULL'
    # H-V24-003
    h003 = 'PASS' if lock_f > 0.85 else 'NULL'
    # H-V24-004 evaluated in ablation

    hyp = {
        'H_V24_001': {'status': h001, 'mean_dj1': round(mean_dj1,3) if not math.isnan(mean_dj1) else None},
        'H_V24_002': {'status': h002, 'ksi_start': round(ks,4), 'ksi_mid': round(km,4)},
        'H_V24_003': {'status': h003, 'lock_frac': round(lock_f,4)},
    }

    if verbose:
        print(f"\n  ── V24 {mode.upper()} SUMMARY ──", flush=True)
        print(f"  text: {text[:200]}", flush=True)
        print(f"  zones:{zone_c}  lock={lock_f:.0%}  noise={noise_frac:.1%}  TTR={ttr:.3f}", flush=True)
        print(f"  max_burst={max_burst}  mean={mean_burst:.2f}  dj1={mean_dj1:.1f}°  sent={n_sent}", flush=True)
        for k, v in hyp.items():
            print(f"  {k}: {v['status']}", flush=True)
        for b in sorted(bursts, key=lambda x: -len(x.split()))[:3]:
            print(f"  [{len(b.split())}w] {b[:100]}", flush=True)

    return {'mode': mode, 'text': text, 'tokens': tokens_out, 'zone_counts': zone_c,
            'mean_burst': round(mean_burst,3), 'max_burst': max_burst, 'bursts': bursts,
            'ttr': round(ttr,4), 'lock_frac': round(lock_f,4),
            'noise_frac': round(noise_frac,4),
            'mean_burst_dj1': round(mean_dj1,3) if not math.isnan(mean_dj1) else None,
            'n_sentences': n_sent, 'hypotheses': hyp,
            'use_flow': use_flow, 'use_lambda': use_lambda,
            'use_helicity': use_helicity,
            'content_frac': round(content_frac,4),
            'lambda_mean': round(float(np.mean(lam_trace)),4) if lam_trace else None}

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION — v25 THREE-ARM DESIGN (ML-100)
# ══════════════════════════════════════════════════════════════════════════════
#
# PRE-REGISTERED (ML-100) — freeze before running.
#
#   Baseline  = v24r6 behavior (no flow, no lambda)  [deterministic argmax]
#   Arm L     = lambda only  (use_lambda=True, use_flow=False)
#   Arm F     = flow only    (use_flow=True,  use_lambda=False)
#   Arm FL    = both         (use_flow=True,  use_lambda=True)
#
# H-V25-001  FLOW RAISES CONTENT FRACTION
#   Arm F content_frac > baseline content_frac + 0.05, with noise_frac < 0.10.
#   (Direction from the geometric successor model carries selection out of the
#    function-word basin toward content territory.)
#
# H-V25-002  LAMBDA BREAKS THE LOOP (DIVERSITY)
#   Arm L ttr > baseline ttr × 1.20, with noise_frac < 0.10.
#   (Commitment-entropy annealing alone increases vocabulary diversity.)
#
# H-V25-003  KILL-CRITERION / GEOMETRY ATTRIBUTION  *** the uncomfortable one ***
#   IF Arm L alone achieves the diversity gain (H-V25-002 PASS) AND Arm F alone
#   does NOT raise content_frac (H-V25-001 FAIL), THEN the loop was a temperature
#   problem, not a geometry problem, and the geometric-successor thesis is NOT
#   supported by this experiment. Record as:
#       attribution = 'temperature'  (geometry not load-bearing here)   [FALSIFIED]
#   IF Arm F raises content_frac (H-V25-001 PASS):
#       attribution = 'geometry'     (flow field is doing real work)     [SUPPORTED]
#   IF both contribute and FL > each alone on its own metric:
#       attribution = 'both'
#
# H-V25-PHASE  FLOW STRUCTURE IS LOAD-BEARING  (flow-field phase-shuffle null)
#   Arm F (real flow) content_frac > Arm F (shuffled flow) content_frac × 1.10.
#   Shuffled flow keeps the marginal displacement statistics but destroys which
#   source cell leads to which target. If real ≈ shuffled, the flow field is only
#   re-expressing momentum / marginals, not genuine successor structure.
# ══════════════════════════════════════════════════════════════════════════════

def _print_arm(name, res):
    print(f"\n  ── {name} ──", flush=True)
    print(f"  content={res['content_frac']:.3f}  TTR={res['ttr']:.3f}  "
          f"max={res['max_burst']:2d}  mean={res['mean_burst']:.2f}  "
          f"lock={res['lock_frac']:.0%}  noise={res['noise_frac']:.1%}  "
          f"lam={res['lambda_mean'] if res['lambda_mean'] is not None else '—'}", flush=True)
    print(f"  text: {res['text'][:140]}", flush=True)

def run_ablation(eng, stiff, hel, mass, tok, n_tokens, out_path, prompt_ids=[],
                 flow_field=None, flow_field_shuf=None, seed=0,
                 helicity_field=None, helicity_field_shuf=None):
    """
    v26 ablation — ML-101.

    Arms: baseline / L_only / F_only / H_only / HL_both
          + F_shuffled (flow null) + H_shuffled (helicity null: shuffle-then-curl)

    PRE-REGISTERED (ML-101) — freeze before running:

    H-V26-001  HELICITY RAISES CONTENT WHERE PLAIN FLOW DID NOT
       H_only content_frac > baseline + 0.05, with noise_frac < 0.10.
       (The braid interface succeeds where marginal displacement failed.)

    H-V26-PHASE  BRAID STRUCTURE IS LOAD-BEARING
       H_only content_frac > H_shuffled content_frac × 1.10.
       H_shuffled curls a SHUFFLED flow field — same marginals, broken braid.
       If real ≈ shuffled, the curl is noise, not structure.

    H-V26-TROUGH  WOUND CHANNELS EXIST
       High-|vorticity| cells form connected channels vs a shuffled baseline.

    ATTRIBUTION TRICHOTOMY (Keaton's framing, frozen into the verdict):
       GEOMETRIC : H-V26-001 PASS and H-V26-PHASE PASS
                   -> kappa links the fractal geometrically. The big result.
       MAP-ONLY  : H-V26-PHASE PASS but H_only adds nothing over F_only
                   -> engram is a record of where things have been, not live geometry.
       ADDRESSING: H-V26-PHASE NULL (helicity ~ its own shuffle)
                   -> not an engram of geometry; a digital fractal, 2^160 addressing
                      space incidental to language.
    KILL-CRITERION: if H_only does no better than F_only on the corpora where the
       plain flow field already failed, helicity is NOT the missing interface — stop
       calling it one, even though it is the favored candidate.
    """
    def _run(**kw):
        return run_v24(eng, stiff, hel, mass, tok, prompt_ids, n_tokens,
                       verbose=False, rng=np.random.default_rng(seed), **kw)

    results = {}
    arms = {
        'baseline': dict(use_flow=False, use_lambda=False, use_helicity=False),
        'L_only':   dict(use_flow=False, use_lambda=True,  use_helicity=False),
        'F_only':   dict(use_flow=True,  use_lambda=False, use_helicity=False, flow_field=flow_field),
        'H_only':   dict(use_flow=False, use_lambda=False, use_helicity=True,  helicity_field=helicity_field),
        'HL_both':  dict(use_flow=False, use_lambda=True,  use_helicity=True,  helicity_field=helicity_field),
    }
    for name, kw in arms.items():
        results[name] = _run(**kw); _print_arm(name, results[name])

    # Null arms
    if flow_field_shuf is not None:
        results['F_shuffled'] = _run(use_flow=True, use_lambda=False, use_helicity=False,
                                     flow_field=flow_field_shuf)
        _print_arm('F_shuffled (flow null)', results['F_shuffled'])
    if helicity_field_shuf is not None:
        results['H_shuffled'] = _run(use_flow=False, use_lambda=False, use_helicity=True,
                                     helicity_field=helicity_field_shuf)
        _print_arm('H_shuffled (braid null: shuffle-then-curl)', results['H_shuffled'])

    base = results['baseline']; L = results['L_only']; F = results['F_only']
    H = results['H_only']; HL = results['HL_both']

    # ── Hypotheses ────────────────────────────────────────────────────────────
    h001 = 'PASS' if (H['content_frac'] > base['content_frac'] + 0.05
                      and H['noise_frac'] < 0.10) else 'NULL'
    h_phase = 'n/a'
    if 'H_shuffled' in results:
        h_phase = 'PASS' if H['content_frac'] > results['H_shuffled']['content_frac'] * 1.10 else 'NULL'

    # Helicity-over-flow margin: does the braid add anything beyond plain flow?
    heli_beats_flow = H['content_frac'] > F['content_frac'] + 0.03

    # Trough diagnostic
    trough = None
    if helicity_field is not None and helicity_field.built:
        trough = helicity_field.trough_report(shuffled_helicity=helicity_field_shuf)

    # ── ATTRIBUTION TRICHOTOMY ─────────────────────────────────────────────────
    if h_phase == 'PASS' and h001 == 'PASS':
        attribution = 'GEOMETRIC'      # the big result: kappa links the fractal
    elif h_phase == 'PASS' and not heli_beats_flow:
        attribution = 'MAP-ONLY'       # structure real but no lift over flow: history map
    elif h_phase == 'PASS' and heli_beats_flow:
        attribution = 'GEOMETRIC'      # braid beats flow AND beats its shuffle
    elif h_phase == 'NULL':
        attribution = 'ADDRESSING'     # curl is noise: digital fractal, not geometry
    else:
        attribution = 'INCONCLUSIVE'

    summary = {
        'H_V26_001_helicity_content': {'status': h001,
            'H_content': H['content_frac'], 'base_content': base['content_frac']},
        'H_V26_PHASE_braid_null': {'status': h_phase,
            'H_content': H['content_frac'],
            'H_shuf_content': results.get('H_shuffled',{}).get('content_frac')},
        'H_V26_TROUGH': trough,
        'heli_vs_flow': {'heli_beats_flow': bool(heli_beats_flow),
            'H_content': H['content_frac'], 'F_content': F['content_frac']},
        'ATTRIBUTION': {'verdict': attribution,
            'legend': 'GEOMETRIC=kappa links fractal; MAP-ONLY=history record; '
                      'ADDRESSING=digital fractal, geometry incidental'},
        'kappa': helicity_field.kappa if (helicity_field and helicity_field.built) else None,
    }
    print(f"\n  {'='*60}")
    print(f"  H-V26-001 (helicity->content):  {h001}", flush=True)
    print(f"  H-V26-PHASE (braid null):       {h_phase}", flush=True)
    print(f"  helicity beats plain flow:      {heli_beats_flow}", flush=True)
    if trough: print(f"  H-V26-TROUGH:                   {trough.get('status')} "
                     f"(conn={trough.get('connectivity_ratio')}, "
                     f"shuf={trough.get('shuffled_connectivity_ratio')})", flush=True)
    print(f"  ATTRIBUTION:                    {attribution}", flush=True)
    print(f"  {'='*60}", flush=True)

    with open(out_path, 'w') as f:
        json.dump({'arms': results, 'summary': summary, 'n_tokens': n_tokens},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  → {out_path}", flush=True)



# ══════════════════════════════════════════════════════════════════════════════
# BURST FISHER
# ══════════════════════════════════════════════════════════════════════════════

def run_fish(eng, stiff, hel, mass, tok, n_tokens, n_fish, out_path, prompt_ids=[]):
    print(f"\n{'='*60}", flush=True)
    print(f"  Burst Fisher v24 — {n_fish} episodes × {n_tokens} tokens", flush=True)
    print(f"{'='*60}", flush=True)
    all_max=[]; all_mean=[]; all_noise=[]; all_lock=[]; all_dj1=[]
    for ep in range(n_fish):
        res = run_v24(eng,stiff,hel,mass,tok,prompt_ids,n_tokens,verbose=False)
        all_max.append(res['max_burst']); all_mean.append(res['mean_burst'])
        all_noise.append(res['noise_frac']); all_lock.append(res['lock_frac'])
        if res['mean_burst_dj1'] is not None: all_dj1.append(res['mean_burst_dj1'])
        if (ep+1) % 5 == 0:
            print(f"  [{ep+1}/{n_fish}]  mean_max={np.mean(all_max):.1f}  "
                  f"dj1={np.mean(all_dj1):.1f}°  lock={np.mean(all_lock):.0%}", flush=True)
    print(f"\n  RESULTS:", flush=True)
    print(f"  max_burst: p50={np.percentile(all_max,50):.0f}  "
          f"p75={np.percentile(all_max,75):.0f}  "
          f"p90={np.percentile(all_max,90):.0f}  max={max(all_max)}", flush=True)
    print(f"  mean_dj1: {np.mean(all_dj1):.1f}°  (target <15°)", flush=True)
    print(f"  lock_frac: {np.mean(all_lock):.0%}  noise: {np.mean(all_noise):.1%}", flush=True)
    h004 = 'PASS' if max(all_max) >= 12 else 'NULL'
    h001 = 'PASS' if np.mean(all_dj1) < 15. else 'NULL'
    print(f"  H-V24-001 (dj1<15°): {h001}", flush=True)
    print(f"  H-V24-004 (max≥12):  {h004}  (max={max(all_max)})", flush=True)
    result = {
        'n_fish': n_fish, 'n_tokens': n_tokens,
        'burst_distribution': {
            'p50': round(float(np.percentile(all_max,50)),1),
            'p75': round(float(np.percentile(all_max,75)),1),
            'p90': round(float(np.percentile(all_max,90)),1),
            'max': int(max(all_max)),
        },
        'mean_dj1': round(float(np.mean(all_dj1)),2) if all_dj1 else None,
        'lock_frac_mean': round(float(np.mean(all_lock)),4),
        'noise_frac_mean': round(float(np.mean(all_noise)),4),
        'H_V24_001': {'status': h001},
        'H_V24_004': {'status': h004, 'max': int(max(all_max))},
    }
    with open(out_path,'w') as f: json.dump(result,f,indent=2)
    print(f"  → {out_path}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE NULL
# ══════════════════════════════════════════════════════════════════════════════

def run_phase_null(eng, stiff, hel, mass, tok, n_tokens, out_path, prompt_ids=[]):
    print(f"\n  Phase Null — H-V24-PHASE  (5 reps each)", flush=True)
    N = 5
    real_r=[]; shuf_r=[]
    for i in range(N):
        r = run_v24(eng,stiff,hel,mass,tok,prompt_ids,n_tokens,shuffled=False,verbose=False)
        s = run_v24(eng,stiff,hel,mass,tok,prompt_ids,n_tokens,shuffled=True, verbose=False)
        real_r.append(r); shuf_r.append(s)
        print(f"  rep {i+1}: real burst={r['mean_burst']:.1f} dj1={r['mean_burst_dj1'] or '—'} | "
              f"shuf burst={s['mean_burst']:.1f} dj1={s['mean_burst_dj1'] or '—'}", flush=True)
    rb = np.mean([r['mean_burst'] for r in real_r])
    sb = np.mean([s['mean_burst'] for s in shuf_r])
    rn = np.mean([r['noise_frac'] for r in real_r])
    sn = np.mean([s['noise_frac'] for s in shuf_r])
    h_phase = 'PASS' if rb > sb * 1.10 else 'NULL'
    print(f"\n  REAL: burst={rb:.2f}  noise={rn:.1%}")
    print(f"  SHUF: burst={sb:.2f}  noise={sn:.1%}")
    print(f"  H-V24-PHASE: {h_phase}")
    result = {
        'real': {'burst_mean': round(rb,3), 'noise_mean': round(rn,4)},
        'shuffled': {'burst_mean': round(sb,3), 'noise_mean': round(sn,4)},
        'H_V24_PHASE': {'status': h_phase},
    }
    with open(out_path,'w') as f: json.dump(result,f,indent=2)
    print(f"  → {out_path}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Orbital Generator v26 — ML-101')
    p.add_argument('command', choices=['generate','ablation','fish','phase_null','trough'])
    p.add_argument('--engram',        default='engram_curve.json')
    p.add_argument('--token_mass',    default='token_mass.json')
    p.add_argument('--stiffness_lut', default='vacuum_stiffness_lut.npy')
    p.add_argument('--helicity_lut',  default='helicity_lut.npy')
    p.add_argument('--model',         default=None)
    p.add_argument('--prompt',        default='The geometry of thought')
    p.add_argument('--n_tokens',      type=int, default=80)
    p.add_argument('--n_fish',        type=int, default=20)
    p.add_argument('--j1_init',       type=float, default=PHI_INIT)
    p.add_argument('--ksi_init',      type=float, default=KSI_INIT)
    p.add_argument('--shuffled',      action='store_true')
    p.add_argument('--mode',          default='cone', choices=['cone'])
    # v25 additions
    p.add_argument('--corpus',        default=None,
                   help='text file to build the flow field (KNOB 1/3). Needs --model.')
    p.add_argument('--use_flow',      action='store_true', help='enable KNOB 1 in generate')
    p.add_argument('--use_lambda',    action='store_true', help='enable KNOB 2 in generate')
    # v26 additions
    p.add_argument('--use_helicity',  action='store_true', help='enable KNOB 3 in generate')
    p.add_argument('--kappa',         type=float, default=ALPHA_PHYSICAL,
                   help=f'helical winding coupling (default alpha_physical={ALPHA_PHYSICAL}; '
                        f'kappa_crit={KAPPA_CRIT})')
    p.add_argument('--seed',          type=int, default=0)
    p.add_argument('--output',        default='v26_out.json')
    args = p.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"  Orbital Generator v26 — ML-101  [{args.command}]", flush=True)
    print(f"{'='*60}\n", flush=True)

    if not Path(args.engram).exists():
        print(f"[ERROR] engram not found: {args.engram}"); sys.exit(1)

    eng   = load_engram(args.engram)
    stiff = load_lut(args.stiffness_lut)
    hel   = load_lut(args.helicity_lut)
    mass  = load_mass(args.token_mass)
    print(f"  Engram: {eng['n']:,} tokens  has_spin={eng['has_spin']}", flush=True)

    tok = None; prompt_ids = []
    if args.model and HAS_TOK:
        try:
            tok = GPT2TokenizerFast.from_pretrained(args.model)
            prompt_ids = tok(args.prompt)['input_ids']
            print(f"  Tokenizer: {args.model}  prompt={len(prompt_ids)}t", flush=True)
        except Exception as ex:
            print(f"  [warn] {ex}", flush=True)

    # ── Build flow + helicity fields if a corpus is provided ──────────────────
    flow_field = None; flow_field_shuf = None
    helicity_field = None; helicity_field_shuf = None
    if args.corpus:
        if tok is None:
            print("  [warn] --corpus needs --model (tokenizer); flow/helicity disabled.", flush=True)
        else:
            random.seed(args.seed)
            print(f"  Building flow field from corpus: {args.corpus}", flush=True)
            flow_field      = build_flow_field(args.corpus, tok, eng, shuffled=False)
            flow_field_shuf = build_flow_field(args.corpus, tok, eng, shuffled=True)
            print(f"  Flow field coverage: {flow_field.coverage():.1%} of cells "
                  f"(>= {FLOW_MIN_COUNT} bigrams)", flush=True)
            # Helicity = curl of the flow field. Null = curl of the SHUFFLED flow
            # field (shuffle-then-curl): same marginals, broken braid.
            helicity_field      = build_helicity_field(flow_field, kappa=args.kappa)
            helicity_field_shuf = build_helicity_field(flow_field_shuf, kappa=args.kappa)
            print(f"  Helicity field built  kappa={args.kappa}  "
                  f"(curl of flow; null = curl of shuffled flow)", flush=True)

    if args.command == 'generate':
        res = run_v24(eng, stiff, hel, mass, tok, prompt_ids, args.n_tokens,
                      j1_init=args.j1_init, ksi_init=args.ksi_init,
                      shuffled=args.shuffled, mode=args.mode,
                      use_flow=args.use_flow, use_lambda=args.use_lambda,
                      flow_field=flow_field,
                      use_helicity=args.use_helicity, helicity_field=helicity_field,
                      rng=np.random.default_rng(args.seed))
        with open(args.output,'w') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\n  → {args.output}", flush=True)

    elif args.command == 'ablation':
        run_ablation(eng, stiff, hel, mass, tok, args.n_tokens, args.output, prompt_ids,
                     flow_field=flow_field, flow_field_shuf=flow_field_shuf, seed=args.seed,
                     helicity_field=helicity_field, helicity_field_shuf=helicity_field_shuf)

    elif args.command == 'trough':
        # Standalone wound-channel diagnostic (H-V26-TROUGH) + scrapes (H-SKIP).
        if helicity_field is None or not helicity_field.built:
            print("  [ERROR] trough needs --corpus and --model to build the helicity field.")
            sys.exit(1)
        report = helicity_field.trough_report(shuffled_helicity=helicity_field_shuf)
        print(f"\n  ── WOUND-CHANNEL DIAGNOSTIC (H-V26-TROUGH) ──", flush=True)
        for k, v in report.items():
            print(f"  {k}: {v}", flush=True)
        # Scrapes: skipping hypothesis (speed) vs density confound. kappa-independent.
        scrapes = helicity_field.scrapes_report(flow_field=flow_field,
                                                shuffled_helicity=helicity_field_shuf)
        print(f"\n  ── SCRAPES ANALYSIS (H-SKIP) ──", flush=True)
        print(f"  speed↔channel r={scrapes.get('r_speed_channel')} "
              f"(null95={scrapes.get('speed_null_95ci')})", flush=True)
        print(f"  mean speed: channel={scrapes.get('mean_speed_channel')} "
              f"speck={scrapes.get('mean_speed_speck')}", flush=True)
        print(f"  count↔channel r={scrapes.get('r_count_channel')} "
              f"(channel={scrapes.get('mean_count_channel')} speck={scrapes.get('mean_count_speck')})", flush=True)
        print(f"  size classes: {scrapes.get('size_classes')}", flush=True)
        print(f"  H-SKIP-001 (skipping): {scrapes.get('H_SKIP_001_skipping')}  "
              f"H-SKIP-002 (density): {scrapes.get('H_SKIP_002_density')}", flush=True)
        print(f"  VERDICT: {scrapes.get('VERDICT')}", flush=True)
        with open(args.output,'w') as f:
            json.dump({'trough': report, 'scrapes': scrapes, 'kappa': args.kappa}, f, indent=2)
        print(f"\n  → {args.output}", flush=True)

    elif args.command == 'fish':
        run_fish(eng, stiff, hel, mass, tok, args.n_tokens, args.n_fish, args.output, prompt_ids)

    elif args.command == 'phase_null':
        run_phase_null(eng, stiff, hel, mass, tok, args.n_tokens, args.output, prompt_ids)

    print(f"\n  ML-101 complete.\n", flush=True)

if __name__ == '__main__':
    main()
