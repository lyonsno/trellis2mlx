# Anaphora: VS3D Receipt Run Review

**Probole:** `reviews/probolai/vs3d-receipt-runs-aposkepsis_2026-06-11.md`
**Reviewer:** Opus 4.6 (Claude Code epistaxis-aposkepsis agent)
**Review context mode:** code + receipt run history in topos (no implementation thread)
**Date:** 2026-06-11
**Target:** `trellmlx/vs3d.py` + `generate.py` VS3D wiring, branch `cc/vs3d-slat-edit-0611`

## Commands run

- Read `trellmlx/vs3d.py` (434 lines, full)
- Read `tests/test_vs3d_editing.py` (400 lines, full)
- Read `generate.py` VS3D wiring (lines 290-740, grep + targeted read)
- Read `trellmlx/samplers.py` `flow_euler_sample` (lines 13-112)
- Read topos `cc-glyph-furnace-bloodhound-vs3d-0611.md` (full)
- Fetched arXiv:2605.07385 HTML: abstract, Sections 2-3.4, Algorithm 1, Appendix A.1

No tests were executed (review-only agent).

---

## Finding 1 [CRITICAL]: The implementation is not VS3D -- it is missing the dual-branch FlowEdit coupling entirely

**Severity:** Critical architectural mismatch. This is the root cause of all five receipt-run failures.

**Paper's architecture (Algorithm 1, Eq. 3-5, 7-9):**

VS3D is built on **FlowEdit coupling**: at each timestep, S independent noise draws produce S source/target latent pairs via:

    z_t^src = (1-t) * x_src + t * eps^(s)
    z_t^tgt = z_t^edit + (z_t^src - x_src)

For each noise sample s, the paper computes:

    v_tgt^(s) = CFG(z_t^tgt, c_tgt, phi; omega_tgt)   [4 model calls per sample:
    v_src^(s) = CFG(z_t^src, c_src, phi; omega_src)     2 for tgt branch, 2 for src branch]
    v_Delta^(s) = v_tgt^(s) - v_src^(s)

PMG then operates on the **velocity differences** v_Delta:

    mu_S = mean of S v_Delta samples
    mu_L = partial mean of L v_Delta samples (smallest norm)
    u = (1+w)*mu_S - w*mu_L

The Euler step updates: `z_edit += dt * u`

**Implementation's architecture (vs3d.py):**

The implementation runs a **single-branch target-only sampler**. There is no source branch, no FlowEdit coupling, no noise-coupled latent pairs, and no v_Delta. Instead:

- `vs3d_flow_sample` (line 365): starts from noise, not from x_src
- `pmg_velocity` (lines 164-171): calls the model S times with the **same** z_t, same cond_tgt, same phi. It computes S copies of the single-branch CFG velocity `v_pos + cfg_w*(v_pos - v_neg)` and takes their mean and partial mean. With a deterministic frozen model, all S samples are identical -- PMG has no variance to exploit and degenerates to plain CFG.
- There is no `z_t^src` formation, no `v_src` computation, no `v_Delta = v_tgt - v_src` computation anywhere in the code.

**What this means:**

The entire paper's mechanism depends on the velocity *difference* between two coupled branches. The source branch with RASI-optimized phi is what suppresses identity leakage on non-edited voxels. Without it, the system is just a regular target-conditioned sampler with extra CFG strength -- which is exactly why every receipt run produces structural damage: there is no identity-preservation mechanism operating.

**File references:**
- `vs3d.py:164-171` -- PMG computes CFG on single z_t, not v_Delta
- `vs3d.py:365` -- sampler starts from noise, not x_src
- `vs3d.py:389-404` -- RASI runs on single z_t with cond_src, no source branch
- Paper Algorithm 1, lines 15-24: Phase 2 starts with `z^edit <- x_src`, loops over S noise draws forming coupled pairs

---

## Finding 2 [CRITICAL]: CFG formula inconsistency between PMG and RASI/fallback

**Severity:** Critical. Two different CFG conventions coexist in the same file.

**Paper's CFG formula (Eq. 4):**

    v_cfg = (1 + omega) * v_theta(z, t, c) - omega * v_theta(z, t, phi)

This is the additive form.

**Implementation:**

- `pmg_velocity` line 170: `v_cfg = v_pos + cfg_w * (v_pos - v_neg)` = `(1 + cfg_w)*v_pos - cfg_w*v_neg`
  - This matches Eq. 4. With cfg_w=9.0: effective = 10*v_pos - 9*v_neg
- `rasi_optimize` line 78: `v_src = cfg_w_src * v_pos + (1.0 - cfg_w_src) * v_neg`
  - This is the linear interpolation form: `w*v_pos + (1-w)*v_neg`. With cfg_w_src=1.5: effective = 1.5*v_pos - 0.5*v_neg
- `vs3d_flow_sample` fallback line 424: `pred = cfg_w_tgt * pred_pos + (1.0 - cfg_w_tgt) * pred_neg`
  - Same linear interpolation form

With the paper's Eq. 4 convention, `omega_src=1.5` means `2.5*v_pos - 1.5*v_neg`. The implementation's RASI uses `1.5*v_pos - 0.5*v_neg`. This is a different guidance strength.

The baseline `flow_euler_sample` (samplers.py:91) also uses the linear interpolation form: `guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg`. So the source pass generating x_src uses `7.5*v_pos - 6.5*v_neg` (linear form with default guidance_strength=7.5).

**File references:**
- `vs3d.py:78` -- RASI CFG (linear interpolation form)
- `vs3d.py:170` -- PMG CFG (additive form, matches paper Eq. 4)
- `vs3d.py:424` -- fallback CFG (linear interpolation form)
- `samplers.py:91` -- baseline CFG (linear interpolation form)

---

## Finding 3 [HIGH]: Source pass uses default guidance, no RASI x_src anchor

**Severity:** High. The x_src anchor latent is generated with wrong configuration.

**Paper:** RASI Phase 1 (Algorithm 1, lines 5-13) starts from `z^edit <- x_src` and runs a full RASI calibration loop to produce per-timestep phi_t embeddings. x_src is the source-encoded latent, not a generated sample.

**Implementation:** `generate.py:491-494` generates x_src by running the regular `flow_euler_sample` with `cond_src` and default `guidance_strength=7.5`, `guidance_rescale=0.7`. This is a full generation from noise -- not encoding the source. More importantly:

- x_src is used as the RASI reconstruction target, but it was generated with guidance_strength=7.5 in the linear interpolation convention, while the RASI probe uses cfg_w_src=1.5 -- a completely different effective guidance.
- The source pass uses `guidance_rescale=0.7` (variance-normalizing rescale from samplers.py:93-104), but `vs3d_flow_sample` never applies any rescale to PMG output.

**File references:**
- `generate.py:491-494` -- x_src generated with default params
- `generate.py:510-511` -- vs3d pass uses cfg_w_src/cfg_w_tgt from CLI
- `vs3d.py:303,369` -- guidance_rescale accepted but never applied in dense loop

---

## Finding 4 [HIGH]: PMG sampling produces identical samples with deterministic model

**Severity:** High. The S Monte Carlo samples in PMG serve no purpose with the current architecture.

**Paper:** The S noise samples in PMG provide variance through the FlowEdit coupling (Eq. 3) -- each eps^(s) produces a different z_t^src and z_t^tgt, giving S different v_Delta estimates. The partial mean then selects the L samples with smallest v_Delta norm (most conservative) and extrapolates away from them.

**Implementation:** `pmg_velocity` (lines 164-171) calls the model S times with identical inputs: same z_t, same t_tensor, same cond_tgt, same phi. The comment at lines 158-163 acknowledges this: "For a fully deterministic frozen model these will be identical; PMG still applies -- the variance collapses and (1+w)*mu_S - w*mu_L = mu_S." This is correct math: with no variance, PMG degenerates to plain CFG. The S=5 Monte Carlo budget (10 extra model calls per step) produces zero additional signal.

**File references:**
- `vs3d.py:158-163` -- comment acknowledging deterministic collapse
- `vs3d.py:164-171` -- identical model calls

---

## Finding 5 [MEDIUM]: RASI objective function does not match paper

**Severity:** Medium (subordinate to Finding 1, but independently wrong).

**Paper's RASI objective (Eq. 7):**

    minimize || z_t^edit + dt * (v_tgt(z_t^tgt, c_src, phi; omega_tgt) - v_src(z_t^src, c_src, phi; omega_src)) - x_src ||^2

Key points:
1. Both branches use **c_src** (not c_tgt) -- this probes identity leakage specifically
2. Both branches retain their respective guidance weights omega_tgt, omega_src
3. The objective operates on the **coupled velocity difference**, not a single-branch prediction
4. Optimization uses **autograd** through the model (Algorithm 1 line 10: Adam with gradient of L_rec)

**Implementation's RASI (lines 72-110):**
1. Uses single-branch source velocity: `v_src = cfg_w_src * v_pos + (1-cfg_w_src) * v_neg` with only cond_src
2. No target branch evaluation at all
3. Uses **finite-difference gradient** with a deterministic probe direction instead of autograd
4. The finite-difference collapses to a scalar: `fd_grad = (loss_plus - loss_minus) / (2*eps)`, which is a single scalar multiplied by the probe direction (line 107: `phi = phi - lr * fd_grad * perturb_dir`). This moves phi along one predetermined direction per step -- not along the true gradient.

The finite-difference approach is an understandable adaptation for MLX (which may lack convenient autograd through arbitrary model calls), but it is a fundamentally different optimization that converges far more slowly and in the wrong direction when the true gradient has components orthogonal to the probe.

**File references:**
- `vs3d.py:72-110` -- RASI implementation
- `vs3d.py:94-107` -- finite-difference gradient computation
- Paper Eq. 7 + Algorithm 1 lines 9-10

---

## Answers to Key Questions

### Q1: PMG formula correctness

The PMG formula `v_pmg = (1+w)*mu_S - w*mu_L` at vs3d.py:202-203 is **correct in form** (matches paper Eq. 8). However, it is applied to the **wrong quantity**. The paper applies this formula to `v_Delta` (the velocity difference between coupled source and target branches). The implementation applies it to single-branch CFG velocities from the target-only sampler. This is the core architectural mismatch.

The per-sample CFG formula at line 170 (`v_pos + cfg_w*(v_pos - v_neg)`) matches the paper's Eq. 4 additive convention. But `w` in the PMG formula is NOT a CFG weight -- it is a subsample extrapolation weight (w=1.2 amplifies the gap between mu_S and mu_L). The per-sample CFG at `cfg_w=9.0` is applied first, then PMG extrapolates the v_Delta means. The implementation conflates these two levels.

### Q2: RASI gradient correctness

The finite-difference gradient is **not a valid approximation** of the paper's RASI gradient for two reasons:

1. The paper uses autograd (Adam optimizer with true gradient of the reconstruction loss w.r.t. phi). The implementation uses a 1D directional finite difference with a fixed deterministic probe direction.
2. The fd_grad scalar multiplied by perturb_dir gives movement along only ONE direction per step. With K=3 steps, phi moves along at most 3 directions (the rolled +/-1 pattern). The true gradient in phi-space (shape [1, 10, 1024] = 10,240 dimensions) has 10,240 independent components. 3 probe directions cannot approximate this.

However, this is subordinate to Finding 1: even perfect RASI would not fix the missing dual-branch architecture.

### Q3: Voxel loss source

The voxel loss (1250 -> 1083, 13% structural loss) is **not caused by RASI or PMG specifically** -- it is caused by the fundamental architectural mismatch. The implementation runs a target-conditioned single-branch sampler with extremely high effective CFG (10x with additive formula). Without the source branch to anchor identity, the velocity field pulls ALL voxels toward the target appearance, shifting the dense logit distribution. Tokens near the occupancy boundary fall below the `logits > 0` threshold at generate.py:526.

Lowering cfg_w_tgt would reduce voxel loss but also weaken the edit signal, because without the FlowEdit coupling there is no mechanism to decouple edit amplification from identity preservation -- that decoupling IS the paper's contribution.

### Q4: cfg_w_tgt=9.0 for Stage 1

The paper's Appendix A.1 states: `omega_src=1.5` and `omega_tgt=9.0` for Stage 1, with the additive CFG convention (Eq. 4). So **9.0 is the correct paper value** for omega_tgt. However, in the paper, omega_tgt=9.0 applies to the target branch of the FlowEdit coupling -- it does NOT mean running a single target-only sampler at CFG 9.0. The dual-branch setup means the effective edit signal is the *difference* between v_tgt (at omega_tgt=9.0) and v_src (at omega_src=1.5), which is much more controlled than raw CFG 9.0 on a single branch.

The paper also uses T=25 timesteps (not 12), and the active window is governed by n_max=12 and n_min=0, meaning the first 12 of 25 steps are active. The implementation uses 12 total steps with guidance_interval (0.6, 1.0).

### Q5: Test coverage gap

The unit tests verify:
- Shape contracts (correct, useful)
- phi differs from neg_cond after RASI (passes because fd does move phi, even if in the wrong direction)
- PMG produces nonzero output (trivially true with any nonzero model)
- PMG makes exactly 2*S model calls (correct count verification)
- TAR gates and preserves correctly (correct, useful)
- End-to-end shape and differs-from-baseline (passes trivially)

**What they cannot catch:**
1. The dual-branch coupling is entirely absent -- no test checks for v_Delta computation
2. The PMG variance exploitation is untestable with deterministic stubs -- no test verifies that S samples actually differ
3. No test compares voxel count before/after VS3D guidance
4. No test verifies that identity is preserved on non-edited tokens (the core VS3D property)

**Minimal changes to catch earlier:**
- A test that verifies PMG receives S different v_Delta samples (not S identical single-branch samples)
- A test that feeds a source latent through vs3d_flow_sample and verifies the output is closer to x_src on non-edited tokens than a plain target-only generation would be
- A voxel-count regression test: run vs3d_flow_sample with a stub model and verify decoded voxel count is within X% of source count

---

## Additional Findings

### Finding 6 [LOW]: Source pass noise seed shared with editing pass

`generate.py:489,499` both call `mx.random.seed(args.seed)` before generating noise. The source pass and the editing pass use the same seed, so `src_noise` and `noise` will be identical. In the paper's FlowEdit coupling, the source and target share the SAME noise by construction (Eq. 3), so using the same seed is actually correct -- but the implementation then starts from this noise rather than from x_src, which is wrong (see Finding 1).

### Finding 7 [LOW]: KV cache optimization missing from VS3D sampler

The baseline `flow_euler_sample` builds cross-attention KV caches (samplers.py:54-59) to avoid recomputing image conditioning projections at every step. The VS3D sampler in `vs3d_flow_sample` does not use KV caches. With S=5 PMG samples * 2 CFG calls + RASI calls per step, this is a significant performance regression (potentially 10-20x more redundant KV computation per step).

---

## Recommended Next Implementation Step

The next receipt run should NOT be a parameter sweep (cfg 3.0/5.0). The problem is architectural, not parametric.

**Recommended: Implement the FlowEdit dual-branch coupling.**

Concrete changes:

1. **vs3d_flow_sample (Stage 1 dense loop):** Start from `z_edit = x_src` (not noise). At each active timestep and for each of S noise samples: form `z_t_src = (1-t)*x_src + t*eps_s` and `z_t_tgt = z_edit + (z_t_src - x_src)`. Compute `v_tgt = CFG(z_t_tgt, c_tgt, phi; omega_tgt)` and `v_src = CFG(z_t_src, c_src, phi; omega_src)`. Set `v_delta_s = v_tgt - v_src`.

2. **RASI:** For each active step, form coupled pairs with c_src on BOTH branches (not c_tgt), compute `v_delta_probe = CFG(z_t_tgt, c_src, phi; omega_tgt) - CFG(z_t_src, c_src, phi; omega_src)`, and minimize `||z_edit + dt*v_delta_probe - x_src||^2` w.r.t. phi. The finite-difference approach can stay as a first pass, but the gradient should be computed per-element or at least per-token, not as a single scalar.

3. **PMG:** Operate on the S v_delta samples. Each sample naturally differs because each eps_s produces different z_t_src, z_t_tgt. Select the L samples with smallest v_delta norm per token, take their partial mean, and extrapolate: `u = (1+w)*mu_S - w*mu_L`. Step: `z_edit = z_edit + dt * u`.

4. **CFG convention:** Unify on the paper's additive form (Eq. 4) everywhere: `(1+omega)*v_pos - omega*v_neg`. Fix RASI line 78 and fallback line 424 to match.

5. **Timesteps:** Consider T=25 steps with active window n_max=12 to match the paper.

This is a substantial rewrite of `vs3d_flow_sample` and modifications to `rasi_optimize` and `pmg_velocity`, but the individual module APIs can remain similar. The model call count per step goes from 2*S (current PMG) to 4*S (paper: 2 branches * 2 CFG calls per branch per sample), plus RASI inner-loop calls. With S=5 and K=3, expect ~32 model calls per active step vs the current ~10.

The first receipt run after this change should use the paper's hyperparameters: T=25, omega_src=1.5, omega_tgt=9.0, S=5, L=2, w=1.2, K=3, lr=1e-5, guidance interval [0.6, 1.0]. RASI can remain finite-difference for the first test, with autograd as a follow-up if the dual-branch coupling alone does not resolve structural quality.
