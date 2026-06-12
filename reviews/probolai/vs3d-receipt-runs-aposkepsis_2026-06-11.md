# Probole: VS3D Receipt Run Review

**Target:** `trellmlx/vs3d.py` + `generate.py` VS3D wiring
**Branch:** `cc/vs3d-slat-edit-0611`
**Worktree:** `/private/tmp/trellis2mlx-vs3d-slat-edit-0611`
**Review context mode:** code + receipt run history in topos (no implementation thread)
**Topos:** `projects/trellis2mlx/topoi/cc-glyph-furnace-bloodhound-vs3d-0611.md`

## Scope

Review `trellmlx/vs3d.py` against arXiv:2605.07385 (VS3D). The implementation
has 10 passing unit tests but 5 receipt runs against the angel→angel+basketball
edit pair have all produced structural damage (holes, missing voxels, blobbing,
mutant hand/basketball junction). The unit tests pass stub models; real receipt
runs expose failure modes the stubs cannot catch.

## Required reads

1. `trellmlx/vs3d.py` — full file
2. `generate.py` — VS3D wiring (search for `vs3d_mode`, `vs3d_flow_sample`)
3. `tests/test_vs3d_editing.py` — understand what the stubs test and what they miss
4. `projects/trellis2mlx/topoi/cc-glyph-furnace-bloodhound-vs3d-0611.md` — receipt run history

## Key questions for the reviewer

1. **PMG formula correctness.** Is `v_pmg = (1+w)*mu_S - w*mu_L` the right formula, where each sample uses `v_cfg = v_pos + cfg_w*(v_pos - v_neg)`? The paper's Section 3.2 should clarify whether `w` is an additive amplification on top of CFG or a replacement for it.

2. **RASI gradient correctness.** The current finite-difference uses `diff_scalar = mx.mean(diff)` — a single scalar collapsed from the entire reconstruction error — as the perturbation direction for φ. This moves all φ elements uniformly. Is this a valid approximation of the paper's RASI gradient, or is it too coarse to produce meaningful source anchoring?

3. **Voxel loss source.** Stage 1 source pass produces 1250 sparse voxels. Every VS3D guided pass produces ~1083 regardless of RASI on/off or cfg_w value. The hypothesis is PMG velocities shift logit distribution at body tokens below the `logits > 0` decode threshold. Is there a cleaner fix (e.g. threshold adjustment, logit clamp) or does this point at a fundamental guidance magnitude problem?

4. **cfg_w_tgt=9.0 for Stage 1.** The paper may specify different guidance strengths for the sparse structure stage vs SLat stages. Does the paper state a Stage 1 guidance value? Is 9.0 appropriate or too high?

5. **Test coverage gap.** The unit tests use stub models that are cond-blind (or lightly cond-sensitive). They cannot catch the over-guidance and voxel-loss failure modes seen in real runs. What minimal test changes would catch these earlier?

## Non-goals

- Do not review the SLat stage wiring (Stage 2a/2c) — that experiment was tried and reverted; Stage 1 quality must be established first.
- Do not review TAR — it is not yet wired into the receipt pipeline.
- Do not review mesh simplification or texture pipeline.

## Expected output

Durable anaphora at `reviews/anaphorai/vs3d-receipt-runs-aposkepsis_2026-06-11.md` with:
- Verdict on each of the 5 key questions above
- Any additional material findings with specific file:line citations
- Recommended next implementation step with enough precision to queue the next receipt run
