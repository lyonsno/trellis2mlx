# VS3D SLAT Editing — trellis2mlx

- Diaulos: `glyph-furnace-bloodhound`
- Tool: Claude Code
- Parent/steward: `coherexivity-foundry-midwife`
- Epistaxis launch worktree: `/private/tmp/epistaxis-ideogram-fp8-mflux-bench-0610` on branch `cc/ideogram-fp8-mflux-bench-0610`
- Status: **Κίνησις.** Implementation landed; all 10 fail-first tests pass. Receipt runs in progress — structural quality not yet satisfactory.
- Session start: 2026-06-11
- Machine: MacBook-Pro-2.local / Apple M4 Max / 128 GB RAM / macOS 15.6 (24G84) / arm64
- Tool: Claude Code (claude-sonnet-4-6)

## Task

Implement VS3D (arXiv:2605.07385) training-free local 3D editing modules in the frozen trellis2mlx ODE loop — no new model weights, no masks, no inversion.

## Implementation worktree

- Repo: `/Users/noahlyons/dev/trellis2mlx`
- Worktree: `/private/tmp/trellis2mlx-vs3d-slat-edit-0611`
- Branch: `cc/vs3d-slat-edit-0611`

## Artifacts

### `trellmlx/vs3d.py` (new)
Three VS3D modules + sampler:
- `rasi_optimize(...)`: K-step φ optimization via finite-difference gradient
- `pmg_velocity(...)`: S Monte Carlo samples, partial mean of L smallest-norm, v_PMG = (1+w)*μ_S − w*μ_L; each sample uses CFG at `cfg_w` strength
- `tar_inject(...)`: per-token cosine agreement gate, residual blend
- `vs3d_flow_sample(...)`: wraps Euler loop; RASI+PMG per guided step at Stage 1 dense

### `tests/test_vs3d_editing.py` (new — fail-first)
10 tests, all passing.

## Receipt run history (angel.jpg → angel_with_basketball.jpg)

### Run 1 (job `ddd027374882`) — original implementation
- **Result:** Basketball appeared at correct position. Body holes (missing faces in torso/base), floating fragments at hand junction.
- **Root cause identified:** PMG CFG strength was `(1+w)*v_pos - w*v_neg` = effective 2.2 (w=1.2 is an amplification weight, not guidance strength). `cfg_w_tgt=9.0` was never applied inside PMG. Separately: RASI finite-difference is a scalar proxy, not per-token.

### Run 2 (job `40625c20b4ec`) — added VS3D to SLat stages
- **Result:** Worse. More distortion. Structural collapse.
- **Root cause:** SLat source anchor was generated from independent noise + cond_src, mismatched with the Stage 1 output coord set. RASI fought a wrong anchor. Reverted.

### Run 3 (job `434c02f64df4`) — PMG CFG fix applied (cfg_w_tgt wired in)
- **Result:** Same mutant as run 1. Voxel count: 1083 vs source 1250. Hand junction still fragmented.
- **Finding:** CFG fix alone did not resolve. 1083 voxels consistently across runs — 13% structural loss vs source pass.

### Run 4 (job `750188cc91bf`) — guidance interval narrowed to (0.0, 0.4)
- **Result:** Basketball grey/absent. Mesh weird. Worse.
- **Finding:** Basketball edit only lands at high-t structural steps (t > 0.6). Low-t only = edit signal lost.

### Run 5 (job `bf734ee7091b`) — RASI disabled (K=0), PMG only, interval (0.6, 1.0)
- **Result:** Blobby/molten. Different failure mode.
- **Finding:** Without RASI, voxel count still 1083. Blobbing = PMG at cfg=9.0 over-guidance even without RASI. RASI is not the voxel-eater; the loss is in PMG.

### Runs 6–9 — CANCELLED (cfg sweep superseded by architecture rewrite)
- All prior failures traced to missing FlowEdit dual-branch coupling (aposkepsis finding). Parameter sweeps of broken architecture were abandoned.

### Run 6 (job `dc4b26e6cd31`) — FlowEdit rewrite, paper hyperparams
- **Implementation:** Full rewrite of `vs3d.py`. z_edit starts at x_src. S eps samples form (z_t_src, z_t_tgt) pairs. PMG on v_delta = v_tgt_cfg - v_src_cfg. RASI uses c_src on both branches. CFG unified to additive form. Euler step: z_edit += dt * u.
- **Params:** omega_src=1.5, omega_tgt=9.0, S=5, L=2, w=1.2, K=3, guidance_interval=(0.6, 1.0), steps=25
- **Status:** Queued. 15 tests pass.

## Architecture state (post-rewrite)

Correct FlowEdit dual-branch coupling is now implemented. Root causes from aposkepsis all addressed:
- ✓ z_edit starts at x_src, not noise
- ✓ PMG on v_delta (not single-branch CFG velocities)
- ✓ S samples use different eps draws (real variance for partial mean)
- ✓ CFG unified: `(1+omega)*v_pos - omega*v_neg` everywhere
- ✓ RASI uses c_src on both branches
- ✓ Euler step: `z_edit += dt * u`

## Open questions (post-rewrite)

1. Voxel count: will dual-branch coupling prevent the 1250→1083 structural loss? Hypothesis: yes, because z_edit starts at x_src and v_delta is a differential — body token logits should stay near source.
2. RASI FD gradient still limited in dimensionality — but the objective is now correct. If voxel loss is resolved, assess whether K=3 RASI adds value or can stay K=0.
3. Steps=25 vs prior steps=12 — paper uses T=25 total steps; check whether worktree default still says 12 and if generate.py needs updating.

## SLAT scout pointer

Full landscape scout: `metadosis/machieavelli-the-mlx-don_2026-06-07.reports/slat-editing-landscape-scout_2026-06-11.md`
