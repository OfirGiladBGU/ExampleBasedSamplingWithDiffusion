# Rebuttal work: context for the next agent

Last updated: 2026-11-15. Keep this file current: update it whenever you submit jobs, get
results, or change a script.

## Goal and ground rules

- This is the StippleDiffusion paper (EG 2027, paper1182). We are preparing for the rebuttal:
  reviews are released Nov 23, 2026 and the rebuttal is due Nov 30. The plan, with experiments
  E0–E12, is in [REBUTTAL_PLAN.md](REBUTTAL_PLAN.md).
- **Do not edit paper text yet.** The user wants experiments fixed and run first. The
  caption/abstract parts of E0 are on hold.
- **Point budgets are shown visually only (user decision, 2026-11-01).** The models were trained
  at one budget (1024 or 1600 points) and never on others, so do not evaluate other budgets
  metrically, and do not put budget metrics in the paper or the rebuttal. Budgets appear only in
  Fig. 4 (the point-count generalization image, regenerated with the exact OT assignment) and in
  the runtime graph, which is to be updated with the new variant. E4 (quantitative budget
  generalization) is an **archived idea**: its results are in `experiments/outputs/a_archive/E4_budget_generalization/` and its scripts in
  `experiments/archive/E4_budget_generalization_scripts/`. Do not use them.
- The main code is in `control_v4/` (model, training, sampling) and `experiments/` (evaluation
  pipelines named `<name>_stage_N.py`, with outputs in `experiments/outputs/`).

## Where things run

- Server: `ofirgila@slurm.bgu.ac.il` (needs VPN; SSH key login works). The repo is at
  `/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion`, branch `fm_and_clip`.
  The server holds all checkpoints and results. The local Windows checkout has code only.
- **Git: rebuttal work lives on branch `rebuttal`** (created 2026-11-16 from `fm_and_clip`, first
  commit bcdea5f, not pushed). The server checkout is on `rebuttal`. Do not switch it back to
  `fm_and_clip` while jobs run: that would remove the rebuttal-only files (ot_assignment.py,
  fast_smart_init.py, the experiments/ scripts). Commit new rebuttal scripts and notes there; results
  in experiments/outputs and archive*/ folders are git-ignored. Unrelated uncommitted files (older
  control_v4_mix / pilot work, config toggles in sample.py, sample_control.py,
  quantitative_advance_metrics_*, .gitignore) were deliberately left out.
- Workflow: edit locally, `scp` the files to the server, then `sed -i "s/\r$//"` on them
  (to strip Windows line endings).
- Conda env: **`qmcdiffusion`** (Python 3.11, torch 2.1.1). The `control` env is old and
  must not be used. Jobs start with `module load anaconda; source activate qmcdiffusion`.
- **Submit every job with `sbatch`, smoke tests included.** Never block the session with
  `srun`. Log to `logs/`, and wait with a background loop that polls `squeue`.
- GPU jobs use `--gpus=rtx_6000:1`. That request has landed on both an RTX 6000 Ada node
  (`ise-cpu256-24`) and an L40S node (`ee-l40s-01`), so absolute times can differ between
  jobs. Stage 1 now records the GPU on every timing row.

## Facts established so far

- Paper checkpoints (Icons-50, 1024 points, grid 32):
  - Ours-GBN: `control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt`
  - Ours-BNOT: `control_v4/train_outputs_Icons-50_1024_BNOT/checkpoints/dynamic_ep5000.ckpt`
  - Ours-WVS: `control_v4/train_outputs_Icons-50_1024_WVS/checkpoints/dynamic_ep5000.ckpt`

  Inference uses `experiments/control_data_gen.py`, which calls
  `run_inference_on_directory` in `control_v4/sample_control.py`. Settings: SDEdit start at
  t = 500 of T = 1000, 500 DDPM steps, Smart Init seed 42. The start noise and step noise
  were not seeded.
- Diffusion details: the model predicts eps; clipping is forced off; the DDPM step uses the
  fixed "large" variance (β); the schedule is linear β from 1e-4 to 1e-2.
- Table 1 per-image metrics are in
  `experiments/outputs/quantitative_advance_metrics-<Dataset>_<N>/target_<METHOD>_<N>_json/`.
  Each has 1000 images and 6 methods. The test images are in
  `experiments/outputs/z_validation_data/<Dataset>/source` (same list as `validation_manifest.json`).
- **Table 2 (ablation)** used the **first 50 images** of the Icons-50 validation manifest.
  That manifest's order comes from the seed-42 random split in `train_control.py`, so in
  effect the subset is random with seed 42 (`NUM_SAMPLES = 50` in
  `ablation_advance_metrics_stage_1.py`). On those same 50 images, Table 1's Ours-GBN has
  capacity 0.0268, against Table 2's SDEdit 0.0255 and 0.0181 over all 1000 images. So the
  mismatch comes from the subset.
- **Metric code drift:** re-running `compute_all_advanced_metrics` today gives slightly
  different numbers from the stored Table 1 JSONs (around the 3rd significant digit). The key
  names also differ: the stored files say `M1_v2_power_cvt_energy` and
  `M2_v2_power_displacement`, while the current code says `M1_v2_hot_energy` and
  `M2_v2_centroid_displacement`. **Never mix freshly computed metrics with the stored Table 1
  numbers.** Re-score the baseline with the same code. This drift is probably the
  third-decimal mismatch noted in E0.
- Costs: about 20 ms per model evaluation at batch 1, and batch 8 costs the same wall time
  (the GPU is limited by per-step launch overhead). DDPM-500 takes about 10 s per image on the
  RTX 6000 Ada. Smart Init takes about 0.25 s on the CPU and the point conversion about 1 ms.
  Scoring takes about 10–33 s per image on the CPU, which is why metrics run as sharded CPU
  job arrays.

## E1: few-step sampling (in progress)

Scripts:
- [few_step_sampling_stage_1.py](few_step_sampling_stage_1.py): GPU sampling. It samples one
  checkpoint with several samplers from the same Smart Init and the same noise tensor (CPU
  generator seeded with `noise_seed + image index`). Sampler names:
  - `ddpm-K`: the paper DDPM loop started at step K
  - `ddim-K` / `ddim-K-eta1`: deterministic / stochastic DDIM with K model evaluations
  - `dpmpp2m-K`: DPM-Solver++(2M) with K model evaluations
  - an optional `-clean` suffix: start from the Smart Init without forward noise

  Few-step samplers start at `--t_start` (500), and their K timesteps are
  `round(linspace(t_start-1, 0, K))`. Images whose `.npy` already exists are skipped
  (resumable). `--tag` keeps the log files of concurrent jobs apart.
- [few_step_sampling_stage_2.py](few_step_sampling_stage_2.py): metrics, using the same code
  as `quantitative_advance_metrics_stage_1.py`, sharded with `--shard/--num-shards` and
  `--workers`. Existing JSONs are skipped.
- [few_step_sampling_stage_3.py](few_step_sampling_stage_3.py): writes `summary.md` and
  `summary.json`. For each sampler: mean, % change vs `ddpm-500`, paired Wilcoxon p-value,
  time per image, and stipples/s per batch size.
- Job scripts:
  - [few_step_sampling_gpu.sbatch](few_step_sampling_gpu.sbatch): array 0 = Ours-GBN,
    1 = Ours-BNOT
  - [few_step_sampling_metrics.sbatch](few_step_sampling_metrics.sbatch): `RUN=<label>_1024`,
    16 shards
  - [few_step_sampling_start_sweep.sbatch](few_step_sampling_start_sweep.sbatch): DDPM
    start-step sweep, Ours-GBN only
- Output: `experiments/outputs/few_step_sampling-Icons-50_1024/{CN-GBN_1024,CN-BNOT_1024}/<sampler>/`.
  Smoke tests are in `experiments/outputs/z_smoke_few_step/`.

Jobs submitted on 2026-10-26 (later jobs depend on earlier ones):

| Job | What |
|---|---|
| 23619165 | Sampling: ddpm-500, ddim-100/50/25/10, dpmpp2m-20/10, plus throughput at batch 1/8/32/64. Task 0 = GBN (RTX 6000 Ada), task 1 = BNOT (L40S) |
| 23619244 / 23619245 | Metrics for GBN / BNOT, after 23619165 |
| 23619246 | Sampling: ddim-100/50/25/10-eta1 for both models, after 23619165 |
| 23619247 / 23619248 | Metrics for the eta1 runs, after 23619246 |
| 23619254 | `summary.md`, after the eta1 metrics (an early summary, superseded by 23619686) |
| 23619684 | Start-step sweep for GBN (3 tasks): ddpm-{500,300,200,100,50}-clean and ddpm-{300,200,100,50}, tags `start0`–`start2` |
| 23619685 | Metrics for the GBN sweep, after 23619684 and 23619247 |
| 23619686 | Final `summary.md`, after 23619685 and 23619248. **FAILED**: `JSONDecodeError` on one corrupted metrics file (see open question 4) |
| 23650999 | 2026-10-28: re-score of that one image (`CN-GBN_1024/ddpm-300`, `google_3_raised_hand`), with the race fixed |
| 23651000 | 2026-10-28: **final `summary.md` / `summary.json`**, after 23650999. COMPLETED |
| 23651337 / 23651338 | 2026-10-28: **variance check** (open question 1): `ddim-500-eta1` / `ddim-250-eta1` on Ours-GBN, tags `var500` / `var250`, no throughput |
| 23651339 | Metrics for the variance check (`RUN=CN-GBN_1024`, 16 shards), after 23651337 and 23651338 |
| 23651340 | `summary.md` again, after 23651339. **Read the ddim-500-eta1 and ddim-250-eta1 rows next** |

The stage-1 refactor that added `ddpm-K` / `-clean` was verified: `ddpm-500` gives bit-identical
points before and after it (smoke job 23619661).

3-image smoke result (CN-GBN, not significant):
- DDIM with η = 0 is 20–50× faster but about 7–14% worse on HOT, capacity and centroid.
  CVT, Sinkhorn and ρs are unchanged.
- That loss is about the same at 100 steps as at 10, so the likely cause is the missing
  per-step noise, not the step count. This is why the eta1 runs were added.
- DPM++ is worse: +29% capacity at 10 steps. This matches the earlier note in
  `control_v4/README.md`.

The user's own idea, being tested in the sweep: start from the Smart Init with no forward
noise and run only the last DDPM steps. Each start step is run both clean and noised, so the
effect of skipping the noise can be told apart from the effect of a shorter run.

### E1 final summary (`summary.md`, 2026-10-28)

`experiments/outputs/few_step_sampling-Icons-50_1024/summary.md` (and `summary.json`) now exist, with all
samplers on 1000 paired images (dpmpp2m included) and paired Wilcoxon p-values. They confirm the
hand-read table below; the % changes agree to within 0.1 percentage point.

- **ddim-10-eta1, Ours-GBN:** every change is an improvement, and all are significant (p < 0.05)
  except EMD (−0.1%) and Sinkhorn (0.0%), which are unchanged.
- **ddim-10-eta1, Ours-BNOT:** 7 of the 8 metrics (the 7 paper metrics plus cell CV) improve
  significantly. Sinkhorn is +0.010% (p = 2e-5): statistically significant, but negligible.
- The only significantly worse cells for any η=1 sampler are Ours-BNOT's Sinkhorn at 10 and 25
  steps (+0.010% and +0.001%). So the rebuttal should say "match or improve", not "improve".
- Measured Smart Init overhead: 0.177 s (GBN job) and 0.132 s (BNOT job) per image, below the earlier
  0.25 s estimate. So ddim-10-eta1 takes about 0.28 s per image in total, Smart Init included.
- The GBN section's header lists two GPUs (L40S and RTX 6000 Ada): its samplers came from
  different jobs. Timing comparisons should still use the throughput rows (open question 3).

### E1 results on 1000 images (read 2026-10-26, before `summary.md`)

Computed by a read-only scan of the per-image JSONs, not by stage 3, so there are no Wilcoxon
p-values yet. Each cell is the % change of the mean against `ddpm-500` on the images both have,
and in brackets the share of images where the sampler is better. CVT and HOT are ×1e5.
`time` is the median `time_denoise` per image at batch 1, without Smart Init.

Ours-GBN:

| sampler | n | time (s) | CVT | HOT | cap δc | centroid | EMD | Sinkhorn | ρs |
|---|---|---|---|---|---|---|---|---|---|
| ddpm-500 (paper) | 1000 | 5.42 | 2.4558 | 4.0817 | 0.0178 | 0.0059 | 0.0132 | 0.0925 | 0.4941 |
| **ddim-10-eta1** | 1000 | **0.11** | −0.4% (75) | −0.4% (55) | **−5.1% (74)** | −1.1% (57) | −0.1% (53) | 0.0% (49) | **+1.5% (96)** |
| ddim-25-eta1 | 1000 | 0.27 | −0.5% (81) | −0.6% (55) | −8.4% (83) | −1.2% (56) | −0.1% (54) | 0.0% (52) | +1.5% (98) |
| ddim-50-eta1 | 1000 | 0.54 | −0.3% (74) | −0.5% (54) | −6.4% (80) | −1.2% (56) | 0.0% (53) | 0.0% (52) | +1.1% (96) |
| ddim-100-eta1 | 1000 | 1.07 | −0.2% (68) | −0.2% (53) | −4.4% (74) | −0.8% (55) | +0.1% (52) | 0.0% (50) | +0.8% (89) |
| ddim-10 (η=0) | 1000 | 0.11 | −0.1% | +13.1% | +6.4% | +19.0% | +6.1% | +0.1% | +1.1% |
| ddim-25 (η=0) | 1000 | 0.26 | 0.0% | +14.4% | +7.8% | +21.0% | +6.6% | +0.2% | +0.9% |
| dpmpp2m-10 | 946 | 0.11 | +1.2% | +17.3% | +23.2% | +25.8% | +8.3% | +0.2% | −1.4% |
| dpmpp2m-20 | 866 | 0.21 | +0.5% | +16.7% | +17.9% | +24.8% | +7.8% | +0.2% | 0.0% |

Ours-BNOT (ddpm-500: CVT 2.4217, HOT 2.8550, δc 0.0194, centroid 0.0032, EMD 0.0110,
Sinkhorn 0.0922, ρs 0.5085):

| sampler | time (s) | CVT | HOT | cap δc | centroid | EMD | ρs |
|---|---|---|---|---|---|---|---|
| **ddim-10-eta1** | 0.12 | −0.6% (87) | −0.1% (61) | −4.6% (77) | −3.3% (66) | −0.1% (61) | +1.8% (99) |
| ddim-25-eta1 | 0.30 | −0.6% (90) | −0.7% (66) | −8.7% (88) | −4.7% (70) | −0.4% (66) | +1.6% (100) |
| ddim-10 (η=0) | 0.06 | 0.0% | +12.5% | +9.1% | +36.8% | +6.8% | +1.2% |
| dpmpp2m-10 | 0.06 | +0.9% | +15.8% | +13.0% | +47.4% | +8.5% | −0.6% |

Start-step sweep (Ours-GBN, % change vs ddpm-500):

| sampler | HOT | cap δc | ρs |
|---|---|---|---|
| ddpm-300 / 200 / 100 / 50 | +19% / +35% / +68% / +111% | +5% / +12% / +43% / +146% | about 0, falling to −4% |
| ddpm-300-clean / 200-clean / 100-clean / 50-clean | +204% / +90% / +78% / +112% | +1095% / +217% / +56% / +138% | −3% to −1% |
| ddpm-500-clean (992 images) | +950% | +6745% | −9% |

Throughput (stipples/s, median, same job for every sampler, so comparable):

| sampler | batch 1 | batch 8 | batch 32 | batch 64 |
|---|---|---|---|---|
| ddpm-500 | 0.19 | 1.35 | 3.35 | 3.29 |
| ddim-25 | 3.81 | 27.8 | 69.1 | 66.0 |
| ddim-10 | 9.61 | 69.1 | 172.4 | 164.9 |

(The throughput file has the η=0 samplers only. η=1 makes the same number of model calls, so
it should be the same, but it was not measured.)

Conclusions:
- **Stochastic DDIM (η=1) with 10 steps is about 50× faster than DDPM-500 and equal or better
  on all 7 metrics, for both checkpoints.** Capacity improves by 5% (74–77% of images), ρs by
  1.5–1.8% (96–99% of images), and nothing gets meaningfully worse (the largest: Ours-BNOT
  Sinkhorn +0.01%, see the final summary). At 25 steps the gains are a bit larger
  (δc −8.4% / −8.7%).
- Rebuttal line (draft): "With stochastic DDIM (η=1), 10 denoising steps match or improve all
  7 metrics against our 500-step sampler, cutting denoising from 5.4 s to 0.11 s per stipple;
  batched, the sampler produces about 170 stipples/s. We will make this the default sampler and
  update Fig. 7."
- **Determinism, not the step count, is what hurts DDIM η=0 and DPM++** (HOT +13–17%,
  centroid +19–48%). The per-step noise is needed for blue-noise spacing. This confirms the
  3-image smoke result.
- **The clean-start idea is a negative result.** Skipping the forward noise is much worse at
  every start step, worst at t=500 (δc +6745%). The model expects the noise level of its
  timestep, so a clean Smart Init is out of distribution. Do not run the sweep on Ours-BNOT.
- **Starting later (ddpm-K with K < 500) also hurts**, monotonically. This supports the
  paper's t=500 SDEdit start.
- Sanity check: ddpm-500 here (HOT 4.08, δc 0.0178) agrees with Table 1's Ours-GBN (4.059,
  0.018) up to the unseeded noise and the metric-code drift noted above.

### Open questions and fixes for E1

1. **Why do 10 η=1 steps beat 500 DDPM steps?** The likely cause is the variance, not the step
   count: the paper's DDPM loop uses the fixed "large" variance β, while DDIM with η=1 uses the
   posterior variance β̃, which is smaller. A reviewer may ask. Cheap check: run `ddim-500-eta1`
   (and maybe `ddim-250-eta1`) on Ours-GBN. If it matches `ddim-10-eta1`, the gain comes from
   the variance choice, and the rebuttal should say so instead of implying fewer steps are
   better.
2. **Report total time, not just denoising.** At 10 steps, Smart Init (measured 0.13–0.18 s on
   the CPU, see the final summary) is larger than the denoising (0.11 s), so per-image time is
   about 0.25–0.28 s. Either report the total or move Smart Init to the GPU and re-time.
3. **Timing across jobs is not comparable.** Ours-BNOT's ddpm-500 took 3.23 s against Ours-GBN's
   5.42 s because the jobs ran on different GPUs (RTX 6000 Ada vs L40S). For the paper, use the
   throughput file or re-time all samplers in one job on one GPU. Add the η=1 samplers to the
   throughput measurement.
4. **FIXED 2026-10-28. Race in `few_step_sampling_stage_2.py` (line 60).** The temporary file was
   `json_path.with_suffix(".tmp")`, so two jobs scoring the same image collide on the rename.
   This caused the "FAILED" shards 23619244_8 and _12 (4 errors in total). It was harmless:
   the other job wrote the JSON, and the summary jobs use `afterany`, so the chain does not
   stall. Fix: include the process id in the temp name, e.g.
   `json_path.with_suffix(f".{os.getpid()}.tmp")`. It also means jobs over the same folders
   duplicate work, which is part of why Ours-GBN dpmpp2m was at 946 / 866 images.
   - The race also corrupted one file: `CN-GBN_1024/ddpm-300_json/google_3_raised_hand.json`
     (a valid JSON object followed by a stray `}`). It made summary job 23619686 fail. A scan of
     all 31,000 E1 metric files found no other bad file and no leftover `.tmp`.
   - Fix applied: line 60 is now `tmp = json_path.with_suffix(f".{os.getpid()}.tmp")`. The
     previous version is saved as `few_step_sampling_stage_2.py.bak_20261028`.
   - The corrupted file is kept in `experiments/outputs/z_corrupt_json_backup/`. It was deleted
     from the run and re-scored (job 23650999). Every sampler now has 1000 JSONs.
5. **Still to do from the E1 plan:** timing at 4096 and 6400 points.
6. **Decision for the user:** for the January revision, either switch the paper's default
   sampler to ddim-10-eta1 (or ddim-25-eta1) and re-run every table, or keep the DDPM-500 tables
   and add the few-step result as a new row and an extra curve in Fig. 7. The second option is
   less work and less risky under the "no large unrequested changes" rule.

## Rebuttal files (2026-10-28)

- [REBUTTAL_DRAFT.md](REBUTTAL_DRAFT.md): the evidence bank (each result with a status) and draft
  answers to the expected concerns. Keep it current as results come in.
- [REBUTTAL_PLAN.md](REBUTTAL_PLAN.md) section 6: new experiments E13–E18, added 2026-10-28.
  **E13 (user request):** once the variance check fixes the sampler, re-run Ours-WVS / Ours-GBN /
  Ours-BNOT on all 4 test sets with it. All 12 epoch-5000 checkpoints exist:
  `control_v4/train_outputs_<Dataset>_<WVS|GBN|BNOT>/checkpoints/dynamic_ep5000.ckpt`, except Icons-50
  GBN, which is `train_outputs_Icons-50_1024_GBN_full`. Stage 1 takes `--source`, `--grid_size`
  (32 / 40) and `--label`, so it needs only a new job script.
- The user's latest paper files are on the server for later edits: `paper4/main_EG_rebuttal_11.10.tex`,
  `paper4/supplemental_EG_rebuttal_11.10.tex` and `paper4/tables_rebuttal/`. **Do not edit them until
  the experiments are done.**

## 2026-10-29 / 10-30: sampler choice, runtime, OT-assignment bug, remaining experiments

### Decisions (user, 2026-10-30)

- **New variant = stochastic DDIM, 25 steps (`ddim-25-eta1`)**, same SDEdit start (t = 500).
- Order of work: first finish every remaining rebuttal experiment (paper configuration, DDPM-500,
  with ddim-25-eta1 sampled on the same images where it is nearly free). Then re-run with the new
  variant for the supplement: the datasets table (E13), the runtime graph, and a copy of the
  ablation table. This is kept in reserve "if we get attacked hard", offered as an improvement
  for a second review round.

### Variance check (A1b) — done

Ours-GBN, Icons-50, change vs ddpm-500 (paired, 1000 images): ddim-500-eta1 δc −1.9%, ρs +0.4%;
ddim-250-eta1 δc −3.0%; ddim-100 −4.4%; ddim-50 −6.4%; **ddim-25 −8.4% (ρs +1.5%)**; ddim-10 −5.1%.
So the smaller posterior variance alone gives a small gain, and fewer steps add most of it.
ddim-25-eta1 is the best on every metric (0.27 s denoising).

### Icons-50 table with ddim-25-eta1 (all six methods re-scored with the current metric code)

`experiments/outputs/rescore_current_metrics-Icons-50_1024/` holds BNOT/GBN/WVS and the paper's
Ours point sets (`CN-*-paper`, symlinks) with current-code JSONs. Change of the new variant vs the
paper's own point sets: Ours-BNOT δc −8.8%, Ours-GBN −9.2%, Ours-WVS −2.9%, ρs +1.5–1.6%, every
other metric equal or better (no significant degradation). Rank preservation on means after the
switch: 73/74 non-tied pairs (unchanged; the flip is ShapeNet2D CVT BNOT vs WVS).

### Runtime with the new sampler (job 23686760, RTX 6000 Ada, ise-6000-08)

`experiments/runtime_fewstep.sbatch` + `runtime_fewstep_stage_2.py` →
`experiments/outputs/runtime_fewstep-Icons-50/summary.md` and `plots/runtime_fewstep.{png,pdf}`.
Ours-GBN, 10 pinned runtime images, 256…6400 points, total = Smart Init + denoise + OT. DDIM-25:
0.38 s (256) / 0.64 s (1024) / 7.25 s (6400); DDIM-10: 0.16 / 0.42 / 7.04 s; ddpm-500 re-timed on
this node: 7.46 / 7.65 / 14.71 s (the paper's Fig. 7 measured 5.88 s at 1024, faster machine).
Denoising is flat (0.36 s DDIM-25, 0.145 s DDIM-10); the CPU Smart Init (mostly the OT assignment)
dominates from ~1600 points. Throughput at batch 32, 1024 points: 60 (DDIM-25) / 149 (DDIM-10)
stipples/s.

### OT-assignment bug at large budgets (E18) — found 2026-10-29

- Smart Init assigns the N points to the OT grid with POT `ot.emd` (`data/Transforms.to_image_optimal_transport`),
  default `numItermax = 100000`. From **3136 points up it stops before optimality** ("numItermax
  reached before optimality") and returns a feasible but non-optimal assignment. Total cost above
  the exact optimum: 3136: 0.1–1.2%, 4096: 9–18%, 5184: 19–46%, 6400: 33–76%; up to 2304 points the
  assignment is exactly optimal. **Training (1024/1600 points) and all paper tables are unaffected;**
  the 4096-point capacity-generalization figure and the 3136–6400-point timings used the truncated
  assignment.
- New `control_v4/ot_assignment.py`: solvers `pot` (paper path), `pot-exact` (ot.emd with
  numItermax 1e9, raises if not optimal), `scipy` (linear_sum_assignment) and `auction` (GPU,
  epsilon scaling). `smart_init_points_to_offsets(points, ot_solver="pot", device)` and
  `few_step_sampling_stage_1.py --ot_solver` (default `pot`, so all stored results stay
  reproducible). Backups `*.bak_20261030`.
- Benchmark (`ot_assignment_benchmark.py`, job 23689195, `outputs/ot_assignment_benchmark_full/summary.md`),
  median s at 4096 / 6400 points: pot 1.6 / 2.5 (truncated), **pot-exact 2.4 / 5.7 (exact)**,
  scipy 7.3 / 26, auction 15 / 18 (exact, but 20–30k sequential rounds; `auction_diag.py`). So the
  fix is simply `pot-exact`; the GPU auction is not worth it.
- Quality check (`ot_exact_quality.sbatch`, jobs 23689520–23 sampling, 23689545–48 metrics; summary by
  `ot_exact_quality_stage_3.py` → `outputs/ot_exact_quality-Icons-50/summary.md`): Ours-GBN, first 200
  Icons test images, 4096 and 6400 points, ddpm-500 and ddim-25-eta1, `pot` vs `pot-exact`, same
  Smart Init points and noise. Visual check on the capacity-figure images
  (`ot_exact_quality_figure.sh`, `outputs/ot_exact_quality-capacity_figure/`): the truncated
  assignment leaves stray points outside the shape at 5184–6400 points; the exact one removes most
  of them; some mottling remains at 6400 either way.
- **Result (200 images, paired Wilcoxon, all significant unless noted):** exact vs paper assignment,
  ddpm-500 / ddim-25-eta1. At 4096: δc −2.8% / −3.2%, centroid −1.8% / −2.3%, EMD −0.7%, HOT −0.4%
  (n.s.) / −1.3%. At 6400: **δc −19.1% / −20.2%**, centroid −10.4% / −12.9%, HOT −8.3% / −9.9%, CVT
  −5.7% / −6.1%, EMD −7.1% / −7.5%, ρs +0.3%. No metric gets worse. Smart Init time 1.66 → 2.52 s
  (4096) and 2.50 → 5.86 s (6400). These metrics only justify the fix internally; they are budget
  metrics, so they stay out of the paper and the rebuttal (see the ground rules).
- `few_step_sampling_metrics.sbatch` now takes `ROOT` and `SRC` (defaults unchanged).

### Remaining rebuttal experiments, submitted 2026-10-30

| Job(s) | What |
|---|---|
| 23689745 | **E8** KDE-loss ablation training: `train_ablation_nokde.sbatch`, Ours-GBN Icons, the "full" run's settings with `--density-loss-weight 0`, out `control_v4/train_outputs_Icons-50_1024_GBN_noKDE`. ~75 s/epoch → ~4 days. Resumable: resubmit the same script if it stops. Then evaluate like Table 2 (50-image subset, `ablation_advance_metrics_stage_*`). |
| 23689783–85 | Re-score all six methods (paper point sets) with the current metric code: `rescore_current_metrics-{CelebA-5K_1024,ShapeNet2D-4K_1600,Anime-Chibi_1600}` (baseline for E2, E5, E13) |
| bg | **E2** `rank_preservation.py` (paired Wilcoxon per pair, means version, Kendall τ) → `outputs/rank_preservation/summary.md`, run after the re-score |
| 23689833–40 | **E5** cross-dataset, `rebuttal_sample.sbatch` (ddpm-500 + ddim-25-eta1): Icons→CelebA, CelebA→Icons, ShapeNet2D→Chibi, Chibi→ShapeNet2D, GBN and BNOT variants → `outputs/cross_dataset-<target>/CN-<M>-from<src>_<N>`; metrics 23689902–09 |
| 23689846–63 | **E4, archived 2026-11-01** (budget generalization, metric version): results moved to `experiments/outputs/a_archive/E4_budget_generalization/`. Not to be used. |
| 23689890–92 | **E7** direct point optimization for tone (`tone_results_stage_4_direct_points.py`): from our uncorrected stipple and from the GBN / BNOT stipples, Adam on the points with repulsion weight μ ∈ {0, 1e-3, 1e-2, 0.1}, same renderer and objective as stage 1; tone + distribution metrics → `outputs/tone_results/direct_points/summary.md` |
| 23689898 | **E17** `rendered_tone_fidelity.py`: every Table 1 point set (and Icons ddim-25) rendered with the tone renderer, PSNR/SSIM/MAE → `outputs/rendered_tone_fidelity/summary.md` |

### E2 rank preservation — done (2026-10-30)

`rank_preservation.py` → `outputs/rank_preservation/summary.md`, all six methods re-scored with the
current metric code on all 4 datasets (paper point sets). 84 pairs (4 datasets × 7 metrics × 3
sampler pairs):
- Paired Wilcoxon (p < 0.05) decides the order: **78 agree, 4 flip, 2 ties** (78 of 82 decided
  pairs = 95%).
- Means at table rounding: 73 agree, 2 flip, 9 ties (the earlier "73 of 74" was from the stored,
  mixed-code JSONs; with one metric code it is 73 of 75).
- Kendall's τ (3 sampler means vs 3 learned means) per dataset × metric: mean 0.90, τ = 1 in 24 of 28.
- Three of the four flips are capacity δc, BNOT vs GBN (Icons, CelebA, ShapeNet2D), where the two
  samplers are within 1–5% of each other (e.g. Icons 0.0094 vs 0.0099); the fourth is the known
  ShapeNet2D CVT, BNOT vs WVS. Ties: ShapeNet2D ρs BNOT–GBN and Chibi δc BNOT–GBN (learned side n.s.).
- Rebuttal wording: "where the exemplar samplers differ significantly, the learned variants
  reproduce their order in 78 of 82 cases; the exceptions are near-ties between BNOT and GBN on
  capacity."

### E5 cross-dataset — done (2026-10-30)

`cross_dataset_stage_3.py` → `outputs/cross_dataset_summary.md`. Paired vs the in-domain paper
result on the same images (current metric code). ddpm-500, δc change vs in-domain:
- Icons → CelebA: GBN +22%, BNOT +8% (HOT +31% / +18%). Ours-GBN from Icons still beats GBN itself
  on HOT (10.8 vs 12.9) and EMD (0.0169 vs 0.0183).
- ShapeNet2D → Chibi: +66% / +51%; Chibi → ShapeNet2D: +54% / +44%. HOT stays at or below GBN.
- **CelebA → Icons fails on capacity:** δc ×5.7 / ×6.1 (0.10 / 0.12 vs 0.018 / 0.019), HOT +46% / +79%.
- ddim-25-eta1 is slightly better than ddpm-500 in every cross-domain cell.
- Reading: transport quality transfers across domains, capacity does not; training on the diverse
  Icons set transfers to faces, training on faces does not transfer to icons.

### E4 — archived (user decision, 2026-11-01)

Budget generalization is shown visually only (see the ground rules). The runs are in `experiments/outputs/a_archive/E4_budget_generalization/`,
the scripts in `experiments/archive/E4_budget_generalization_scripts/`.

### E7 direct point optimization — done, finer sweep running (23703842–44)

`outputs/tone_results/direct_points/summary.md` (24 targets). Ours-field (the paper's method):
PSNR 16.59, δc 0.049, clumped 5.4%. Direct optimization of the points, from any start (ours / GBN /
BNOT, nearly identical results): μ=0 PSNR 18.9–19.0 but δc 0.36, centroid 0.093, 20% clumped;
μ=0.01 PSNR 17.2, δc 0.10, 9.5–9.9% clumped; μ=0.1 collapses (PSNR 12.4). So direct optimization
buys tone by breaking the point distribution; whether some μ matches our distribution quality
with higher PSNR is what the finer sweep (μ = 0.02, 0.03, 0.05 → `direct_points_fine/`) checks.
Caveat for the rebuttal: the uncorrected GBN stipple already renders at PSNR 16.81 (> ours-field
16.59) with this renderer; our tone claim is the differentiable correction of our sampler
(15.76 → 16.59), not beating GBN's tone.

### E17 rendered-tone fidelity — done

`outputs/rendered_tone_fidelity/summary.md`. GBN has the best rendered tone on all 4 datasets;
the learned variants are 0.3–1 dB below their samplers (Icons: GBN 16.77, Ours-GBN 16.44, Ours-GBN
ddim-25 16.50 dB). The learned variants keep the samplers' order on Icons (PSNR GBN > BNOT > WVS,
SSIM GBN > WVS > BNOT, same for Ours). Weak as ammo; use only for rank preservation if asked.

### E3 step 1 (aspect ratio) — done, negative

`aspect_ratio_capacity.py` → `outputs/aspect_ratio_capacity/summary.md` (stored Table 1 metrics).
**Aspect ratio is not the cause of the capacity gap.** On images that are already square
(|log(w/h)| < 0.05), the learned variants still have 2–4× the samplers' δc: Chibi Ours-GBN 0.094 vs
GBN 0.024 (107 images); ShapeNet2D 0.035 vs 0.012 (163 images). The per-image gap correlates only
weakly with anisotropy (Spearman ≤ 0.17 on Chibi, ≈ 0 on ShapeNet2D). So E3 steps 2–3 (padding,
rectangular grid) were not run. Across datasets the gap ratio (learned / sampler δc, GBN) is 1.8×
(Icons, 10k training images, 1024 pts), 1.7× (CelebA, 5k, 1024), 3.0× (ShapeNet2D, 4k, 1600),
3.6× (Chibi, 2.6k, 1600): it tracks training-set size and point budget, which are confounded here.

### E6 / E10 cost numbers (from stored measurements)

- BNOT per image (paper's runtime measurement): 13.6 s at 1024, 25.5 s at 1600 points. Exemplar
  generation therefore ≈ 10000 × 13.6 s = 38 h (Icons), 19 h (CelebA), 28 h (ShapeNet2D), 19 h
  (Chibi) of single-image BNOT time; GBN is 8.1 s per image at 1024, so BNOT exemplars cost only
  ~1.7× GBN's. Contradicts the "BNOT is impractical for exemplars" wording.
- Training wall time per checkpoint (checkpoint mtimes, upper bounds): 31–95 h for 5000 epochs
  (Icons GBN full 95 h, CelebA 34–57 h, ShapeNet2D 48–68 h, Chibi 31–50 h; Icons WVS 159 h, likely
  with queue gaps).

## 2026-10-30 / 10-31: tone correction, runtime (recorded 2026-11-01)

These results existed on the server but were not yet in this file. No jobs were running on
2026-11-01.

### Tone correction: the paper's numbers come from the one-step proxy (important)

- **How the paper's Table 3 was made.** `tone_results_stage_1_optimize.py` (`MODE = "onestep"`,
  `T_PROBE = 120`) renders `stippler(rho, generator=gen)` with a fixed generator (`SEED = 0`). So
  the stipples scored in Table 3 (uncorrected 15.76 → field 16.59 dB, +0.83 dB, 24/24) are the
  **one-step decode at t = 120 with the same noise used during optimization**, not outputs of the
  real sampler. The paper only says the *gradient* goes through one step, so a reader assumes the
  real sampler's output was scored.
- **On the real sampler the one-step result mostly disappears**
  (`outputs/tone_results/full_sampler/summary.md`, 24 targets, fresh seeds 0/1/2):
  - The real sampler is already better uncorrected: ddpm-500 16.51 dB, ddim-25-eta1 16.55 dB
    (vs the proxy's 15.76).
  - The one-step-optimized **field** gives +0.09 dB (10/24 images improved, p = 0.9) on
    ddpm-500 and +0.09 dB (11/24, p = 0.8) on ddim-25-eta1: no significant gain.
  - The one-step-optimized **tone curve** still helps: +0.28 dB on both samplers (20/24 and 18/24,
    p ≤ 4e-4).
- **E16, gradients through the full DDIM-25 chain, fixes it**
  (`outputs/tone_results/full_sampler_ddim25/summary.md`, fresh seeds):
  - field: ddim-25-eta1 16.95 dB, +0.39 dB (18/24, p = 0.002); ddpm-500 16.89 dB, +0.38 dB (17/24,
    p = 0.004).
  - curve: +0.36 / +0.37 dB (20/24 and 21/24).
  - On its own optimization path (fixed noise), the field reaches 17.29 dB (+0.76 dB, 21/24). The
    drop to +0.39 dB on fresh seeds is presumably the optimization fitting the fixed noise (not
    verified).
  - **The corrected 16.95 dB beats GBN's own rendered tone (16.81 dB on the same 24 targets,
    `direct_points_fine`)**, which the paper's one-step 16.59 dB did not.
- **Cost (D3, `outputs/tone_results/cost/summary.md`, RTX 6000 Ada):** one step 0.027 s per
  optimizer step and 2.5 GiB, 8 s per image for 300 steps; full DDIM-25 chain 0.557 s per step and
  5.1 GiB, about 155–167 s per image.
- **D1, gradient-free fixed point** (`outputs/tone_results/fixed_point_smoke/summary.md`): a smoke
  test on 2 images only. One multiplicative update gives +0.05–0.08 dB on GBN and −0.07 dB on
  ddim-25-eta1. Feeding *our* DDIM-25-optimized densities to GBN gave +0.31 dB (curve) and
  +0.39 dB (field) on those 2 images, a hint that the corrected density transfers to other
  samplers. Not significant (n = 2); run on all 24 targets before using it.
- **What to do in the revision:** replace Table 3 with the E16 numbers on the real sampler (fresh
  seeds) and describe the protocol exactly. Change "improves all 24 targets" to the measured count
  (18/24). The "full back-propagation as future work" sentence becomes a result.

### E7 finer sweep — done (`outputs/tone_results/direct_points_fine/summary.md`)

Direct optimization of the points, μ = 0.02 / 0.03 / 0.05, from our / GBN / BNOT start (nearly
identical). μ = 0.02: PSNR 16.12–16.16, δc 0.156–0.159, 2.1% clumped. μ = 0.03: 15.27, δc 0.23.
μ = 0.05: 14.0, δc 0.35. So **no repulsion weight gives both our tone and our distribution**: the
weights that keep spacing (μ ≥ 0.02) render worse than our corrected sampler (16.59 one-step,
16.95 full chain) with 3–7× its capacity deviation (ours-field δc 0.049); μ = 0 renders at
19.0 dB only by destroying the distribution (δc 0.36, 20% clumped). Note that the direct
optimization raises ρs (0.60–0.63 vs 0.49) because of the repulsion term, while capacity
collapses, so quote δc and clumping, not ρs, in this comparison.

### Runtime of the new variant and the exact-OT figure (2026-10-31)

- `runtime_new_variant.sbatch` (ddim-25-eta1 and ddpm-500 with `--ot_solver pot-exact`, 10 pinned
  images) wrote `outputs/runtime_new_variant-Icons-50/runtimes_avg_new_variant.json` and plots.
  Two re-timing runs, `runtime_new_variant_exclusive-Icons-50/` and
  `runtime_new_variant_dedicated-Icons-50/` (1024–6400 points), have timing files but no summary.
  **Superseded by the clean run (2026-11-01 section below); the first run is contaminated.**
- Reference times from the same JSON (paper measurement): BNOT 13.6 s at 1024 and 1802 s at 6400;
  GBN 8.1 s / 10.2 s; the paper's sampler (CN) 5.88 s / 9.08 s.
- `grid_size_results/pointcount_generalization_exactOT_{ddpm-500,ddim-25-eta1}.{png,pdf}`: Fig. 4
  regenerated with the exact OT assignment (E18). Use it in the revision. Together with the
  runtime graph, this is the only place point budgets appear.

## 2026-11-01: exact-OT check, clean runtime, E8 evaluation

### Exact OT is used where it matters (checked)

- Fig. 4: `grid_size_exact_ot.sbatch` (job 23736982) passes `--ot_solver pot-exact`; every row of
  `grid_size_results/exact_ot/CN-GBN_<N>/timings.jsonl` (576-6400) records `"ot_solver": "pot-exact"`.
  The figures `pointcount_generalization_exactOT_{ddpm-500,ddim-25-eta1}` (10-31 16:41) were drawn
  after those runs, with the paper's columns (grids 24/32/48/64 = 576-4096 points).
- Runtime: all three new-variant runs (`runtime_new_variant`, `_exclusive`, `_dedicated`) record
  `pot-exact` in every row. The older `runtime_fewstep-Icons-50` (10-29) used the truncated `pot`:
  do not use its 3136-6400-point numbers. At ≤ 2304 points `pot` is exact, so E13 (1024/1600) and
  the tone experiments (grid 32) are not affected.

### The plotted runtime run is contaminated; clean protocol

- `runtime_new_variant-Icons-50` (job 23736983, the one `runtimes_avg_new_variant.json` and its
  plot were made from) ran on ise-6000-09 **at the same time as the grid-size job on the same node**
  (shared 8-GPU node). Its Smart Init is inflated from 3136 points on: ddim-25 total 9.55 s at 5184
  vs 4.21 s (exclusive) and 4.47 s (16-CPU) runs; 9.18 vs 6.76 / 8.05 s at 6400. **Do not use it.**
- The exclusive (23737224, whole cpu256 node) and 16-CPU (23737225) runs agree up to 5184 points
  but differ at 6400 (Smart Init 6.44 vs 7.75 s), single repetition each, thread pools not pinned.
- **Clean run, job 23786262** (`runtime_clean.sbatch`): exclusive single-GPU cpu256 node (RTX 6000
  Ada), thread pools pinned to 16 (OMP/MKL/OpenBLAS/NumExpr), CPU model / GPU / load / git hash
  logged, ddim-25-eta1 and ddpm-500 with `pot-exact`, 10 pinned images, 256-6400 points, warm-up
  per budget, **3 full repetitions** (`outputs/runtime_clean-Icons-50/rep{1,2,3}`). Summary:
  `python experiments/runtime_clean_stage_3.py` (per-image median over reps, then mean over images;
  writes `summary.md`, `runtimes_avg_clean.json` and the Fig. 7 plot with the re-timed DDPM-500).
- **Baseline runtimes are final (user decision, 2026-11-01).** BNOT / WVS / GBN and the paper's
  CN curve stay the stored paper measurements (`icons_results_runtimes/runtimes_avg.json`). Do not
  re-time the baselines. Any re-run of our runtime uses only the same 10 pinned runtime images
  (`runtime_fewstep-Icons-50/source`), as `runtime_clean.sbatch` does.
- Our timed total = Smart Init + denoising + OT-grid to points; image loading and condition
  preparation are not included (as in runtime_fewstep / runtime_new_variant).

### E8 evaluation, submitted 2026-11-01

Training finished 2026-11-01 15:25 (`train_outputs_Icons-50_1024_GBN_noKDE`, epochs 500-5000).
Evaluated with the Table 2 "Full + SDEdit" settings on the same 50 images, into its own folder
`outputs/ablation_nokde_e500_b50_1024/` (resources symlinked to the Table 2 folder; the shared
ablation scripts are not edited, `ablation_nokde_stage_{1,2,3}.py` set their globals):
- job 23786263 (`ablation_nokde_export.sbatch`): `sdedit_noKDE` (every 500 epochs) and
  `sdedit_seeded` (the paper's full checkpoint, epoch 5000). Table 2 did not seed the noise (its
  sdedit δc is 0.0273 at epoch 4500 vs 0.0255 at 5000), so both are sampled with
  `torch.manual_seed(crc32(stem))` per image, giving a paired comparison.
- job 23786265 (array 0-10, after the export): scoring. Then `python experiments/ablation_nokde_stage_3.py`
  → `summary.md` (paired Wilcoxon at epoch 5000 + noKDE curves).

## Notes for the rebuttal (2026-11-01)

1. **Tone correction is the main exposure.** If a reviewer asks how the tone stipples were made, or
   questions the one-step gradient, answer with E16 on the real sampler (+0.39 dB, 18/24, and above
   GBN's tone). Do not quote +0.83 dB or "all 24 targets" for the real sampler.
2. **Point budgets: visual only.** Answer budget questions with Fig. 4 (exact OT) and the runtime
   graph. No metrics at budgets the model was not trained on. E4 is archived.
3. **The capacity-gap explanation changed.** E3 ruled out the aspect ratio. Icons (10k training
   images) and CelebA (5k), both at 1024 points, have the same gap ratio (1.8× and 1.7×), so halving
   the training set did not change it; the 1600-point datasets have 3.0–3.6×. This points at the
   point budget (or the domain) more than the training-set size, but the experiments do not
   separate them.
4. **Consider running E13 now rather than keeping it in reserve.** It costs minutes of sampling and
   about 36 CPU-hours of scoring, and the rebuttal week leaves no time to compute. The draft already
   commits to the faster sampler.

## Next steps

1. E8: when jobs 23786263 / 23786265 finish, run `ablation_nokde_stage_3.py` and record the result.
2. Runtime: when job 23786262 finishes, run `runtime_clean_stage_3.py` and use only that run for
   the runtime graph (baselines stay the stored paper numbers).
3. D1 fixed point on all 24 targets, if the transfer-to-GBN hint is worth checking.
4. User's second phase: E13 full re-run with ddim-25-eta1 (see note 4), ablation table copy, E14
   seeds, E15 spectra.

## 2026-11-07: E8 result, D1 restart, interim runtime graph

- **E8 done** (`outputs/ablation_nokde_e500_b50_1024/summary.md`, `ablation_nokde_stage_3.py`): at epoch 5000 on the 50 paired images, noKDE vs full differs on no metric (all p > 0.08): δc 0.0232 vs 0.0264 (-12%, p=0.64), HOT +2.7% (p=0.23), D +3.8% (p=0.084), EMD +1.4%, CVT / Sinkhorn / ρs within 0.3%. The KDE loss has no measurable effect on the final checkpoint; the noKDE run is slower early (δc 0.033 at epoch 500-1500 vs 0.024-0.029 from 2000 on). Do not claim a KDE-loss benefit from Table 2.
- **D1 first full run (23737218) failed:** GBN hung for 4 h on google_4_family_man_man_boy at iteration 8 and the job was cancelled; only the GBN in-loop numbers exist (alpha 1: mean PSNR 16.78 at iteration 0, 17.05 at iteration 2, then oscillating 16.78-16.98). Kept in `tone_results/fixed_point_alpha1_partial/`.
- **D1 re-run, job 23962618** (`tone_d1_d3.sbatch`, RUN_D3=0): stage 7 now runs GBN one process per image with a 300 s timeout and one retry (failed images keep their density and are left out), alpha 0.5, 6 iterations. -> `tone_results/fixed_point/summary.md`.
- **Interim runtime graph** from the exclusive run 23737224 (whole cpu256 node, RTX 6000 Ada, 1 rep): `runtime_new_variant_exclusive-Icons-50/plots/runtime_scaling_by_points.{png,pdf}` (`runtime_new_variant_stage_3.py --with-ddpm-exact`). DDIM-25 exact OT 0.31/0.46/1.02/3.08/6.76 s at 256/1024/2304/4096/6400; DDPM-500 exact OT 6.21/6.30/7.00/9.16/13.20 s. The clean 3-rep job 23786262 is estimated to start 2026-11-08 22:05; use it instead when it finishes.

## 2026-11-10: fast Smart Init and pure-noise start (runtime)

### Decisions (user, 2026-11-10)

- Goal: make inference feel instant. From ~2300 points the CPU Smart Init (rejection sampling +
  exact OT assignment) dominates our runtime (6400 points: ~6.4 s of 6.8 s); denoising is flat
  (~0.3 s, DDIM-25).
- SDEdit was introduced to cut runtime, not for quality (the quality gain was a bonus). So a
  pure-noise start with few DDIM steps is a legitimate alternative if it is fast and as good.
- **We stay a pure diffusion method.** "Ours + a few teacher (GBN/WVS) iterations" is a last
  resort, only if the reviews make it necessary. Do not run or propose it before then.

### Fast assignment candidates (`control_v4/fast_smart_init.py`, new)

`ot_assignment.py` (backup `.bak_20261110`) has three new `--ot_solver` values, all returning the
same offset grid as `to_image_optimal_transport` (cell order checked against it):
- `bisection`: recursive median split of the points along a recursive split of the grid.
- `bisection-swap`: + local 2-opt swaps between nearby cells.
- `bisection-blocks`: + exact assignment (linear_sum_assignment) inside 8 x 8 cell tiles,
  tiling shifted by half a tile every other sweep, 6 sweeps.
Also tried and dropped (synthetic densities, local test): sort-by-x-then-y (+40-90% cost) and a
Knothe-Rosenblatt grid warp that needs no assignment (+4-70% vs the exact cost of rejection points,
bad on thin lines and multi-part shapes).

Smoke job 24052948 (real icons, 10 runtime images; cost ratio vs exact, median / max, time):
1024 points: exact 0.31 s; bisection 1.086 / 1.48, 4 ms; -swap 1.045 / 1.33, 12 ms; -blocks
1.011 / 1.08, 49 ms. 4096 points: exact 6.7 s; bisection 1.080, 13 ms; -swap 1.049, 48 ms;
-blocks 1.028 / 1.15, 0.23 s. Rejection sampling itself: 3-7 ms. Reference from E18: the truncated
paper solver at 4096 was 9-18% above optimal and cost ~3% δc; at 3136 it was 0.1-1.2% and harmless.

### Jobs submitted 2026-11-10

| Job(s) | What |
|---|---|
| 24052987 | `fast_ot_benchmark.sbatch`: cost / time of every candidate, 10 runtime images at 256-6400 points + 200 Icons images at 1024 → `outputs/fast_ot_benchmark/summary.md` |
| 24052988 / 90 / 92 / 94 | quality check at the training budget (1024), first 200 Icons test images, ddpm-500 + ddim-25-eta1, same Smart Init points and noise, solvers pot-exact / bisection / bisection-swap / bisection-blocks (`ot_exact_quality.sbatch 32 <solver> <src> outputs/fast_ot_quality-Icons-50 200 16`); metrics 24052989 / 91 / 93 / 95. Then per candidate: `python experiments/ot_exact_quality_stage_3.py experiments/outputs/fast_ot_quality-Icons-50 --paper pot-exact --exact <solver> --name summary_<solver>` (new `--paper / --exact / --name` options, backup `.bak_20261110`) |
| 24052996-98 | `fast_ot_visual.sbatch` (SOLVER=bisection / -swap / -blocks): Fig. 4 images, ddim-25-eta1, grids 24-80 → `grid_size_results/fast_ot_<solver>/` and `pointcount_generalization_<solver>_ddim-25-eta1.pdf`; compare visually with `pointcount_generalization_exactOT_ddim-25-eta1.pdf` (budgets ≠ 1024: visual only) |
| 24052999 (array 0-2) | **pure-noise start** (`few_step_noise.sbatch`): ddim-25/50/100-eta1-noise and ddpm-1000-noise, all 1000 Icons test images → `outputs/few_step_noise-Icons-50_1024/CN-GBN_1024/` (links to the E1 ddpm-500 and ddim-25-eta1 results); metrics 24053000. Then `python experiments/few_step_sampling_stage_3.py experiments/outputs/few_step_noise-Icons-50_1024` |

`few_step_sampling_stage_1.py` (backup `.bak_20261110`) has a new `-noise` sampler suffix: start
from the per-image noise tensor at t = T - 1 (timesteps T-1 ... 0), no Smart Init; when every
sampler of a run is `-noise` the Smart Init is skipped and its time is recorded as 0.

### Next

1. Pick the solver: the fastest whose 1024-point metrics are not significantly worse than
   pot-exact and whose Fig. 4 images look the same; compare against the pure-noise result.
2. Then re-time our method with it (`runtime_clean.sbatch`, 3 reps, same 10 images; job 23786262
   with pot-exact has been waiting for a whole free node since 2026-11-01).

## 2026-11-10 (later): results of the Nov 10 jobs, D1 fix

All Nov 10 jobs completed. Their stage-3 summaries had not been built; job 24060326 built them.

### Fast Smart Init assignment: none passes at 1024 points

`outputs/fast_ot_quality-Icons-50/summary_<solver>.md` (200 Icons images, same Smart Init points
and noise, paired Wilcoxon). Change vs `pot-exact`, ddim-25-eta1 / ddpm-500:

| solver | cap δc | centroid | HOT | Smart Init s (exact 0.15) |
|---|---|---|---|---|
| bisection | +37.3%* / +30.5%* | +15.9%* / +19.1%* | +11.2%* / +12.6%* | 0.00 |
| bisection-swap | +21.4%* / +17.2%* | +15.1%* / +14.3%* | +9.6%* / +8.6%* | 0.01 |
| bisection-blocks | +13.6%* / +12.6%* | +10.0%* / +9.6%* | +5.7% / +4.1%* | 0.04 |

- Every candidate is significantly worse on capacity and centroid at the training budget. Even
  bisection-blocks, whose assignment cost is only ~1% above optimal, gives +13% δc, which would
  cancel most of ddim-25-eta1's −8.4% gain. **The model is sensitive to small assignment errors.**
- At 1024 points the exact assignment costs only 0.15 s, so a fast solver buys little there; it
  matters only at large budgets (exact 6.4 s of 6.7 s at 6400 points, clean run below).
- Not quality-tested yet: `bisection-blocks16` (benchmark cost ratio 1.004 median at 1024, 0.20 s;
  1.018 at 6400, 1.97 s). It is the only candidate close to exact.
- `outputs/fast_ot_benchmark/summary.md` has the cost/time table (256–6400 points).

### Pure-noise start: worse than the SDEdit start

`outputs/few_step_noise-Icons-50_1024/summary.md` (1000 images, vs ddpm-500 with SDEdit):

| sampler | denoise s | cap δc | HOT | centroid | EMD | ρs |
|---|---|---|---|---|---|---|
| ddim-25-eta1 (SDEdit, the new variant) | 0.27 | −8.4%* | −0.6%* | −1.2%* | −0.1%* | +1.5%* |
| ddim-25-eta1-noise | 0.26 | +1.5%* | +12.2%* | +14.5% | +16.3%* | +2.0%* |
| ddim-50-eta1-noise | 0.52 | −1.7%* | +10.5%* | +12.2% | +14.5%* | +1.8%* |
| ddim-100-eta1-noise | 1.04 | −0.8%* | +9.6% | +11.2% | +13.3%* | +1.4%* |
| ddpm-1000-noise | 10.57 | +4.8% | +8.0% | +8.9% | +12.3%* | +0.3%* |

- **The Smart Init (SDEdit) start improves quality, not only runtime.** From pure noise, every
  sampler is 8–16% worse on HOT, centroid and EMD than the SDEdit start, including the full
  1000-step DDPM. Keep the SDEdit start. This is also an argument for the design if a reviewer
  questions it.

### Clean runtime (job 23786262, final)

`outputs/runtime_clean-Icons-50/summary.md` (+ `runtimes_avg_clean.json` and the Fig. 7 plot from
`runtime_clean_stage_3.py`). Exclusive node, RTX 6000 Ada, exact OT, 10 pinned images, 3 reps, max
spread over reps ≤ 6.8%. Total s (Smart Init + denoise):

| points | 256 | 1024 | 1600 | 2304 | 4096 | 6400 |
|---|---|---|---|---|---|---|
| DDIM-25 | 0.31 | 0.44 | 0.61 | 0.99 | 3.06 | 6.71 |
| DDPM-500 (re-timed) | 6.18 | 6.29 | 6.53 | 6.64 | 9.12 | 13.15 |
| Smart Init (both) | 0.007 | 0.14 | 0.31 | 0.70 | 2.76 | 6.39 |

Denoising is flat (0.30 s DDIM-25, ~6.2 s DDPM-500). At 1024 points DDIM-25 is 14× faster than our
re-timed DDPM-500 (the stored paper baselines: GBN 8.1 s, BNOT 13.6 s). From ~2300 points the
exact Smart Init dominates. **Use this run for the runtime graph.**

### D1 fixed and resubmitted

- Job 23962618 had failed at import: `tone_results_stage_7_fixed_point.py` imports
  `budget_teachers_driver.gbn_cmd`, which moved to `archive/E4_budget_generalization_scripts/` when
  E4 was archived. Fix: stage 7 now adds that folder to `sys.path` (2 lines, backup
  `.bak_20261110`); the GBN command builder is unchanged. Import verified.
- **Job 24060328**: D1 re-run with the 11-07 settings (`tone_d1_d3.sbatch`, RUN_D3=0, alpha 0.5,
  6 iterations, per-image GBN timeout) → `tone_results/fixed_point/summary.md`.

### Next

1. Read D1 (job 24060328) when it finishes.
2. Smart Init speed: decide whether large-budget runtime matters enough to quality-test
   `bisection-blocks16` (or a hybrid: exact up to ~2300 points, fast above, where budgets are
   visual only). At the training budgets keep `pot-exact`.
3. E13 (all 4 datasets with ddim-25-eta1) is still open.

### 2026-11-10: bisection-blocks16 quality check — passes (no significant change)

- `bisection-blocks16` = bisection + exact assignment inside 16 x 16 cell tiles, 4 sweeps (the
  entry already in `fast_smart_init.ASSIGNERS`, the one the benchmark measured). Registered as an
  `--ot_solver` value in `control_v4/ot_assignment.py` (backups `ot_assignment.py.bak_20261110b`,
  `fast_smart_init.py.bak_20261110b`). Check on random 1024 points: valid permutation, cost 1.0001x
  exact.
- Jobs 24060378 (sampling, 200 Icons images, ddpm-500 + ddim-25-eta1) / 24060379 (metrics) /
  24060380 (summary) → `outputs/fast_ot_quality-Icons-50/summary_bisection-blocks16.md`.
- Change vs pot-exact, ddim-25-eta1 / ddpm-500, **none significant** (paired Wilcoxon, 200 images):
  δc +4.1% / +3.5%, centroid +5.0% / +5.9%, HOT +3.2% / +2.4%, EMD +1.1% / +1.5%, CVT, cell CV,
  Sinkhorn, ρs within 0.2%. Smart Init 0.10 s vs 0.15 s at 1024.
- Reading: it meets the criterion (not significantly worse), but every difference leans slightly
  worse, and 200 images may not detect a ~4% capacity change. At 1024 it saves only 0.05 s.
  Its value is at large budgets: benchmark 1.97 s at 6400 points (cost 1.018x exact) vs 6.39 s for
  the exact Smart Init in the clean runtime run, i.e. DDIM-25 total ~6.7 s → ~2.3 s (not re-timed).
- Suggested use: keep `pot-exact` at the training budgets (1024 / 1600: all tables), and use
  blocks16 only above ~2304 points, where budgets are visual only. Before that: Fig. 4 images with
  blocks16 (`fast_ot_visual.sbatch`, SOLVER=bisection-blocks16) and a clean re-time.

### 2026-11-10: Fig. 4 with bisection-blocks16, D1 result, clean re-time queued

- Job 24061456 (`fast_ot_visual.sbatch`, SOLVER=bisection-blocks16, ddim-25-eta1) →
  `outputs/grid_size_results/pointcount_generalization_bisection-blocks16_ddim-25-eta1.pdf`
  (Target + 576 / 1024 / 2304 / 4096 columns; 6400 is sampled in `fast_ot_bisection-blocks16/CN-GBN_6400`
  but not a column of the figure). Visual compare against `..._exactOT_ddim-25-eta1.pdf`: no visible
  difference at any budget (monkey: same edges, eyes, tail; stress test: same blob shapes, a few stray
  points at 2304 / 4096 in both). blocks16 is fine for the visual-only budgets.
- Clean re-time: `runtime_clean.sbatch` now takes `OT_SOLVER` (default pot-exact; backup
  `runtime_clean.sbatch.bak_20261110`). Hybrid run folder `outputs/runtime_clean_blocks16-Icons-50`:
  256-2304 symlinked from the exact-OT clean run, 3136-6400 re-timed with blocks16 in job 24061457
  (exclusive node, pending), summary + plot in job 24061458 (afterok) → `summary.md` there.
- D1 (job 24060328, completed 1:03 h) → `outputs/tone_results/fixed_point/summary.md`. Gradient-free
  multiplicative fixed point (6 updates, 7-8 sampler calls / image) on the fresh run: GBN +0.24 dB
  (19/24), ddpm-500 +0.15 dB, ddim-25-eta1 +0.18 dB. Our DDIM-25-gradient rho* in one call: +0.36 /
  +0.37-0.38 / +0.36-0.39 dB. So the gradient correction beats the in-loop fixed point at ~1/8 the
  sampler calls, also for GBN (curve +0.36 dB vs +0.24 dB fresh). Possible answer to "why not just
  iterate the sampler": cheaper and better.

## 2026-11-15: mock review R1, gap checks

The user received an example review (`reviews/mock_review_R1.md`, 4/10). Point-by-point answers and
the evidence they rest on: **REBUTTAL_DRAFT.md section 5** (new evidence A14–A17).

### Facts established today

- **Student vs GBN, paired** (1000 Icons images, current metric code): Ours-GBN HOT −22% (better on
  64% of images, p = 8e-29), centroid −21%, EMD −13%; from pure noise (no Smart Init) still HOT −15%.
  Not the KDE loss (E8 null). On the 50-image ablation subset the HOT gap to GBN is not significant.
- **Test split**: the Icons-50 test set is exactly the trainer's held-out 10% (reproduced the
  seed-42 randperm over the sorted file list; zero overlap with the 9000 training icons). But 997/1000
  test icons share their emoji concept (another vendor / version) with a training icon.
- **EMD (M3) is W2**: `compute_m3_emd` takes the sqrt of the OT cost with squared distances.
- **Aspect ratio is not a network input**: `load_condition` resizes the condition to the grid and
  points are normalized per axis (E3 already shows it is not the cause of the capacity gap).
- **Anisotropy of the stored paper-sampler spectra** (mean dB, peak band 0.75–1.5): GBN −9.6,
  Ours-GBN −8.4, WVS −14.2, Ours-WVS −9.7, BNOT −15.2. Ours-WVS is clearly more anisotropic than WVS.
- **Baseline implementations** (Q1): WVS = Rougier's NumPy/SciPy code (CPU); BNOT = the authors' C++
  CLI (CPU); GBN = the authors' CUDA binary `gbn-adaptive-linux` (GPU, RTX 4090 / 6000).

### Jobs submitted 2026-11-15 (submit script: `logs/submit_gaps_20261115.sh`)

| Job(s) | What |
|---|---|
| 24174333 | `near_duplicate_check.sbatch` (W6): per test icon, max Pearson correlation (128x128) with a same-concept training icon vs the learned/GBN metric ratio (Spearman, quartiles) → `outputs/near_duplicate_check/summary.md` |
| 24174334 | `train_ablation_nogate.sbatch` (W5): the "full" run without gated injection (`--no-enable-adaptive-gate-injection`, everything else default = full) → `control_v4/train_outputs_Icons-50_1024_GBN_noGate`. Resumable. Evaluate like E8 when done (the `ablation_nokde_stage_1.py` wrapper hard-codes gating on: add a model entry with `ENABLE_ADAPTIVE_GATE_INJECTION = False`). |
| 24174335 | `spectral_ddim25.sbatch` (E15 / W7): Ours-GBN and Ours-WVS with ddim-25-eta1 on the spectral sources → `outputs/spectral_analysis_ddim25/`, spectra in `spectral_analysis/target_CN-{GBN,WVS}-DDIM25_1024_spectral`, table `spectral_analysis/anisotropy_summary.md` (`spectral_anisotropy_summary.py`) |
| 24174336 (array 0-11) | **E13** `e13_ddim25.sbatch`: ddim-25-eta1 (exact OT) for all 12 Table 1 checkpoints → `outputs/e13_ddim25-<dataset>/CN-<sampler>_<N>/`; scorers 24174337–48 (one `few_step_sampling_metrics.sbatch` array per checkpoint, afterok on its task). Then build the 6-column table against `rescore_current_metrics-<dataset>` (same metric code). |

### Results already in (2026-11-15)

- **Near-duplicates (24174333):** no memorization effect. Look-alike score median 0.74; learned/GBN
  ratio vs look-alike score: |Spearman| <= 0.14 for every metric; HOT ratio 1.06 (lowest quartile)
  vs 1.18 (highest), i.e. slightly worse on close look-alikes; δc 1.94 vs 1.78; EMD flat. → A16.
- **E15 (24174335):** ddim-25-eta1 spectra. Anisotropy around the peak: Ours-GBN −7.8 dB (DDPM-500
  −8.4), Ours-WVS −8.5 dB (−9.7); low-frequency power lower (bluer): Ours-GBN 0.0035 (0.0057),
  Ours-WVS 0.0097 (0.0114), WVS 0.0209. Table: `spectral_analysis/anisotropy_summary.md`. → A17.

### E13 done (2026-11-16)

All 12 sampled and scored (Icons 1000 / CelebA 500 / ShapeNet2D 400 / Chibi 263 images per
checkpoint). `experiments/e13_summary.py` → `outputs/e13_ddim25_summary.md`: six-method table per
dataset (samplers from `rescore_current_metrics-<dataset>`, same metric code) and the paired change
of each learned variant vs its paper point sets. Capacity −1.0% to −9.2% and ρs +1.1% to +2.3%,
significant in all 12; 1 of 84 cells significantly worse (ShapeNet2D Ours-WVS CVT +0.1%). → A18.

### Next

1. noGate training (24174334): epoch 1625/5000 on 2026-11-16 at ~46 s/epoch, i.e. done ~2026-11-18.
2. When the noGate training finishes: Table 2-protocol evaluation (paired, seeded, vs `sdedit_seeded`).
3. Still open: E14 (seeds), Fig. 3 strip percentages (one realization or an average?).
