# Rebuttal work: context for the next agent

Last updated: 2026-11-18. Keep this file current: update it whenever you submit jobs, get
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
- The user's latest paper files are on the server for later edits: `paper_rebuttal/main_EG_rebuttal_11.10.tex`,
  `paper_rebuttal/supplemental_EG_rebuttal_11.10.tex` and `paper_rebuttal/tables_rebuttal/`. **Do not edit them until
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

## 2026-11-17: status check, and work from 2026-11-16 that was not recorded above

- **noGate training (24174334)**: running, epoch 4509/5000 on 2026-11-17 11:30 (checkpoint
  `dynamic_ep4500.ckpt` saved), ~46 s/epoch, i.e. done the evening of 2026-11-17. Nothing else is queued.
- **blocks16 clean re-time (24061457 / 24061458) is done**, not pending as the 2026-11-10 section says:
  `outputs/runtime_clean_blocks16-Icons-50/summary.md`. DDIM-25 total 0.74 / 0.94 / 1.13 / 1.38 s at
  3136 / 4096 / 5184 / 6400 points (Smart Init 0.44 / 0.63 / 0.82 / 1.07 s), 3 reps, spread <= 2%.
  Already used in REBUTTAL_DRAFT.md A1. Two things to know: the summary's title still says "exact OT"
  (true only up to 2304), and the hybrid curve is not monotone (exact OT at 2304 = 0.99 s, blocks16 at
  3136 = 0.74 s).
- **Generalization diagnostics (2026-11-16 02:05–02:53; added to REBUTTAL_DRAFT.md as A19 on 2026-11-17)**:
  `diag_generalization_stage_0.py`, `diag_generalization.sbatch` (array 0-4),
  `diag_generalization_timestep_loss.py`, `diag_generalization_summary.py` →
  `outputs/diag_generalization/summary.md` and `timestep_loss.md` (Ours-GBN, Icons-50, 200 training +
  200 held-out images; scorer arrays 24212916 / 24212917). It needs the new `--init_dir` option of
  `few_step_sampling_stage_1.py` (start from given offset grids, e.g. the teacher's exemplar), which
  is **uncommitted** on `rebuttal`, as are the four diag_generalization files.
  What the tables show:
  - Denoising error is far larger on held-out images: eps MSE held-out / train = 1.3 at t = 2, 5.8 at
    t = 50, 20–24 at t = 150–500; x0 RMS error at t = 300: 0.068 cells (train) vs 0.338 (held-out).
  - Yet the quality gap to the teacher is nearly the same on training images. Normal sampling,
    learned / exemplar capacity ratio: ddpm-500 1.88 (train) vs 2.03 (held-out), p = 0.27;
    ddim-25-eta1 1.61 vs 1.83, p = 0.02. CVT ratio 1.03 vs 1.04. HOT and centroid ratios are not
    significantly different.
  - Starting from the teacher's own exemplar noised to step K: capacity ratio is already 1.37 / 1.38
    (train / held-out) at K = 10 and 1.78 / 1.96 at K = 500. So the last few low-noise steps alone
    lose most of the capacity quality, on training images too (eps MSE at t <= 20 is 0.3–0.8 on
    training images, where the Min-SNR weight is 0.002–0.02).
  - Reading (not verified further): the capacity gap is mainly a low-noise denoising limit of the
    model, not a train/test generalization gap; the held-out images add a smaller part on top.
- Other jobs in sacct since 2026-11-10 that failed or ran out of memory (pocfm-dti-deg, ml-step0,
  hrs-a, conv-gbn / conv-wvs / conv-bnot) are not described in this file.

### 2026-11-17: noGate (W5) evaluation submitted

Table 2 protocol as in E8: "Full + SDEdit" settings, the same 50 validation images, the same
per-image noise, compared paired against `sdedit_seeded` (the paper's full checkpoint, already
exported and scored for E8; the metric code has not changed since 2026-10-19). Output goes next to
E8: `outputs/ablation_nokde_e500_b50_1024/sdedit_noGate/` and `summary_noGate.md`.

- Scripts: `ablation_nokde_stage_1.py` has a new `--model nogate` (the only entry with gated
  injection off; `nokde` and `reference` behave as before), `ablation_nokde_stage_3.py` takes
  `--variant / --label / --title / --summary` (defaults = the E8 behaviour). Backups
  `*.bak_20261117`. New `ablation_nogate_export.sbatch` (CKPTS = checkpoint-name substrings, default
  all) and `ablation_nogate_metrics.sbatch` (array task i = epoch (i + 1) * 500). All uncommitted.
- Jobs (setup script kept at `logs/nogate_setup_20261117.sh`):

| Job | What |
|---|---|
| 24257553 | `nogate-export-a`: epoch 500 only. It was meant for 500–4500, but `sbatch --export=ALL,CKPTS=a,b,...` splits on the commas, so only the first name arrived |
| 24257658 | `nogate-export-a2`: epochs 1000–4500 (CKPTS passed through the environment instead) |
| 24257659 (array 0-8) | `nogate-metrics-a`: scores epochs 500–4500, afterok 24257553 and 24257658 (replaces the cancelled 24257554) |
| 24257555 | `nogate-export-b`: epoch 5000, **afterok the training job 24174334**; it stays pending forever if the training job fails (then resubmit the training script, it resumes, and resubmit this with `CKPTS=dynamic_ep5000`) |
| 24257556 (array 9) | `nogate-metrics-b`: scores epoch 5000, afterok 24257555 |
| 24257660 (replaces the cancelled 24257557) | `nogate-summary`: `ablation_nokde_stage_3.py --variant sdedit_noGate --label noGate --title "W5: gated-injection ablation" --summary summary_noGate.md`, afterok both scorers |

- When `summary_noGate.md` exists: record the paired result at epoch 5000 here, fill the "W5 gating
  without the rest" row and the "Gating" status row of REBUTTAL_DRAFT.md (now "pending"), and add an
  evidence entry (A20). E8 showed that 50 images detect only large differences (δc −12% was
  p = 0.64), so a null result here means "no measurable effect on 50 images", not "no effect".

## 2026-11-18: noGate result, a fix to the paired summaries, commit

- **noGate (W5) is done**: training 24174334 finished 2026-11-17 17:23, all export / scorer / summary
  jobs completed. Result at epoch 5000, 50 paired images (→ REBUTTAL_DRAFT.md A20): without gated
  injection the model is worse on CVT (+0.8% median, 78% of images, p = 2e-5), capacity δc (+7.7%
  median, 82%, p = 5e-5) and ρs (−0.7%, 76%, p = 4e-6); HOT, centroid, EMD, Sinkhorn, power-cell CV
  show no difference. Gating is supported, with a modest effect.
- **Trap found and fixed: mean change vs paired test.** The first `summary_noGate.md` said
  "δc −4.1% (better), p = 4.65e-05". The mean is lower because of a few outlier images, but the
  Wilcoxon test is paired and significant in the other direction (worse on 82% of the images).
  `ablation_nokde_stage_3.py` (backup `.bak_20261118`) now prints, per metric: both means, the change
  of the mean, the median per-image change, the share of images where the variant is worse, p, and a
  verdict taken from the paired columns. Both summaries were regenerated (job 24280672); the E8
  verdicts did not change (all "no difference").
- **Same check on the headline results** (E1 and the 12 E13 comparisons, 6 metrics each): mean and
  paired median agree on every significant cell except ShapeNet2D Ours-WVS CVT in E13 (mean +0.1%,
  median −0.4%, p < 0.05), which is therefore an improvement. So E13 has **no** significantly worse
  cell (A18 corrected). `e13_summary.py` (backup `.bak_20261118`) now prints "mean / median" per cell.
- **Rule for every paired table from now on:** when a p-value comes from a paired rank test, state
  the direction from the median per-image change or the share of images, never from the means alone.
- Committed and pushed on `rebuttal` (see `git log`): the generalization diagnostics
  (`diag_generalization_*`, `--init_dir` in `few_step_sampling_stage_1.py`), the noGate evaluation
  scripts, the two summary fixes, and these notes.

### What the diagnostics say about the model (for work after the rebuttal)

The capacity / CVT gap is not a data gap (same gap on training images, A19) and it arises in the
last low-noise steps (10 DDPM steps from the teacher's exemplar already give δc 1.37×). Training
uses Min-SNR with γ = 5, which gives those steps a weight of 0.002–0.02. Not tested yet: a short
fine-tune from `dynamic_ep5000.ckpt` with more weight on t < ~100 (larger γ, or no cap there),
evaluated with the Table 2 protocol and the from-exemplar test. Also free and untested: 8× flip /
rotation augmentation (the trainer has none), and stopping at ~2500 epochs (the validation metrics
are flat from there on).

## 2026-11-18: low-noise fine-tune started (model study, not a rebuttal item)

Tests the reading of A19: the capacity / CVT gap arises in the last low-noise steps, which the
Min-SNR weighting (γ = 5) nearly ignores. Three 300-epoch fine-tunes of the paper's full Ours-GBN
checkpoint (epoch 5000 copied into a new run folder and resumed, optimizer state included; all
other settings = the full run):

| Task | Run folder | Weighting |
|---|---|---|
| 24280693_0 | `control_v4/train_outputs_Icons-50_1024_GBN_ft-g5` | γ = 5, **control** (more training only) |
| 24280693_1 | `..._ft-g50` | γ = 50: low-noise weight ×10 (t = 10: 0.008 → 0.08; t = 50: 0.09 → 0.88) |
| 24280693_2 | `..._ft-g0` | no Min-SNR: plain eps loss, every step weight 1 |

`ft_lownoise_train.sbatch` (array 0-2; checkpoints at 5100 / 5200 / 5300). All three resumed at
start_epoch = 5000 (checked in the logs).

Evaluation, chained (`logs/submit_ft_lownoise_20261118.sh`): `ft_lownoise_eval.sbatch` (array
24280694, aftercorr the training tasks) samples the 200 held-out diagnostic images
(`outputs/diag_generalization/test_*`) with the epoch-5300 checkpoint, same Smart Init and noise as
the baseline runs `CN-GBN-test_1024` / `CN-GBN-test-fromGT_1024`: normal sampling (ddpm-500,
ddim-25-eta1), start from the teacher's exemplar at K = 10 … 500, and the denoising error per noise
level (`timestep_loss_ft-<tag>.md`; `diag_generalization_timestep_loss.py` got `--name`). Scorers
24280695–700, then 24280701 runs `ft_lownoise_summary.py` → `outputs/diag_generalization/
ft_lownoise_summary.md` (paired per image: ratio to the teacher, median per-image change, share of
images worse, p, paired verdict).

How to read it: compare g50 / g0 against **g5**, not only against the paper checkpoint. If g5 moves
as much as the others, the change is from extra training, not from the weighting. If capacity
improves at low K in the from-exemplar test but normal sampling gets worse on HOT / centroid, the
low-noise emphasis is costing the coarse steps (then try γ = 50 with more epochs, or a t-sampling
biased to low t, instead of removing Min-SNR).

### 2026-11-18: low-noise fine-tune result — the reweighting works

All jobs completed (training ~3.5 h each). `outputs/diag_generalization/ft_lownoise_summary.md`,
200 held-out images, paired per image against the paper checkpoint (median per-image change; share
of images where the fine-tuned model is better):

| variant | sampler | capacity δc | ρs | CVT | HOT / centroid / W2 |
|---|---|---|---|---|---|
| γ = 5 (control) | DDPM-500 | +0.7%, no difference (p = 0.69) | +0.1% | +0.3% (worse on 64%) | −1.8% / −4.7% / −1.1%, better |
| γ = 50 | DDPM-500 | **−13.2%** (better on 92%) | +0.7% (86%) | no difference | no difference (W2 +0.1%) |
| γ = 50 | DDIM-25 | −9.7% (84%) | +0.3% | +0.1% (worse) | +1.3% / +2.4% / +0.5%, worse |
| **no Min-SNR** | DDPM-500 | **−12.8%** (92%) | **+1.5%** (95%) | **−0.6%** (82%) | no difference |
| **no Min-SNR** | DDIM-25 | −7.2% (76%) | +0.6% (76%) | −0.3% (70%) | no difference |

- **The gain comes from the weighting, not from more training:** the control (γ = 5, same 300
  epochs) does not change capacity at all; it only improves the transport metrics a little.
- **Removing Min-SNR is the best variant:** capacity, ρs and CVT all improve on 82–95% of the images
  with the paper sampler, and nothing gets worse. γ = 50 gets the same capacity gain but a smaller ρs
  gain, no CVT gain, and slightly worse transport metrics with DDIM-25.
- **It confirms where the loss was:** from the teacher's exemplar at K = 10, capacity goes from 1.28×
  to 1.17× the teacher's (no Min-SNR), at K = 500 from 1.75× to 1.53×; x0 RMS error at t ≤ 20 drops by
  10–17% on held-out images. The coarse steps (t ≥ 300) are unchanged on held-out images and
  slightly worse on training images.
- **The gap is reduced, not closed:** capacity to the teacher 1.78× → 1.57× (DDPM-500), 1.63× → 1.50×
  (DDIM-25); CVT 1.028× → 1.021×; ρs 0.962 → 0.977. This is after only 300 epochs on a checkpoint
  trained 5000 epochs with the old weighting.
- Checkpoints: `control_v4/train_outputs_Icons-50_1024_GBN_ft-{g5,g50,g0}/checkpoints/dynamic_ep{5100,5200,5300}.ckpt`.
- Not done: the trend over 5100 / 5200 / 5300 (is it still improving?), a longer fine-tune, training
  from scratch without Min-SNR, the other 11 checkpoints. For the rebuttal this is at most one
  sentence ("the capacity gap comes from the loss weighting of the last denoising steps; a
  300-epoch fine-tune without it lowers δc by 13% on 92% of held-out images with no metric worse"),
  since every table is from the paper checkpoints.

### 2026-11-18: no-Min-SNR fine-tune continued to epoch 6000, trend evaluation

Is the gain still growing? (`logs/submit_ft_lownoise_trend_20261118.sh`)

| Job(s) | What |
|---|---|
| 24292272 / 24292275 | evaluate the saved ft-g0 checkpoints at epochs 5100 and 5200 (`ft_lownoise_eval.sbatch --array=2`, `EPOCH=<e>`; run folders `CN-GBN-test-ft-g0-ep<e>[-fromGT]_1024`, `timestep_loss_ft-g0-ep<e>`); scorers 24292273–74, 24292276–77 |
| 24292278 | early trend table (`ft_lownoise_trend.py` → `outputs/diag_generalization/ft_lownoise_trend_g0.md`), epochs 5100–5300 |
| 24292279_2 | **ft-g0 continued to epoch 6000** (`ft_lownoise_train.sbatch --array=2`, `EPOCHS=6000`); resumed at start_epoch = 5300 (checked); ~8 h; checkpoints every 100 epochs |
| 24292280 / 24292283 | evaluate epochs 5600 and 6000 (afterok the training); scorers 24292281–82, 24292284–85 |
| 24292286 | final trend table (same file, epochs 5100–6000) |

`ft_lownoise_eval.sbatch` now names its outputs by epoch (5300 keeps the plain `ft-<tag>` name).
The control (γ = 5) is not continued: in the first 300 epochs it left capacity unchanged.
Read the trend from the paired columns; if capacity and ρs are flat from 5300 on, the fine-tune has
converged and the remaining gap needs something else (training from scratch without Min-SNR, or a
t-sampling biased to low noise).

## TODO (user, 2026-11-18): the BNOT runtime numbers do not make sense — investigate

> **RESOLVED 2026-11-18 (evening). Full write-up: `/groups/asharf_group/ofirgila/BNOT_V2/CONTEXT.md`.**
> Cause: our BNOT port (`BNOT_new/ibnot_cli`) had replaced the original's SuiteSparse QR by Eigen's
> `SparseQR`, which behaves like a dense factorization on the weight Laplacian (6.7 s per solve at
> 3136 sites, 260 s at 10,000). Fixed in BNOT_new (commit 7ba8a7c, branch `scripts-update`):
> SuiteSparse QR with the same solution the Eigen path returned. **The point sets are unchanged**
> (the 10 runtime images at 256–6400 points reproduce the stored results to < 1e-5 of a site
> spacing), so all BNOT exemplars and BNOT quality numbers stay valid. Only the runtimes were wrong.
>
> **BNOT on the 10 pinned runtime images, re-run with the fixed binary** (AMD EPYC 7702P, 1 thread;
> `summary.md` and `runtimes_avg_bnot_fixed.json` in the re-run's folder; the stored
> `runtimes_avg.json` is being replaced, see the last section of this file):
>
> | points | 256 | 576 | 1024 | 1600 | 2304 | 3136 | 4096 | 5184 | 6400 |
> |---|---|---|---|---|---|---|---|---|---|
> | stored BNOT (s) | 7.2 | 8.8 | 13.6 | 25.5 | 59.2 | 333 | 393 | 872 | 1802 |
> | fixed BNOT (s) | 5.2 | 5.8 | 7.1 | 8.5 | 9.5 | 20.9 | 12.9 | 15.3 | 17.2 |
>
> 100K points on the BNOT paper's `memorial.png`: ~2 min with the fixed binary (paper: 498 s in 2012).
>
> **Consequences, not applied yet (user decisions):**
> - Fig. 7's BNOT curve and every sentence built on it are wrong: "1802 s at 6400 points", "BNOT's
>   cost grows by more than 100x", "13.6 s per image" (exemplar cost ≈ 38 h for Icons → about 20 h).
>   BNOT is 5–17 s from 256 to 6400 points; our DDIM-25 sampler (0.31–1.38 s) is still faster at
>   every budget, by about 10–28x instead of up to 1300x (different machines, so indicative).
> - REBUTTAL_DRAFT.md entries to revise with these numbers: A1, A3, A11, watch-out 6, and the
>   W1 / W2 / Q1 rows of section 5. The argument "BNOT exemplars are affordable" gets stronger; the
>   argument "BNOT does not scale" is gone.
> - If one machine for all curves matters, re-time BNOT on the node used for our clean run.

**Problem.** The BNOT paper (de Goes et al. 2012, "Blue Noise through Optimal Transport") reports
stippling **100K points in 498 s** (per the user; confirm the exact figure, density and machine in
the paper). Our stored measurement passes that time with far fewer points:

| points | 256 | 576 | 1024 | 1600 | 2304 | 3136 | 4096 | 5184 | 6400 |
|---|---|---|---|---|---|---|---|---|---|
| BNOT, stored (s) | 7.2 | 8.8 | 13.6 | 25.5 | 59.2 | 333 | 393 | 872 | 1802 |

(`outputs/icons_results_runtimes/runtimes_avg.json`, 10 pinned Icons-50 images, driver
`/groups/asharf_group/ofirgila/BNOT_new/run_bnot_data_gen_v2_all.py` → `bnot_data_gen.py`, the
authors' C++ CLI; parameters in `archive/E4_budget_generalization_scripts/budget_teachers_driver.py`:
max_iters 25, max_newton_iters 50, epsilon 1.0, seed 7, invert, 512 x 512 input.) 1802 s for 6400
points is ~3.6x the paper's time for ~16x fewer points, so something in our setup is off.

**Until this is resolved, do not lean on BNOT's large-budget runtime** (the 333–1802 s values, "more
than 100x growth", "1802 s at 6400") in the rebuttal or the revision. This overrides the earlier
"baseline runtimes are final" rule for BNOT only; WVS and GBN stay as stored.

**Things to check** (none done yet):
1. The jump between 2304 and 3136 points: 59 s → 333 s (5.6x for 1.36x the points). It looks like a
   regime change, not scaling: iteration caps being hit, a fallback path, or memory / swap.
   Per-image timings and the solver's own iteration logs (`--keep_stats`) for 2304 vs 3136.
2. Convergence settings. `outputs/spectral_analysis/FINDINGS.md` found that `epsilon` scales the
   exit thresholds (at epsilon 1.0 the coarse loop stopped after 2 iterations on a uniform image).
   Which thresholds end the run on the icon images, and are the Newton iterations (max 50) all used?
3. The density. The paper's timings are probably on smooth densities; our icons have large
   zero-ink (white) regions after inversion, which can make the power-diagram / Newton solve
   ill-conditioned. Time a constant and a smooth-ramp image at 1024 / 4096 / 6400 points.
4. Fixed overhead: 7.2 s at 256 points suggests ~7 s of startup, image loading or I/O per image
   inside `total_wall_seconds`. Separate solver time from wrapper time.
5. The build: optimization flags (-O2 / -O3 vs a debug build), CGAL version and kernel, single
   thread. Compare with the paper's stated machine.
6. How the timing was taken: node, concurrency (were several BNOT processes or shards running on
   the same node?), and which clock.
7. What the paper's 498 s covers (which density, which stopping criterion, multi-scale init or not),
   so the comparison is like for like.

**Where the BNOT runtime is used and must be re-checked afterwards:** paper Fig. 7 and its text;
REBUTTAL_DRAFT.md A1 (13.56 / 25.51 s and 1802 s), A3 ("BNOT: 13.6 s at 1024 points, 1802 s at
6400"), A11 (exemplar cost ≈ 38 h for Icons from 13.6 s per image), the W1 / W2 / Q1 rows of
section 5; this file's runtime sections. The 1024 / 1600-point values (13.6 / 25.5 s) are less
suspicious than the large budgets but include the possible fixed overhead (item 4).

## 2026-11-18 (evening): remaining rebuttal items

- **Open-items list: REBUTTAL_DRAFT.md section 6** (what is running, the decisions that are the
  user's, the writing that can be prepared, the answers that stay weak).
- **E14 + Fig. 3 strips + parameter counts submitted** (`logs/submit_seeds_20261118.sh`,
  `rebuttal_seeds.sbatch` array 24303895): tasks 0 / 1 = Ours-GBN on the 1000 Icons test images with
  Smart Init seeds 43 / 44 and noise seeds 100000 / 200000 (seed 0 = the E1 run), ddpm-500 +
  ddim-25-eta1 → `outputs/e14_seeds-Icons-50_1024/CN-GBN-seed<k>_1024`; scorers 24303896 / 97;
  `e14_summary.py` (24303898) → `summary.md` there. Task 2 = the Fig. 3 quadratic image, 25
  realizations of Ours-GBN and Ours-WVS → `outputs/fig3_strip_realizations/summary.md`
  (`fig3_strip_realizations.py`), then `model_stats.py` → `outputs/model_stats.md`.
- **Fig. 3 fact:** the strip percentages in the paper are from one stored point set per method
  (`experiments/compare_advance_metrics_stage_1.py`, `images_results_metrics/quadratic/target_*`).
- **Fine-tune early trend** (`outputs/diag_generalization/ft_lownoise_trend_g0.md`, job 24292278):
  with no Min-SNR the gain is complete by epoch 5200: capacity −5.5% (5100), −15.0% (5200), −12.8%
  (5300) with DDPM-500; ρs +1.5% from epoch 5100 on; CVT −0.4% → −0.6%. The run to 6000
  (24292279_2) will show whether anything is left; expect a plateau.
- The `bnot-time` jobs in the queue belong to the other session that investigates the BNOT runtime.
- **Rebuttal text, draft 1: `experiments/REBUTTAL_TEXT.md`** (2026-11-18). Part 1 = the one-page
  text ordered for the mock review; Part 2 = short answers to use only if asked; Part 3 = what must
  be settled before sending (sampler wording, title, BNOT runtime, the pending E14 / Fig. 3 /
  parameter-count numbers). Every number in it is a FINAL entry of REBUTTAL_DRAFT.md; no BNOT time
  and no budget other than the training budget is quantified.

### 2026-11-18: sampler decision confirmed; supplement section for the few-step sampler

- **Decision (user):** the 25-step sampler goes into the **supplement** as a boost found over the
  original idea: a faster sampling variant of the same trained weights. Call it a sampler, not a new
  model (no retraining). The main paper gets its curve in the runtime figure and a short paragraph;
  every main table keeps DDPM-500. REBUTTAL_TEXT.md and REBUTTAL_DRAFT.md are written this way.
- **Supplement section, what exists:** Icons-50 comparison on 1000 images (A1), step sweep (A1b),
  four-dataset table for all 12 checkpoints (A18, `outputs/e13_ddim25_summary.md`), deterministic
  solvers fail (A6), clean runtime (`runtime_clean_blocks16-Icons-50`), spectra and anisotropy
  (A17), Fig. 4 with the sampler (`grid_size_results/pointcount_generalization_*_ddim-25-eta1`),
  seed robustness (E14, running). Report with it that the sampler is ~1 dB more anisotropic around
  the blue-noise peak, although bluer at low frequency.
- **Ablation table with the sampler, submitted** (`logs/submit_ablation_ddim25_20261118.sh`):
  `ablation_ddim25.sbatch` (array 24304064) samples the Table 2 variants at epoch 5000 (vanilla,
  unfrozen, gecco, agi, full) plus noGate and noKDE with ddim-25-eta1 and ddpm-500 from the SDEdit
  start, on the 50 Table 2 images, same Smart Init and noise for every variant →
  `outputs/ablation_ddim25_e5000_b50_1024/CN-<variant>_1024`; scorers 24304065–71;
  `ablation_ddim25_summary.py` (24304072) → `summary.md` (means per variant, and each variant
  paired against full). Difference from the paper's Table 2: there the architecture rows are sampled
  from pure noise with 1000 DDPM steps and only the last row uses SDEdit; here every row uses SDEdit.
- `few_step_sampling_stage_1.py` got `--base_ckpt_path`, `--[no-]gecco`, `--[no-]gate-injection`
  (defaults = the paper's full model), needed to load the frozen-base and partial variants.

## 2026-11-18 (night): BNOT runtimes replaced, old ones archived — SUBMITTED, NOT CHECKED YET

The user asked to update `runtimes_avg.json` and the runtime graphs with the fixed BNOT times and to
archive the previous ones, and will check the results later. Nothing below was verified after
submission; **check the job logs before using the new JSON or graphs.**

| Job | What |
|---|---|
| 24307078 | BNOT with the fixed binary at the four larger budgets (7744, 9216, 10816, 12544 points), so the new JSON keeps all 13 budgets. Same 10 images and settings, into `outputs/icons_results_runtimes_bnot_fixed/` (256–6400 were done in job 24306496). `BNOT_V2/timing/fix/ten_images.sbatch` with `BUDGETS=...` |
| 24307095 (afterok 24307078) | `BNOT_V2/timing/fix/archive_and_rebuild.sbatch`: archive, swap, rebuild (below). It stops without changing anything if any budget has fewer than 10 timing files |

What job 24307095 does, in order:
1. **Archive** to `outputs/a_archive/bnot_runtimes_eigen_solver_20261118/` (with a README): the old
   per-method folder `icons_results_runtimes_bnot/` (moved); the old `runtimes_avg.json`, the old
   Fig. 7 plots and the BNOT `timestamps_BNOT_*` folders of `icons_results_runtimes/`; copies of
   `runtimes_avg_clean.json`, `summary.md` and `plots/` of `runtime_clean-Icons-50/` and
   `runtime_clean_blocks16-Icons-50/`; and `icons_results_runtimes_bnot_spqr_minnorm_partial/`.
2. **Swap**: `icons_results_runtimes_bnot_fixed/` becomes `icons_results_runtimes_bnot/`.
3. **Rebuild**: `merge_runtime_results_stage_1.py --sources icons_results_runtimes_bnot`, then
   stage 2 (`icons_results_runtimes/runtimes_avg.json`) and stage 3 (Fig. 7:
   `icons_results_runtimes/plots/runtime_scaling_by_points.{png,pdf}`), then
   `runtime_clean_stage_3.py` for `runtime_clean-Icons-50` and for
   `runtime_clean_blocks16-Icons-50` (the new-variant graphs).
4. **Check**, printed at the end of the log: CN / GBN / WVS entries of the JSON unchanged, BNOT has
   13 budgets with 10 images each, and a previous-vs-new BNOT table.

**How to check it** (logs in `/groups/asharf_group/ofirgila/BNOT_V2/timing/fix/logs/`):
- `ten-images-24307078.out` ends with `All budgets done. failed=[]` and has no `Skipped` lines.
- `archive-24307095.out` shows `CN unchanged: True`, `GBN unchanged: True`, `WVS unchanged: True`,
  BNOT counts `[10]`, and new BNOT times of about 5–17 s up to 6400 points (measured in job
  24306496: 5.2 / 5.8 / 7.1 / 8.5 / 9.5 / 20.9 / 12.9 / 15.3 / 17.2 s at 256 … 6400).
- If job 24307078 fails, 24307095 never runs and nothing is replaced; rerun the same two sbatch
  commands from `BNOT_V2/timing/fix/`.
- Look at the three regenerated graphs: the BNOT curve should be nearly flat (5–17 s up to 6400
  points) instead of climbing to 1802 s.

**Still to do after the check (not done, on purpose):**
- REBUTTAL_DRAFT.md still quotes the old BNOT times in A1 ("BNOT 13.56 / 25.51 s", "BNOT 1802 s"),
  A3 ("13.6 s at 1024 points, 1802 s at 6400"), A11 ("13.6 s … 25.5 s", exemplar cost 38 / 19 / 28 /
  19 h, "1.7× GBN's"), the draft answer "they cost about 1.7× GBN's", and the W1 / W2 / Q1 rows of
  section 5. Replace with: BNOT 7.1 s at 1024, 8.5 s at 1600, 17.2 s at 6400; exemplar generation
  with the fixed build ≈ 20 h (Icons), 10 h (CelebA), 9 h (ShapeNet2D), 6 h (Chibi), about the same
  as GBN (8.1 s per image). The exemplars themselves were generated with the slow build (38 / 19 /
  28 / 19 h); the point sets are identical. Watch-out 6 of the draft already warns about this.
- REBUTTAL_PLAN.md says BNOT's cost "grows by about two orders of magnitude over the budget range":
  obsolete, it grows about 3x (5 → 17 s).
- The paper: Fig. 7 (copy the regenerated `runtime_scaling_by_points.pdf`) and the sentences built
  on the old BNOT curve.
- The curves come from different machines: BNOT's new times are from an AMD EPYC 7702P node, the
  stored WVS / GBN / CN ones from the original (unrecorded) machine, ours from the clean run.

### 2026-11-18 21:15: BNOT runtime replacement CHECKED (jobs 24307078 / 24307095)

Checked by another session, from the logs, the new JSON and the regenerated graphs:

- `ten-images-24307078.out`: all four budgets `processed=10, skipped=0`; ends with
  `All budgets done. failed=[]`.
- `archive-24307095.out`: `CN unchanged: True`, `GBN unchanged: True`, `WVS unchanged: True`; BNOT has
  13 budgets x 10 images; archive folder `a_archive/bnot_runtimes_eigen_solver_20261118/` exists with
  its README; `runtimes_avg.json` and the three graphs are dated 20:56.
- New BNOT means (s): 5.2 / 5.8 / 7.1 / 8.5 / 9.5 / 20.9 / 12.9 / 15.3 / 17.2 at 256 … 6400, then
  **179.7 / 41.7 / 23.4 / 25.0 at 7744 / 9216 / 10816 / 12544**.
- **The means at 3136, 7744 and 9216 are each carried by one or two slow images**, not by scaling.
  Per-image `native_wall_seconds`: at 3136, `twitter_0_arrow_pointing_rightwards_then_curving_upwards`
  88 s and `emoji-one_3_waving_hand_sign` 39 s, the other eight 7–15 s; at 7744, the same arrow icon
  1191 s and `samsung_4_squared_cjk_unified_ideograph_7533` 470 s, the other eight 10–38 s; at 9216,
  the ideograph 256 s, the other nine 12–34 s. The median over images would be flat (about 10–15 s).
  If Fig. 7 keeps the mean, say so; or plot the median for every method.
- The graphs stop at 6400 points, so only the 3136 bump is visible (BNOT 20.9 s, just above WVS).
- **New consequence for the main paper's curve (DDPM-500, stored 5.65–9 s):** it is no longer
  faster than BNOT at small budgets. BNOT 5.2 s vs ours ~5.7 s at 256 points, equal at 576, and only
  ~1.2x slower than ours at 1024 (7.1 vs 5.8 s). Do not write "faster than every sampler" for the
  DDPM-500 configuration at the training budgets; the claim holds for the 25-step sampler
  (0.31–1.38 s, about 10–28x below BNOT, different machines).
- Still not done: the REBUTTAL_DRAFT.md / REBUTTAL_TEXT.md / REBUTTAL_PLAN.md / paper edits listed
  just above.

### 2026-11-18 21:40: BNOT numbers updated in REBUTTAL_DRAFT.md and REBUTTAL_TEXT.md

Done on the user's request (backups `*.bak_20261118_bnot`; patch script
`logs/patch_bnot_times_20261118.py`; uncommitted):
- REBUTTAL_DRAFT.md: watch-out 6 rewritten (all nine fixed-build times, and three cautions: the
  500-step sampler is not clearly faster than BNOT, the 3136 mean is two slow images, the machines
  differ); A1 (BNOT 7.08 / 8.49 s at 1024 / 1600, 17.2 s at 6400), A3, A11 (exemplar cost ≈ 20 / 10 /
  9 / 6 h, about GBN's; 38 h was the slow build), the "impractical" draft answer, paper-change item 6
  (Fig. 7's BNOT curve), the W1 / W2 / Q1 rows, and the open-items list.
- REBUTTAL_TEXT.md: BNOT 7.1 / 8.5 s added to the Speed paragraph, the exemplar cost to the BNOT
  paragraph, the correction of the submitted BNOT times as a Part 2 (Q1) answer, Part 3 item 3
  marked resolved with one open decision (whether the correction moves to Part 1).
- Not touched: REBUTTAL_PLAN.md ("two orders of magnitude"), the paper files (Fig. 7 and its text),
  and Part 3 item 4 of REBUTTAL_TEXT.md, which still lists E14 / Fig. 3 / parameter count as pending
  although their summaries exist.

### 2026-11-18 21:50: BNOT default parameters, re-timing on the paper's CPU, convergence test running

**Original BNOT defaults** (the user's copy, `BNOT_V2/bnot-src/ibnot`, the stippling variant):
`m_epsilon = 1.0`, `m_max_iters = 500` (outer iterations), 500 Newton iterations per weight solve
(hard-coded in the `optimize_all(stepW, stepX, 500, epsilon, max_iters)` call), `stepX = stepW = 0`
(0 = line search), thresholds `0.1 * epsilon * A / N` (weights) and `0.1 * epsilon * sqrt(A^3) / N`
(positions); coarse loop at 2x the threshold, then the fine loop. (`pbnot` / `bbnot`, the
constant-density variants, use 0.03 instead of 0.1.) Our port's CLI has the same defaults (500 / 500
/ 1.0). **Our Python driver overrides two of them: `max_iters = 25`, `max_newton_iters = 50`**
(`bnot_data_gen.py`, `budget_teachers_driver.py`); epsilon stays 1.0.
- So the paper's "published default stopping criteria ... up to 25 Newton iterations for BNOT" is
  not accurate: the defaults are 500 outer / 500 Newton, and ours are 25 outer / 50 Newton.
- The outer cap does not bind: the fixed-build runs of the 10 runtime images stop after 2–4 outer
  iterations at 256 / 1024 / 1600 / 3136 / 6400 points (stats `iterations:`). Whether the Newton cap
  of 50 binds on the slow images is not known yet.
- **Job 24308577** (`bnot_convergence_check.sbatch`, `bnot_convergence_summary.py`) tests it on the
  10 runtime images: epsilon 1.0 with 25 / 50 (paper) and 500 / 500 (original defaults), epsilon 0.1
  and 0.01 with 500 / 500, epsilon 0.01 with 25 / 50; 1024 points scored with the paper metrics,
  6400 points timed. → `outputs/bnot_convergence_check/summary.md`.

**BNOT re-timed on the paper's machine type** (job 24308505, `bnot_retime_epyc7343.sbatch`, node
ise-6000-02, AMD EPYC 7343, load ~30 of 64 threads; the supplement states RTX 6000 + EPYC 7343 for
Fig. 7) → `outputs/icons_results_runtimes_bnot_epyc7343/`. Mean / median s, 10 images:

| points | 256 | 576 | 1024 | 1600 | 2304 | 3136 | 4096 | 5184 | 6400 |
|---|---|---|---|---|---|---|---|---|---|
| EPYC 7343, mean | 2.98 | 3.47 | 4.26 | 4.94 | 5.79 | 12.77 | 7.67 | 9.07 | 10.51 |
| EPYC 7343, median | 2.70 | 3.41 | 4.26 | 4.79 | 4.96 | 6.50 | 7.19 | 7.21 | 8.51 |
| EPYC 7702P, mean (in `runtimes_avg.json`) | 5.22 | 5.81 | 7.08 | 8.49 | 9.46 | 20.89 | 12.88 | 15.33 | 17.20 |
| paper's own curve (DDPM-500, stored) | 5.75 | 5.80 | 5.88 | 6.02 | 6.33 | 6.77 | 7.66 | 7.92 | 9.08 |

**On the paper's CPU, BNOT (epsilon 1.0) is faster than the paper's 500-step sampler from 256 to
2304 points** (4.26 vs 5.88 s at 1024) and about equal or slower above. `runtimes_avg.json` and the
graphs still hold the EPYC 7702P times; nothing was replaced. The paper files are not edited yet:
the BNOT curve depends on the tolerance question above (user, 2026-11-18).

### 2026-11-18 21:42: where the paper's Fig. 7 runtimes were measured (correction to the note above)

The user: the paper's BNOT times were measured on a CPU-only machine, not on the RTX 6000 node.
Checked against the file dates of the stored runs and the interactive `pycharm_srv` sessions in
`sacct` (the runs were started from those sessions; the match is by start time, so it is an
inference, not a recorded fact):

| curve | stored files | session it matches | node type |
|---|---|---|---|
| BNOT 256–2304 | 2026-08-09 18:52–19:01 (log `run_bnot_data_gen_v2_all.py_20260809_185210.log`) | 19971816, cs-cpu-08, from 18:51 | CPU only |
| BNOT, larger budgets | logs of 08-11 16:03, 08-13 14:02, 08-14 23:54 | ise-cpu128-11, cs-cpu-11, ise-cpu-intl-07 | CPU only, three other nodes |
| WVS 1024 | 2026-08-10 18:51 | 20001090, ise-cpu-intl-04, from 18:49 | CPU only |
| Ours (CN) 1024 | 2026-08-15 02:03 | 20168179, ise-6000-06, from 01:59 | RTX 6000 node |
| GBN 1024 | 2026-08-15 17:17 | 20189152, ise-cpu256-10, from 17:12 | RTX 6000 on a cpu256 node |

- So the supplement's sentence "measured on one machine with an RTX 6000 GPU and an AMD EPYC 7343
  CPU; WVS and BNOT run on the CPU" is not accurate: the four curves come from at least four nodes,
  and the old BNOT curve itself from several CPU-only nodes. It has to be rewritten whatever BNOT
  numbers are used.
- **The BNOT curve for the paper stays the CPU-only measurement** already in `runtimes_avg.json`
  (fixed build, ise-cpu128-07, AMD EPYC 7702P: 7.08 s at 1024, 17.2 s at 6400), which is the same
  kind of machine as the original. The EPYC 7343 run (`icons_results_runtimes_bnot_epyc7343/`,
  4.26 s at 1024) is a side measurement of how much the CPU matters (about 1.6x); do not put it in
  Fig. 7 unless the user decides to time everything on one node.
- Comparison that follows for the paper's 500-step sampler: 5.88 s against BNOT 7.08 s at 1024 and
  6.02 against 8.49 s at 1600, on different machines; BNOT faster at 256, equal at 576.

### 2026-11-18 21:55: BNOT convergence test restarted with a per-image time limit (job 24308720)

- **First attempt (24308577) cancelled after 10 min.** At epsilon 0.1 with the original caps
  (500 / 500), six images finished in ~40 s, then the seventh, `emoji-one_3_waving_hand_sign`, ran
  for more than 8 min without finishing (the icon `BNOT_V2/CONTEXT.md` already flags: large empty
  regions, nearly singular weight Laplacian). The script ran all 10 images in one call under one
  2-hour limit, so a stall blocked the rest of the setting.
- **Restart: job 24308720**, same `bnot_convergence_check.sbatch`, now one image per call
  (`--start i --end i+1`) under `timeout`: 600 s per image at 1024 points, 1800 s at 6400. A stopped
  image is printed in the log ("STOPPED after ... s") and listed in `summary.md`. A setting counts
  as done when `data/timestamps_<tag>_<points>/` exists, so the two finished settings were kept.
- **Partial results at 1024 points (10 runtime images, fixed build, node ise-cpu128-01):**
  - epsilon 1.0, caps 25 / 50 (paper): 2–3 outer iterations per image, 3.8–10.5 s.
  - epsilon 1.0, caps 500 / 500 (original defaults): the same iteration counts and capacity errors,
    3.4–9.5 s. **9 of 10 point sets are byte-identical to the paper setting**; the one that differs is
    `twitter_0_arrow_pointing_rightwards_then_curving_upwards` (same outer iterations, same reported
    capacity error, so probably the Newton cap of 50 inside one weight solve). How far its points
    move is not measured yet (`summary.md`, nearest-point distance column).
  - epsilon 0.1, 500 / 500, the six images that finished in the first attempt: 3–5 outer iterations
    (1–2 more than at epsilon 1.0), capacity errors 5e-8 … 3e-5. Not scored yet.
- **Reading so far:** the 25-iteration cap is not what stops BNOT at epsilon 1.0; the tolerance is.
  A tighter tolerance needs only a few more iterations on most icons but can stall on some, which is
  the practical reason for a cap or a time limit. Whether it improves the paper metrics is what the
  scored 1024-point table of `outputs/bnot_convergence_check/summary.md` will show.
- **If epsilon is changed for the paper**, it is not only Fig. 7: the BNOT columns of Table 1 and the
  Ours-BNOT training exemplars were generated at epsilon 1.0 with caps 25 / 50.

### 2026-11-18 22:05: user decision (epsilon stays 1.0) and the exemplar check (job 24308964)

- **Decision (user): epsilon 1.0 is enough for all results.** No BNOT result is regenerated at a
  tighter tolerance. The convergence job 24308720 is left running for the record only; so far, at
  epsilon 0.1 with caps 500 / 500, 9 of 10 images finish and `emoji-one_3_waving_hand_sign` is
  stopped by the 600 s limit.
- **Open question (user): does the solver fix change the BNOT point sets, i.e. do the exemplars have
  to be regenerated and Ours-BNOT retrained?** Evidence before this check: the other session's
  comparison of the 10 runtime images at 256–6400 points (stored vs fixed build, differences
  <= 3e-6 of a point spacing) and six old-vs-fixed runs (`BNOT_V2/CONTEXT.md`).
- **Job 24308964** (array 0-3, `bnot_fix_validation.sbatch` + `bnot_fix_validation.py`) tests it on
  the training exemplars themselves: a seeded random sample of each BNOT dataset (Icons-50 60 images,
  CelebA-5K 40, ShapeNet2D-4K 40, Anime-Chibi 40) is regenerated with the **old** binary
  (`BNOT_new/ibnot_cli/build.eigen_20261118/ibnot_new_cli`) and with the **fixed** one, same inputs
  and settings (seed 7, 25 / 50, epsilon 1.0, invert, 900 s limit per image). The input size is
  chosen per dataset as the one whose result reproduces the stored exemplar (none / 512 / 448).
  Compared: nearest-point distance old vs fixed and old vs stored (in point spacings), outer
  iterations, capacity error, and the paper metrics of both sets.
  → `outputs/bnot_fix_validation/summary.md` (per dataset: `result.json`).
- How to read it: "old vs fixed" near zero on every image = no regeneration and no retraining.
  "old vs stored" near zero confirms the test used the real generation settings; if it is large for
  a dataset, the settings were not reproduced and only the old-vs-fixed column counts there.

## 2026-11-19 00:30: BNOT runtime at its TRUE defaults (user request)

- **User decision:** BNOT results stay at the default tolerance (epsilon 1.0), and the runtime graphs
  must use BNOT's true default parameters: **max_iters 500, max_newton_iters 500**, not the paper
  driver's 25 / 50 caps (point sets are the same at 1024 points on 9/10 images, see the convergence
  check, but the timing must be of the real defaults).
- `bnot_retime_defaults.sbatch` (fixed build, 10 pinned runtime images, 512 x 512, seed 7, invert,
  1 thread, CPU-only EPYC 7702P cpu128 node like the current fixed-build curve; the job exits if the
  CPU differs; one image per call with a 3600 s guard, a stopped image is printed)
  -> `outputs/icons_results_runtimes_bnot_defaults500/`.
  Job 24310372: 256-6400. Job 24310374 (afterany): 7744-12544.
- `bnot_defaults_runtime_stage_3.py` (jobs 24310373 after the first, 24310375 after the second)
  -> `outputs/runtime_graphs_bnot_defaults500/`: `summary.md` (500/500 vs 25/50 per budget: mean,
  median, max, counts), `runtimes_avg.json` (= stored JSON with BNOT replaced; budgets with < 10
  timings left out), Fig. 7 in `plots/`, the new-variant graph (runtime_clean_stage_3 on the
  runtime_clean_blocks16 reps, symlinked) in `new_variant/plots/`. **Nothing stored is replaced**;
  swap into icons_results_runtimes only after the user has seen the graphs.

## 2026-11-19 02:00: runtime graphs re-measured on a NEW image set (user decision)

- The BNOT true-defaults re-time on the old 10 pinned images (job 24310372, cancelled at 4096) showed
  two icons dominating: at 3136 points `twitter_0_arrow_pointing_rightwards_then_curving_upwards`
  840 s (25/50 caps: 88 s) and `emoji-one_3_waving_hand_sign` 273 s (39 s); the other eight 8-15 s.
  Up to 2304 points 500/500 cost only 2-7% more than 25/50 (5.5 / 6.2 / 7.4 / 8.6 / 10.2 s). Folder
  `outputs/icons_results_runtimes_bnot_defaults500/` (256-3136 complete). The 50-Newton cap was
  binding on exactly those two icons.
- **User decision: re-time ALL methods on a new set, the first 10 (sorted) Icons-50 validation
  images** (`outputs/runtime_v10_subset.txt`: apple_0_automobile ... apple_0_eagle), staged with
  `budget_teachers_stage_0.py --prefix runtime_v10` (WVS/BNOT from Icons-50_1024_WVS/source, GBN from
  its own source) and `outputs/runtime_v10_source` (symlinks, ours). BNOT at its TRUE defaults
  (500 / 500, epsilon 1.0). If any image again blows up, report it; do not swap images silently.
- Jobs: 24310751 `runtime_v10_cpu.sbatch` (BNOT then WVS, one EPYC 7702P cpu128 node, 1 thread);
  24310752 `runtime_v10_gpu.sbatch` (GBN then ours, exclusive cpu256 RTX 6000 Ada node, ours 3 reps,
  pot-exact at all budgets + bisection-blocks16 at 3136-6400); 24310753 `runtime_v10_stage_3.py`
  (afterany both) -> `outputs/runtime_v10/`: summary.md (mean / median / max per method and budget),
  runtimes_avg_{exact,blocks16}.json, plots_{exact,blocks16}/{fig7,new_variant}/. Nothing stored is
  replaced. The exclusive GPU node may queue for days (the last one did); fallback = same script
  without --exclusive (dedicated GPU, CPUs reserved), to be decided with the user.

## 2026-11-19 12:30: runtime_v10 CPU part done; GPU part queued

- **BNOT (true defaults) and WVS on the new 10 images: done** (job 24310751, ise-cpu128-09, EPYC 7702P).
  BNOT mean / median s: 14.3 / 3.6 (256), 4.1 / 3.9, 5.9 / 6.1, 5.8 / 5.8, 6.7 / 7.0, 8.2 / 8.3,
  9.5 / 9.4, 11.0 / 10.8, 12.9 / 12.8 (6400). WVS 1.8 ... 50.3 s (close to the stored curve).
  No image stopped. **One outlier: apple_0_cloud_with_lightning at 256 points, 110 s** (others ~3.6 s).
  Re-run alone (`outputs/runtime_v10_cloud_check/`, job bnot-cloud): 121.5 / 122.1 s with 500 / 500,
  15.9 s with 25 / 50, 4 outer iterations each; the 500/500 and 25/50 point sets differ by at most
  3.7e-4 point spacings (same energy / capacity error). So it is reproducible: BNOT's default Newton
  cap spends ~100 s polishing converged weights on this image/budget; not noise, not a bad image.
- **User decision:** generate the new runtime graphs from the new images, with the **mean** for now;
  also compute the **median** as a backup. `runtime_v10_stage_3.py` now writes both:
  `runtimes_avg_<ot>.json` + `plots_<ot>/` (mean) and `runtimes_avg_<ot>_median.json` +
  `plots_<ot>_median/` (median; "avg" holds the median there), ot = exact / blocks16. Tested on the
  CPU part only (GBN / ours empty), then the test folder was removed.
- GPU job 24310752 (GBN + ours, exclusive cpu256 RTX 6000 Ada) pending, Slurm estimate 2026-11-20
  05:01; summary job 24310753 runs after it.

## 2026-11-20: runtime_v10 GPU part moved off the exclusive node
The exclusive cpu256 job (24310752) was estimated to start Nov 23 15:43 (all 32 cpu256 nodes full of
other users CPU jobs until Nov 23-29). User decision: shared node, one dedicated RTX 6000 Ada,
our own runs strictly one at a time. runtime_v10_gpu.sbatch now: --constraint=rtx_6000,
--cpus-per-task=16 (= THREADS), no --exclusive; logs load and GPU process count per step.
Cancelled 24310752 / 24310753; resubmitted as 24332116 (GPU) and 24332117 (summary, afterany).
Caveat for the write-up: GBN / ours timed on a shared node (reserved cores and GPU), BNOT / WVS on
a CPU-only node as before.

## 2026-11-20: runtime_v10 done (new image set, all four methods re-timed)
GPU 24332116 (cs-6000-03, RTX 6000 Ada, 16 cores; node load 4-25 of 64, no other GPU processes)
and summary 24332117 finished 01:49. Outputs in experiments/outputs/runtime_v10/ (summary.md, JSONs,
plots_{exact,blocks16}{,_median}/{fig7,new_variant}). Mean at 256 / 3136 / 6400 points (s):
BNOT 14.3 / 8.2 / 12.9 (256 mean = one 110 s image; median 3.6); WVS 1.8 / 21.5 / 50.3;
GBN 8.2 / 9.7 / 10.2; ours ddpm-500 exact 5.3 / 6.8 / 13.1, blocks16 5.3 / 5.9 / 6.9;
ours ddim-25 exact 0.27 / 1.6 / 7.5, blocks16 0.27 / 0.69 / 1.3. Stored runtimes_avg.json and the
paper figures are NOT replaced (waiting for the user).

## 2026-11-20 17:20: runtime_v10 SWAPPED IN (user decision: median, blocks16 OT)
icons_results_runtimes/runtimes_avg.json and plots/ now = runtime_v10 median / blocks16 (fig7 = paper
figure; plots/runtime_scaling_by_points_with_ddim25.* = with the DDIM-25 curve, for the supplement).
"avg" in the JSON holds the MEDIAN over 10 images. Old: runtimes_avg.json.bak_20261120 (13 budgets),
plots_bak_20261120/. Provenance: icons_results_runtimes/RUNTIMES_SOURCE.txt. The timestamps_* folders
there are the old raw timings. runtime_clean*-Icons-50 graphs are superseded (not deleted).
New medians (s) at 1024 / 1600 / 6400 points: BNOT 6.06 / 5.83 / 12.79 (true defaults) | WVS 6.78 /
10.98 / 50.76 | GBN 9.50 / 9.47 / 10.11 | ours DDPM-500 5.50 / 5.88 / 6.96 | ours DDIM-25 0.42 /
0.60 / 1.33. At 256: BNOT 3.63, WVS 1.67, GBN 9.27, ours 5.31 / 0.27.
STALE NUMBERS to replace in REBUTTAL_DRAFT.md (watch-out 6, A1, A3, A11, W1/Q1 rows, section 6,
break-even "7.7 s") and REBUTTAL_TEXT.md (lines ~22, 89-96: 0.44 / 0.61, 7.1 / 17.2, GBN 8.1 s).

## 2026-11-20 evening: paper tex in two versions; rebuttal drafts updated
User decisions: DDIM-25 is reported in the SUPPLEMENT only (new section "Faster Sampling" with a
six-method table: BNOT, WVS, GBN + our three variants sampled with ddim-25-eta1, mean +- std, and a
runtime graph with DDIM-25). Two versions of the paper (local paper_rebuttal/, mirrored to server paper_rebuttal/):
- *_rebuttal_11.10.tex (non-full): submitted text/numbers untouched; new results only. Supplement
  graph = the submitted Fig. 7 curves + DDIM-25 (runtime_v10, blocks16, MEAN). One pointer sentence
  in the main runtime paragraph.
- *_rebuttal_full_11.10.tex (full): every fix too. Fig. 7 = runtime_v10 median blocks16; Fig. 4 exact
  OT; BNOT exemplar / checkpoint / abstract wording (12 checkpoints); A2 rank statistic; A4 capacity
  paragraph; ablation caption (50-image subset); tone Table 3 + text on the real sampler (DDPM-500
  eval, fresh seeds, gradient through DDIM-25; random search replaced by D1 feedback); supplement
  hardware paragraph; OT-truncation note. Open (marked % TODO): tone figures + ink-gain sweep are
  still the one-step proxy.
New scripts: experiments/e13_ddim25_tex_table.py -> outputs/e13_ddim25_tex/,
experiments/runtime_rebuttal_graphs.py -> outputs/runtime_rebuttal_graphs/{full,nonfull}/.
Backups: paper_rebuttal/archive/rebuttal_bak_20261120/ (local), REBUTTAL_{DRAFT,TEXT}.md.bak_20261120.
REBUTTAL_DRAFT.md watch-out 7 = runtimes of record (runtime_v10); REBUTTAL_TEXT.md numbers updated.

## 2026-11-20 (late): tone figures rebuilt from the full-chain results (full version only)

- New `experiments/tone_results_stage_9_fullchain_plots.py` (jobs 24361835 render, 24361962 `--replot`)
  -> `outputs/tone_results/plots_fullchain/` (tone_improvement, tone_explain_compact_lg_2_woman_with_bunny_ears,
  per_image.csv, summary.md). Densities from the E16 stage-1 run (`tone_results/ddim25`), stipples from
  DDPM-500 on fresh seeds 0/1/2 (stage 5 samples; feedback = D1 fresh re-evaluation). Means reproduce
  full_sampler_ddim25 and fixed_point summaries exactly (16.51 / 16.66 / 16.88 / 16.89).
- Per image: field - curve +0.01 dB, field better on 11/24 (p=1); field - feedback +0.23 (16/24, p=0.06);
  curve - feedback +0.22 (18/24, p=0.02). Hero lg_2_woman_with_bunny_ears: curve +1.29 > field +0.80
  (seed mean); the figure shows seed 0, RMS 0.0773 -> 0.0690.
- paper4: `figures_rebuttal/tone_{improvement,explain_compact}_fullchain.pdf` used in
  main_EG_rebuttal_full_11.10.tex (Figs ToneMechanism/ToneGraph, captions rewritten; Feedback replaces
  Random Search). Backup: paper_rebuttal/archive/rebuttal_bak_20261120/main_EG_rebuttal_full_11.10.before_tonefigs.tex.
  Still open: supplement ink-gain sweep (one-step proxy; a full-chain re-run is needed).

## 2026-11-21: ink-gain sweep on the real sampler (job 24362421, done, 3:37 h)

- `tone_gain_sweep_ddim25.sbatch`: stage 2 now takes `--mode` (default onestep = submitted sweep);
  run with `--mode ddim`, submitted size (12 targets, 150 steps, x0.25..x4), then
  `tone_results_stage_10_gain_sweep_fullchain.py` (prepare / summarize): ddpm-500, fresh seeds 0/1/2,
  rendered at each gain -> `outputs/tone_results/gain_sweep_ddim25/{summary.md,gain_sweep_real_sampler.csv,
  tone_gain_sweep_compact.*}` (`gain_sweep.csv` there is stage 2's own stage-1-path table).
- field - none: +0.03, +0.04, +0.25 (9/12, p=0.06), +0.82 (10/12, p=0.003), +1.36 dB (11/12, p=0.001).
- field - curve: -0.09 (p=0.02, curve better), -0.14, -0.17, +0.12, +0.07 dB (none of the last four significant).
  So the gain over uncorrected grows with ink gain, but the field does not beat the tone curve at any gain;
  the submitted caption's "ordering unchanged" claim does not hold. Not yet put in the supplement (user decision).
- 2026-11-21: put into paper_rebuttal/supplemental_EG_rebuttal_full_11.10.tex as figures_rebuttal/tone_gain_sweep_fullchain.pdf
  (intro sentence and caption rewritten: gain over uncorrected grows with dot gain; field and curve within
  0.2 dB at every gain). Backup: archive/rebuttal_bak_20261120/supplemental_EG_rebuttal_full_11.10.before_gainsweep.tex.
  No tone TODOs remain in the full version.

- 2026-11-21: REBUTTAL_DRAFT.md (watch-out 1, A12, section 2 tone answer, section 3, item 5, W3 rows, section 6) and REBUTTAL_TEXT.md (Tone paragraph, committed revisions) updated with the real-sampler tone results; backups .bak_20261121.

## 2026-11-21 (night): check of the week's changes; pending results recorded

Checked by the first rebuttal session. Backups of the three docs: `*.bak_20261121b`.

- **Recorded (REBUTTAL_DRAFT.md A21–A25, status table, section 5 rows; REBUTTAL_TEXT.md placeholders
  filled):** E14 seeds, Fig. 3 strips over 25 realizations, model size (44.1 M), the ablation with
  the few-step sampler, and the final low-noise fine-tune trend.
- **Two statements softened because of the new results:**
  1. The gap to GBN is seed-dependent in size: HOT mean −21 / −12 / −20% in three runs (always
     significant). REBUTTAL_TEXT.md now says "12–21% lower in three runs", not "−22%".
  2. Gating: a second, independent evaluation gives a smaller effect than A20 (CVT +0.3%, capacity
     +4.5% median). The text now gives the range and calls the effect small.
- **Checked and consistent:** REBUTTAL_TEXT.md runtimes against `runtime_v10/summary.md` (medians,
  blocks16); the full paper's runtime sentences (fastest from 1024 points on except 1600, on par
  with BNOT there; WVS and BNOT faster at 256 / 576); the supplement's hardware sentence (job
  24332116 ran on cs-6000-03, AMD EPYC 7343, RTX 6000 Ada).
- **Still wrong in the paper files (not edited by this session):**
  1. Fig. 3 paragraph, both versions: "within about one percentage point" (twice) does not hold for
     Ours-WVS (1.9 points on average, A22).
  2. The non-full version keeps, by design, the submitted BNOT curve (up to 1802 s) and the sentence
     "faster than BNOT and GBN at every evaluated budget, with the gap to BNOT widening", which we
     now know are wrong; its new supplement figure shows that curve next to DDIM-25. If the non-full
     version is the one that is submitted, it knowingly carries the error.
  3. If the supplement gets an ablation table with noGate / noKDE rows as means, Full is not the best
     on mean capacity (A24 caution).
- **Uncommitted on `rebuttal` as of this check:** everything since e420c8b (2026-11-18 20:18): the
  three docs, `paper_rebuttal/` (both versions, figures_rebuttal, two new tables), and the new scripts
  (bnot_*, runtime_v10_*, runtime_rebuttal_graphs.py, e13_ddim25_tex_table.py, tone stage 9 / 10,
  tone_gain_sweep_ddim25.sbatch, the stage-2 `--mode` change).

## 2026-11-22: paper folder renamed, untracked and trimmed (user request)

- **`paper4/` is now `paper_rebuttal/`** (every `paper4/` path in this file and in REBUTTAL_DRAFT.md
  was rewritten to the new name, also in older entries).
- **The paper files are no longer tracked by git and must not be committed.** They were removed
  from the index (they stay on disk) and `paper*/` is in `.gitignore`. They remain in the history
  of the `rebuttal` branch on GitHub (commits bcdea5f and c396da8); removing them from there would
  need a history rewrite and a force push, which was not done.
- **`paper/` (untracked, ignored) is the user's snapshot of the current Overleaf project**, uploaded
  2026-11-21. `paper_rebuttal/` now holds only files that are new or changed against it: the four
  tex files, seven figures in `figures_rebuttal/`, two tables in `tables_rebuttal/`. The two tables
  identical to Overleaf's (`quantitative_comparison_table.tex`, `..._full_table.tex`) were removed.
  The upload list is `paper_rebuttal/UPLOAD_TO_OVERLEAF.md`; `python3 logs/refs_check_paper.py
  paper_rebuttal` re-checks that every file the tex files reference exists in one of the two folders.

## 2026-11-22: paper files purged from the `rebuttal` branch history (user request)

- The 14 commits of `rebuttal` were rewritten without `paper4/` and force-pushed (with lease).
  Messages, authors and dates are unchanged; the final tree is identical, so the checkout was not
  touched. Branch point (`fm_and_clip`, 5587875) and every other branch are unchanged.
- **Every rebuttal commit hash changed.** Hashes quoted earlier in this file are the old ones:

  | old | new | commit |
  |---|---|---|
  | bcdea5f | 6df34e1 | Rebuttal experiments (EG 2027, paper1182) |
  | e83e840 | cb538b0 | CONTEXT: record the rebuttal branch |
  | 6d92ec8 | d0c4818 | Rebuttal: gating ablation, generalization diagnostics, paired-summary fix |
  | e72f87e | 2df7a93 | Low-noise fine-tune experiment |
  | 1ea96e4 | 5fe435c | CONTEXT: low-noise fine-tune result |
  | 7b191dc | c478eb4 | Low-noise fine-tune: continue to epoch 6000 |
  | a6155d6 | 22040d3 | TODO: investigate BNOT runtime numbers |
  | 6c99561 | 8b20b64 | Rebuttal: E14 seed robustness, Fig. 3 strip realizations, parameter counts |
  | 09b9cea | e2f5a3f | Rebuttal text, draft 1 |
  | 3081ea1 | 7776403 | Few-step sampler goes to the supplement; ablation table with DDIM-25 |
  | 0e1122e | d37dbf8 | BNOT runtime TODO resolved |
  | e420c8b | 4893ded | CONTEXT: BNOT runtimes being replaced and archived |
  | c396da8 | 639e5e1 | Rebuttal: BNOT runtime fix, runtime_v10, tone on the real sampler |
  | 5419ce1 | df36c23 | Stop tracking the paper files; ignore paper folders |

- **Local safety copy of the old history:** branch `backup/rebuttal-before-purge-20261122` in this
  server repository only (never pushed). It still contains the paper files; delete it with
  `git branch -D backup/rebuttal-before-purge-20261122` when it is no longer wanted.
- **What the purge does not do:** GitHub still serves the old commits to anyone who asks for them by
  hash (checked right after the push), until GitHub's own garbage collection removes them; only
  GitHub Support can force that. The repository is public (its page loads without login), so the
  rebuttal notes on this branch (this file, REBUTTAL_DRAFT.md, REBUTTAL_TEXT.md, the mock review)
  are public too. Whether that is acceptable is the user's decision.
- Anyone who had fetched the old `rebuttal` must reset to `origin/rebuttal`, not merge.
