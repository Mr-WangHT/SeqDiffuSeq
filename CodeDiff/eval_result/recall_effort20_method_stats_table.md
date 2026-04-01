# Recall@20 / Effort@20 Statistical Table

| method | mean Recall@20%LOC | mean Effort@20%Recall | recall rank (high better) | effort rank (low better) | ΔRecall vs baseline | ΔEffort vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| full_diffusion_prototype | 0.222634 | 0.319059 | 1 | 1 | +0.024079 (+12.13%) | -0.013629 (-4.10%) |
| one_step_classifier | 0.216548 | 0.319666 | 2 | 2 | +0.017993 (+9.06%) | -0.013022 (-3.91%) |
| one_step_prototype | 0.214885 | 0.328199 | 3 | 3 | +0.016330 (+8.22%) | -0.004489 (-1.35%) |
| baseline_within_release | 0.198555 | 0.332688 | 4 | 5 | +0.000000 (+0.00%) | +0.000000 (+0.00%) |
| consistency_full_w01_resume | 0.184245 | 0.329031 | 5 | 4 | -0.014310 (-7.21%) | -0.003657 (-1.10%) |

Notes: Recall@20%LOC higher is better; Effort@20%Recall lower is better.