# CodeDiff 全量想法、实验与结果总报告（2026-03-24）

## 1. 目标
本报告把我“想做的事情”全部落地为可执行实验，并统一记录：
- 想法（为什么做）
- 实验设计（怎么做）
- 结果（数字）
- 结论（下一步）

范围以当前 activemq 任务为主，聚焦 Recall@20%LOC 与 Effort@20%Recall，以及训练/验证行为。

---

## 2. 想法清单（含执行状态）

1) 想法A：比较不同推理范式（分类头 vs 原型相似度；one-step vs full diffusion）
- 状态：已完成
- 目的：找出在不改训练的前提下，最有效的推理方式

2) 想法B：验证采样随机性是否是主要问题（随机 reverse vs 确定性 posterior mean）
- 状态：已完成
- 目的：排除“随机采样噪声导致性能差”的假设

3) 想法C：检查“teacher-forced 可分性”与“完整反向链可分性”是否一致
- 状态：已完成
- 目的：确认训练-推理不一致是否存在

4) 想法D：新增训练侧“采样路径一致性损失”并做快速 A/B
- 状态：已完成（快速对照）
- 目的：验证 consistency loss 是否能改善验证指标方向

5) 想法E：做完整规模 consistency 训练并给出最终 Recall/Effort
- 状态：已完成（20 epoch，含预测与 Recall/Effort 评估）
- 备注：使用可断点恢复配置（按 epoch 存盘）完成训练

---

## 3. 历史主实验结果（已完成）

来源：CodeDiff/eval_result/recall_effort20_method_stats_table.md 与相关 summary csv/json。

### 3.1 方法横向对比（核心）

| 方法 | mean Recall@20%LOC | mean Effort@20%Recall | 结论 |
|---|---:|---:|---|
| full_diffusion_prototype | 0.222634 | 0.319059 | 当前最优（高 Recall、低 Effort） |
| one_step_classifier | 0.216548 | 0.319666 | 次优 |
| one_step_prototype | 0.214885 | 0.328199 | 优于 baseline |
| baseline_within_release | 0.198555 | 0.332688 | 最差（对照） |

### 3.2 相对 baseline 的提升
- full_diffusion_prototype：Recall +12.13%，Effort -4.10%
- one_step_classifier：Recall +9.06%，Effort -3.91%
- one_step_prototype：Recall +8.22%，Effort -1.35%

结论：推理侧改造有效，但仍未达到理想水平，说明瓶颈很可能在训练目标与推理路径不一致。

---

## 4. 关键诊断实验（已完成）

### 4.1 随机 reverse vs 确定性 reverse（posterior mean）

对比对象：
- stochastic full-diffusion prototype
- deterministic full-diffusion prototype

结果：
- stochastic：mean Recall@20 = 0.222634，mean Effort@20 = 0.319059
- deterministic：mean Recall@20 = 0.195503，mean Effort@20 = 0.350885

结论：确定性反向链更差，说明“每步随机性”不是当前主问题。

### 4.2 可分性诊断（t-wise 与 full chain）

在 t=0/10/19 的 teacher-forced 诊断中，margin_gap 很大且分类几乎完美；
但 full diffusion release 统计显示：
- prob_q50 约 0.505
- 预测正例比例约 0.59
- precision 极低（约 0.001~0.005）
- recall 中高（约 0.58~0.62）

结论：存在明显训练-推理路径差异，full chain 下概率塌缩到 0.5 附近并偏向正类，导致 precision 崩塌。

---

## 5. 本轮新增实验：Consistency Loss 快速 A/B（已执行）

## 5.1 实验目的
验证新增训练项（采样路径一致性损失）是否至少在短程训练里带来“方向正确”的信号。

## 5.2 统一设置
- dataset: activemq
- num_epochs: 2
- diffusion_steps: 20
- line_window_size/stride: 64/64
- max_lines_per_file: 64
- batch_size: 32
- do_predict: false（仅做快速验证）
- 其余参数保持一致

实验组：
- w0：consistency_weight = 0.0
- w0.1：consistency_weight = 0.1

## 5.3 命令

w0:
bash codediff/train_scripts/run.sh --python-bin /mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python --exp-name activemq_consistency_ablation_w0_quick_20260324 --num-epochs 2 --eval-every-epochs 1 --line-window-size 64 --line-window-stride 64 --max-lines-per-file 64 --batch-size 32 --consistency-weight 0.0 --no-predict

w0.1:
bash codediff/train_scripts/run.sh --python-bin /mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python --exp-name activemq_consistency_ablation_w01_quick_20260324 --num-epochs 2 --eval-every-epochs 1 --line-window-size 64 --line-window-stride 64 --max-lines-per-file 64 --batch-size 32 --consistency-weight 0.1 --no-predict

## 5.4 结果

### w0（consistency=0.0）
- epoch1: valid_f1=0.0074316, valid_acc=0.6129, valid_recall=0.3917
- epoch2: valid_f1=0.0073820, valid_acc=0.1681, valid_recall=0.8362
- best(valid_f1): 0.0074316（epoch1）

### w0.1（consistency=0.1）
- epoch1: valid_f1=0.0076200, valid_acc=0.6308, valid_recall=0.3832
- epoch2: valid_f1=0.0073747, valid_acc=0.1801, valid_recall=0.8234
- best(valid_f1): 0.0076200（epoch1）

### 快速结论
- 在“短程+轻量窗口”设置下，consistency=0.1 的最佳 valid_f1 略好于 w0（0.007620 vs 0.007432）。
- 两组在第2轮都出现 recall 上升、accuracy 降低的倾向，说明正类偏置问题依旧明显。
- 一致性项有潜在正向信号，但幅度小，仍需完整训练与最终 Recall/Effort 指标确认。

---

## 6. 综合结论

1) 推理侧：full_diffusion_prototype 依然是当前最优选择。
2) 采样随机性：不是主瓶颈，确定性链路反而更差。
3) 真正问题：更像是训练目标与推理路径不匹配，叠加类不平衡导致 precision 崩塌。
4) 新增 consistency loss：快速 A/B 显示弱正向信号，但完整训练后未带来 Recall@20 提升。

---

## 6.1 完整规模 consistency 实验（已完成）

### 配置
- exp_name: activemq_consistency_full_w01_resume_20260324
- num_epochs: 20
- eval_every_epochs: 5
- consistency_weight: 0.1
- save policy: 按 epoch 存盘（可断点续跑）
- prediction: 开启（within-release 4 个版本）

### 训练关键结果
- epoch5: valid_f1=0.007722, valid_recall=0.947293, valid_acc=0.099380
- epoch10: valid_f1=0.007551, valid_recall=0.947293, valid_acc=0.078897
- epoch15: valid_f1=0.007785, valid_recall=0.951567, valid_acc=0.102710
- epoch20: valid_f1=0.007903, valid_recall=0.837607, valid_acc=0.222104

### 最终 Recall/Effort（基于预测文件）
- mean Recall@20%LOC = 0.184245
- mean Effort@20%Recall = 0.329031

### 与 baseline 对比
- Recall@20%LOC: 0.184245 vs 0.198555（-0.014310, -7.21%）
- Effort@20%Recall: 0.329031 vs 0.332688（-0.003657, -1.10%）

结论：完整 consistency 训练在 effort 上有小幅改善，但 recall 明显下降，综合不如 baseline 与已有最优方法。

---

## 7. 下一步实验建议（按优先级）

1) 在 consistency 基础上加入类别不平衡加权（pos_weight 或 focal-like），优先减少假阳性并保持 recall。
2) 将一致性项改为“按 t 分段加权”或“仅对中后段 t 生效”，避免过强约束损伤排序能力。
3) 验证指标加入 effort-aware 代理（而不只看 valid_f1），降低“高 recall/低 precision”偏置。

---

## 8. 结果文件索引

- 统计总表：CodeDiff/eval_result/recall_effort20_method_stats_table.md
- 方法均值：CodeDiff/eval_result/recall_effort20_method_compare_summary.csv
- full diffusion prototype summary：CodeDiff/eval_result/full-diffusion-prototype/recall_effort20_summary.json
- deterministic summary：CodeDiff/eval_result/full-diffusion-prototype-deterministic/recall_effort20_summary.json
- 诊断 summary：CodeDiff/eval_result/diagnostics/activemq_sw_20260323_114113/prototype_diagnosis_summary.json
- 新增快速A/B（w0）：
  - codediff/output/model/CodeDiff/activemq/activemq_consistency_ablation_w0_quick_20260324/run_config.json
  - codediff/output/loss/CodeDiff/activemq_consistency_ablation_w0_quick_20260324/activemq-loss_record.csv
- 新增快速A/B（w0.1）：
  - codediff/output/model/CodeDiff/activemq/activemq_consistency_ablation_w01_quick_20260324/run_config.json
  - codediff/output/loss/CodeDiff/activemq_consistency_ablation_w01_quick_20260324/activemq-loss_record.csv
- 新增完整 consistency（w0.1, 20 epoch）：
  - codediff/output/model/CodeDiff/activemq/activemq_consistency_full_w01_resume_20260324/run_config.json
  - codediff/output/loss/CodeDiff/activemq_consistency_full_w01_resume_20260324/activemq-loss_record.csv
  - codediff/output/prediction/CodeDiff/activemq_consistency_full_w01_resume_20260324/within-release/*.csv
  - CodeDiff/eval_result/consistency-full-w01-resume-20260324/recall_effort20_summary.json
