# codediffpro 说明

本目录是基于原型判别（无分类头）的实验实现，采用扩散去噪表示进行行级缺陷识别。

## 第一部分：分类改进（已完成）

### 1. 当前代码中实际使用的损失

总损失（见 `codediffpro/modeling/diffusion_minimal.py`）：

$$
L = L_{mse} + \lambda_c L_{consistency} + \lambda_p L_{proto} + \lambda_r L_{ranking} + \lambda_h L_{hardneg}
$$

其中：

- `L_mse`：扩散噪声回归主损失，保证去噪学习稳定。
- `L_consistency`：相邻时间步一致性约束，减少反推过程抖动。
- `L_proto`：基于 0/1 原型的交叉熵损失（含 `proto_margin` 与类别权重），增强类别可分性并缓解不平衡。
- `L_ranking`：正样本相对负样本的间隔约束。
- `L_hardneg`：针对高分负样本的惩罚。

### 2. 为什么这些损失起作用

- `mse + consistency` 提供了稳定的去噪表示基础。
- `proto` 直接把去噪表示拉向正确原型，提升行级判别质量。
- `proto_pos_weight` 对稀有正类加权，缓解“全判负”倾向。
- `ranking + hardneg` 在 hard-noise 分支上生效后，能压制高分误报负样本，提高 precision 与 F1。

### 3. 指标改进（分类视角）

- 基线（minimal v1）最佳 `valid_f1`: `0.010622`
- 当前 40 epoch 实验最佳 `valid_f1`: `0.222304`（epoch 27）
- `valid_precision` 峰值：`0.297655`（epoch 28）
- `valid_recall` 峰值：`0.196078`（epoch 15）

曲线图：

![验证曲线](eval_result/activemq_codediffpro_rankhn_hardt_ep40_20260326/curves_f1_p_r_recall20_effort20.png)

收敛判断（分类相关）：

- `valid_f1` 在后期进入平台区，最后 10 个 epoch 均值/标准差约为 `0.202667 ± 0.012146`。

## 第二部分：排序改进（待完成）

> 本部分将聚焦文件内排序目标（Recall@20 / Effort@20），后续与分类目标分开设计与验证。

计划补充内容：

- 排序目标定义与优化目标函数。
- 排序导向训练策略（采样、损失权重、模型选择准则）。
- 与第一部分的对照实验与消融结果。
- 排序指标收敛分析与最佳 checkpoint 选择规则。

## 复现命令

### 训练

```bash
/mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python codediffpro/train_minimal.py \
  --dataset activemq \
  --data-dir codediff/data/lineDP_dataset \
  --tokenizer-json codediff/data/lineDP_dataset/activemq_bpe/tokenizer.json \
  --exp-name activemq_codediffpro_rankhn_hardt_ep40_20260326 \
  --num-epochs 40 \
  --loss-mode hybrid \
  --save-all-checkpoints \
  --device cuda:1
```

`--loss-mode` 可选值：

- `classify_focus`：只使用分类导向损失（`mse + consistency + proto`）。
- `rank_focus`：强调排序导向损失（`ranking + hard_negative`，并保留较小权重的 `proto` 稳定项）。
- `hybrid`：按配置权重同时使用分类与排序损失。

兼容说明：`reclassify` 与 `rerank` 仍可使用，但已标记为旧别名，建议改为 `classify_focus` / `rank_focus`。

### 推理

```bash
/mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python codediffpro/infer_prototype.py \
  --model-dir codediffpro/output/model/activemq/activemq_codediffpro_rankhn_hardt_ep40_20260326 \
  --release activemq-5.2.0 \
  --tau auto
```

## 输出目录

- 模型 checkpoint：`codediffpro/output/model/<dataset>/<exp_name>/`
- 训练日志与 CSV：`codediffpro/output/loss/<exp_name>/`
- 预测结果：`codediffpro/output/prediction/<exp_name>/`
- 评估结果：`codediffpro/eval_result/<exp_name>/`
