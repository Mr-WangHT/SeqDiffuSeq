```mermaid
flowchart TD
    A[Batch fields 批数据字段<br/>src ids and masks 源端id和mask<br/>tgt ids and masks 目标端id和mask] --> B[Trainer forward backward 训练前反传<br/>split microbatch 按微批切分]
    B --> C[Sample timestep and weight 采样时间步和权重]
    C --> D[Call diffusion training losses 调用扩散损失]

    D --> E[Take target ids from decoder ids 从解码端取目标id]
    E --> F[Embed target ids to x start mean 目标id映射到x start均值]
    F --> G[Build x start with std at t zero 用t0方差构造x start]
    G --> H[Sample gaussian noise 采样高斯噪声]
    H --> I[Forward diffusion q sample to x t 前向扩散得到x t]

    I --> J[Self conditioning 自条件]
    J --> J1[Init self cond as zeros 初始化为全零]
    J1 --> J2{coin flip 随机分支}
    J2 -->|yes 是| J3[No grad draft forward 无梯度草稿前向]
    J3 --> J4[Set self cond from draft 用草稿更新自条件]
    J2 -->|no 否| J5[Keep zeros 保持全零]

    J4 --> K[Main model forward 主模型前向]
    J5 --> K

    A --> S1[Source condition ids and mask 源端条件]
    A --> S2[Target decoder mask 目标端解码mask]
    S1 --> K
    S2 --> K

    K --> L[Model output 模型输出]
    L --> M[Choose target type 选择监督目标]
    M --> M1[Predict x start branch 预测x start分支]
    M --> M2[Predict noise branch 预测噪声分支]

    M1 --> N[Compute mse loss 计算mse损失]
    M2 --> N
    N --> N2[At t zero use x start recon loss t0使用重建损失]

    L --> O[Decoder nll via lm head logits 词表nll损失]
    I --> P[Terminal prior loss 末时刻先验损失]

    N2 --> Q[Total loss is mse plus nll plus prior 总损失汇总]
    O --> Q
    P --> Q

    Q --> R[Apply timestep weight and mean 加权并取均值]
    R --> T[Backward optimize and ema 反传优化与EMA]
```

## 中文说明（单步前向）

1. **读取 batch**
    - 从 DataLoader 取到四类字段：`input_ids`、`attention_mask`、`decoder_input_ids`、`decoder_attention_mask`。
    - 在 `Trainer.forward_backward` 中按 `microbatch` 切分后逐段计算。

2. **采样时间步并进入扩散损失**
    - `schedule_sampler` 采样当前时间步 `t` 和权重 `weights`。
    - 调用 `diffusion.training_losses(...)` 进入核心损失计算。

3. **构造扩散输入**
    - 目标端 token（`decoder_input_ids`）先映射到 embedding，得到 `x_start_mean`。
    - 在 `x_start_mean` 上加标准差扰动得到 `x_start`，再通过前向扩散 `q_sample` 得到带噪的 `x_t`。

4. **Self-conditioning**
    - 先把 `self_conditions` 设为全零。
    - 以约 50% 概率先无梯度跑一遍模型拿到草稿输出，再把该输出 `detach` 后作为 `self_conditions` 进行正式前向。

5. **主模型前向（Encoder-Decoder）**
    - 源端条件：`input_ids + attention_mask`。
    - 目标端输入：`x_t`（以及 `decoder_attention_mask`）。
    - 模型输出 `model_output`，用于预测目标扩散量（通常是 `x_start` 或噪声）。

6. **计算损失项并汇总**
    - **MSE 主损失**：`(target - model_output)^2`。
    - **t=0 特例**：用 `x_start` 重建误差替换对应位置。
    - **Decoder NLL**：通过 `lm_head` 的词表 logits 与离散 token 做监督。
    - **tT prior loss**：末时刻先验正则项。
    - 总损失：`loss = mse + decoder_nll + tT_loss`。

7. **回传与更新**
    - 训练侧对总损失乘以时间步权重并求均值。
    - 执行 `backward`、`optimizer.step()` 和 `EMA` 参数更新。