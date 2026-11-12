# P2 预注册：生成过程中的跨步累积（冻结于 2026-10-08，在任何 P2 数据计算之前）

## 问题
在动作生成的 10 个去噪步里，扰动造成的最终动作变化有多大，主要取决于以下哪一项？
- 每一步推得多大（逐步强度）；
- 有效推动发生在多少步上（时间分布）；
- 各步是否朝同一方向推（方向一致性）。

## 数据
- 来源：D3（native-visual-multi-20260927-v2），原始 trace `traces/qNNN.npz`。
- 范围：每个 episode 3 个查询，即第一次查询 q0 和两个 suite 专属 landmark。
- 读取方式：只按字节范围读取下列成员：

| 成员 | 形状 | 来源标签 |
|---|---|---|
| native/{actual,clean}/velocity | (10,10,32)，bf16 位存为 uint16 | 原生 |
| native/{actual,clean}/x_t | (10,10,32) f32 | 原生 |
| native/{actual,clean}/time | (10,) | 原生 |
| native_{actual,clean}_candidate_actions | (10,32) f32 | 原生 |
| {actual,clean}_physical_actions | (10,7) f64 | adapter 之后 |
| free/velocity_delta | — | 记录器 summary |
| shared_xt/velocity_delta | (3,10,32) | 反事实 / 影子 |
| shared_indices | — | — |
| shadow_error/{actual,clean}/velocity_rms_error | — | 数值误差 |

- clean 的含义：在同一已访问状态上换回 clean 渲染，用同一噪声做的原生生成（clean@visited）。它测的是局部视觉敏感性，不包含轨迹分叉。
- 主体人群：10 个持续条件。
  - q0 位置的 pulse 与持续条件相同。
  - landmark 位置的 pulse，当前输入已是 clean，只用作健全性检查（预期 Δ≈0），不进入主分析。

## 定义
- 单位：配对 (episode, 查询)。第 k 步（k=0..9）：
  - dt_k = time 的差分；
  - u_k = dt_k · (v^actual_k − v^clean_k)。
- 坐标空间：
  - 主空间 E：执行块，即位置 0..4 × 模型维度 0..6，共 35 维（归一化动作空间）。
  - 次空间 F：位置 0..9 × 维度 0..6，共 70 维。
  - 维度 7..31 是填充维，不使用。
- 量：
  - 逐步能量 S = Σ_k ‖u_k‖²
  - 最终变化 Δx = Σ_k u_k，F = ‖Δx‖²
  - 有效步数 K_eff = (Σ_k‖u_k‖)² / S，范围 1..10
  - 方向一致性 c = ‖Σu_k‖ / Σ_k‖u_k‖，范围 0..1（c=1 表示完全同向，c<1 表示存在抵消）
  - 恒等式：F = S · K_eff · c²，即 log F = log S + log K_eff + 2 log c
- 逐步剖面：share_k = ‖u_k‖²/S；cos_k = cos(u_k, Δx)。
- 反事实对照（影子 / 反事实，仅作解释）：在 k ∈ {0,5,9} 上，比较 shared_xt 的速度差范数与自然路径的速度差范数，取比值。

## QC（未通过即停止并调查）
- QC1：两个分支的 x_t[0] 完全相等。共享噪声，要求 Δx_0 = 0。
- QC2：恒等式残差 ‖Σu_k − Δcandidate‖/‖Δcandidate‖，在 E 空间内中位数 < 1%，99 分位 < 5%。
- QC3：landmark 处 pulse 的 ‖Δx‖ 相对持续条件的中位数 < 5%。
- QC4：报告 shadow_error 的速度数值误差；不把它当作除数或底线合并使用。

## 主分析
- 方差分配：对 log F、log S、log K_eff、2 log c 分别分解，层级为条件（10）、任务（40）、条件×任务、格内（初态 / 种子 / 查询）。
  - 报告 Var(log S)、Var(log K_eff)、Var(2 log c) 以及三个 2Cov 项。
  - 先按查询分开报告（q0、landmark1、landmark2），再报告合并结果。
- 判据（冻结）：
  - **D2-时间累积是独立因子**：在某一层级上，[Var(log K_eff)+Var(2 log c)] / [Var(log S)+Var(log K_eff)+Var(2 log c)] ≥ 0.25；并且 4 折留一 suite 时，该比例与全数据值相差都不超过 ±0.10。否则结论为"逐步强度主导"。
  - **D3-系统抵消**：只有当任务簇 bootstrap（2000 次）给出的 c 中位数 95% 区间上界 < 0.9 时，才报告"存在系统性的跨步抵消"；同时报告 c 的完整分布与各条件的中位数。
- 只描述、不作判据：S、K_eff、c 与条件、查询位置的关系；c 与反事实比值的关系。

## 不在 P2 内
与结局的关系属于 P3；读出方向分解属于 P1。P2 不对成功率下任何结论。
