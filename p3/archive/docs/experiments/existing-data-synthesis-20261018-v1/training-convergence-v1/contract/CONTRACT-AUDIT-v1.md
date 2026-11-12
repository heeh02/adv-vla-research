# C/T 10k→20k 训练合同：实际做法与实验意义

审计始于2026-10-18，完成于2026-10-19。四条 C/T seed0/1 分支都有真实20,000-step完成回执和父10k全状态恢复证明。恢复保持 optimizer、EMA、RNG 和 global step，第一新步10001；这是已核实的执行事实。

这个实验检验的是：**同一批示教、模型和更新预算下，扩大示教内的时间位置覆盖，是否比反复使用每条示教的两个固定状态更有效。** 它是时间支持范围对照，不是 response-guided augmentation、对抗训练或新loss方法。20k表示追加了更新预算，并不自动表示训练充分。

| 项目 | C | T | 对解释的影响 |
|---|---|---|---|
| 训练示教 | 相同1,600条、40任务 | 相同1,600条、40任务 | 没有新增示教或任务 |
| 每epoch输入 | 每demo固定两时刻，共3,200状态 | 每demo两个固定时间分层各轮换一处物理时刻，共3,200slot | C/T唯一改变因素是跨epoch的物理时间支持 |
| 10k→20k | 继续重复固定两时刻 | 重用原50轮排程，global epoch50…99映射physical epoch0…49 | T后半程不增加新位置 |
| batch | 16 anchors×2 clean副本=32行 | 16 slots×2 clean副本=32行 | 每步只有16个物理状态；副本有各自普通增强/flow随机量 |
| 累计预算 | 10k=50epochs/320k行；20k=100epochs/640k行 | 完全相同 | 一个update就是一次stock optimizer update，无梯度累计 |
| 模型更新范围 | 动作专家+action/time投影 | 完全相同 | 19叶、430,098,464可训练参数；视觉/语言前缀32叶、2,923,335,408参数冻结 |
| 目标 | 普通监督flow velocity MSE | 完全相同 | 无额外adversarial、response或consistency项 |

模型是绑定原artifact `sha256:5df68cd8…` 的 π0.5 LIBERO：非LoRA，Gemma2B前缀、Gemma300M动作专家，原BF16及必要FP32路径，H10/D32。配置 `discrete_state_input=False`，没有state token/模型proprioception路径；state字段用于形状与来源时间/动作完整性。原10×7示教delta action按原quantile统计归一化，再补零到32维；`extra_delta_transform=False`，不再减state。输入是单时刻两相机图像与任务指令，不是视频。源码依据见 `SOURCE-LINES-v1.json` 的 `profile`、`profile_config`、`runtime_config`、`reader_no_padding`。

精确loss为：ε∼N(0,I)，t=.999×Beta(1.5,1)+.001，xₜ=tε+(1−t)a，uₜ=ε−a，求 `mean((vθ−uₜ)^2)`。先对32动作维求均值，再对B×10位置求均值；训练B32时分母为32×10×32。没有只保留执行7维的loss mask。所有10个真实时间点均valid，没有时间padding；25个补零动作维仍有flow噪声目标，参与25/32的通道权重。因此总loss不是7维动作MSE，更不是闭环成功率。源码依据：`loss_targets`、`loss_reduction`、`training_loss`、`padding`、`reader_no_padding`。

| 曲线 | 实际口径 | 可比较范围 |
|---|---|---|
| `UPDATES-v1.jsonl/stock_info.loss` | online参数、训练B32、clean/clean双行；随stock step换key；更新前loss写在更新完成后 | 每条分支的优化轨迹；C/T训练物理样本不同，不能称同样例风险差 |
| `INTERNAL_EMA-stepS.json` | 更新至S后的EMA、固定800内部验证anchor、4条件、固定anchor key；逻辑B1 | 同global step/condition下可直接比较C/T与seed，因为验证人口和口径相同 |
| `RESTORE_CHECK_INTERNAL_EMA-step10000.json` | 原10k同状态、同人口、同key的恢复核对 | 单独保存；明确不算新学习曲线点 |

训练日志 `effective_update=s` 的loss来自更新前s−1参数，应标为“第s次更新使用的loss”。monitor为更新至S后的EMA。monitor调用 `compute_loss(train=True)`，包含固定随机键下的crop/resize/rotate/color jitter、flow noise/time；它不是禁增强的纯teacher action回归。固定键方便跨checkpoint比较，但不代表多次独立Monte Carlo采样。四条件共享同anchor key；每次800×4=3,200逻辑B1评分，按anchor→source demo→task取宏平均。不要采用早期PLAN中“两monitor seeds/6,400行”的候选设定代替实际执行代码。源码依据：`scorer`、`four_condition_keys`、`augmentation`、`monitor`、`monitor_call`。

验证集确实与本轮优化数据按完整示教隔离：训练3,200 anchors/1,600 demos；验证800 anchors/400 demos；anchor与source_demo_id交集均0，任务交集为全部40任务。每验证demo两处anchor，每任务10个验证demo。四条实际父/续训合同的validation IDs与协议相同，只延长监测步数。原PLAN明确 `blind_validation_claimed=false`，保留已探查demo标签（含固定validation demo31），也没有证明它们不曾进入原基础模型训练。因此应称“本轮finetune的demo留出内部监控”，不可称盲测、未见任务或预训练独立heldout。计数、交集及精确索引SHA在 `CONTRACT-AUDIT-v1.json/population`。

四条件是原生渲染2×2：原agentview或world-z orbit +15°，乘compiled renderer headlight baseline或active=0。相机orbit保留wrist姿态与场景灯世界姿态，material变换为identity；不是像素编辑。headlight指renderer camera-attached headlight，不等于所有场景光源；baseline原本inactive时off可以是no-op，源码显式记录 `degenerate_headlight_factor`。四条件在同preaction状态重渲染，并保留相同原expert H10标签。源码依据：`native_conditions`、`headlight_scope`、`camera_scope`；producer依赖SHA已核。

优化器为AdamW β1=.9、β2=.95、ε=1e−8、weight_decay=1e−10，global gradient norm clip=1.0，EMA=.999。原LR为warmup1000、peak2.5e−5、decay_steps10000、floor2.5e−6。20k续训只改总更新数，原LR callable和optimizer计数保留，无重复warmup、无延长cosine。四个真实restore10000 monitor均记录LR=`2.499999936844688e-06`。所以后半程是在原下限上多训练10k；若晚期loss变平，只能支持这个低LR目标上的饱和，不能证明更高LR、不同数据支持或更多更新都无益。源码依据：`adam`、`lr_implementation`、`tail_migration`、`tail_contract`。

| 分支 | 真实job | 已核完成 | 20k包关键源码核验 |
|---|---|---:|---:|
| C0 | `redacted-2d875ec70628` | 20,000 | 20/20 |
| C1 | `redacted-fa4cf052ec4b` | 20,000 | 20/20 |
| T0 | `redacted-ca38f5c92628` | 20,000 | 20/20 |
| T1 | `redacted-5087d3cd8361` | 20,000 | 20/20 |

四条job的source manifest SHA与实际prepared bundle/descriptor相符；80个关键包内源码SHA检查通过。当前研究工作树`gemma.py`已有后续变动，但四个执行包仍是冻结SHA `737d0299…`，执行来源已明确。所引损失/训练/optimizer/transform文件与包内字节一致。全部父commit、合同canonical SHA、source/receipt绝对路径保存在完整JSON，不凭目录名认定完成。

T两seed真实10k请求的 `resume_commit=null`；10k outer commit均记录 `required_step201_cold_restore_completed=true`。正式范围为before-cold 1–201、after-cold 202–10000、tail10001–20000：worker在接受第201次有限更新后写日志并停止，保存该201；supervisor独立BOS恢复201后再继续。原始log逐行连续性仍由总控读取确认。若同step出现不同loss，必须保留attempt/process/source身份并按实际祖先commit判断，不能按step覆盖或选择更低loss；被放弃或未commit的尝试单列。

源码计划每1000global updates监控一次，后半程预期11k…20k。**本子任务实际本地读到的是四条run的restore10k、迁移、完成和15k/20k checkpoint元数据；预期URI不等于各monitor日志已经存在。** `expected_from_source_not_GET_verified_logs`专门标记尚待总控云端读回的日志，不从complete状态推导逐件保存。15k是恢复/过程checkpoint，20k是预先指定终点，不是按loss挑出的最佳checkpoint。

本次未云GET、未恢复模型或大数组、未新增训练/评测/GPU、未写Notion或registry。`SCIENCE-CONTRACT.json`是可嵌入云分析worker的16KB短合同；完整证据保存在另外两个JSON。实际“是否训练到位”需用分开的online/EMA曲线及既有行为结果判断，本审计不预先宣称收敛、过拟合或欠训练。

梯度图标注：`stock_info.grad_norm`为裁剪前原始梯度全局范数（scripts/train.py:733–735）；`optimizer_grad_norm`也在裁剪前，只是经过可选sanitize，本C/T的sanitize=False，故口径相同。实际clip在optimizer.py:81–85，阈值1.0后接AdamW。`optimizer_clip_fraction`是每步是否超过阈值的0/1指示，不是梯度元素被裁剪比例。accepted update s的`LR_before_update`为schedule(s−1)；monitor记录的LR为schedule(S)。

即使已有结果显示T20比C20好，它支持的也是“同一示教、相同batch与更新预算下，扩大示教内时间覆盖可能更有效”；它不能被改称对抗训练、环境响应引导训练或响应子空间训练的收益。两组的原始权重artifact、initialization_seed=0、normalization和前缀冻结范围均由实际父/续训合同与绑定执行源码一致固定；training_seed0/1改变的是训练随机数与排序，不能将两个seed当作两批新数据。
