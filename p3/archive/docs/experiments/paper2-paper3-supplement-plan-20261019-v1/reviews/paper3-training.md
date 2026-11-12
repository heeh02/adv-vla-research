# 第三篇训练线：证据审查与最小补实验建议

审查者：独立训练研究子代理。2026-10-19。范围：只读本地合同、小源码、既有标量摘要与CSV；未加载模型/大数组，未发云任务、未改已有实验/论文、未写Notion。这里的补实验均为设计建议，不是提交授权。当前有效授权的训练范围仍是已完成的T20接续及既定行为验证；下述新U/L或新训练seed需另行冻结与准入。

**建议先把第三篇定位为“时间采样策略、条件选择与优化预算如何影响微调后的行为”。先闭合T30，再决定是否补齐已有2×3训练对照的四个缺失run；现在不宜直接启动机制指导的对抗训练方法。** 现有证据支持训练方案间的拟合/留出差异，但没有证明超过原模型的稳定鲁棒性收益，也没有单独识别“时间覆盖广度”的机制。

## 1. 实际训练是什么

一个样本由示范时刻j确定：从`states[j]`恢复动作前状态，重渲染该时刻主/腕相机各一张图，加任务指令，监督为`actions[j:j+10]`。这是**单时刻观测→H10动作chunk**，不是输入10帧，也不是整段示范或视频。数据字段中的8维proprio/state没有在本批`discrete_state_input=False`配置中进入state-token模型路径。物理j、flow时间t、chunk动作位置须分开。[S1,S2]

共同源为原LIBERO四suite、40任务、1600条训练示范；内部监控是同40任务的另400条示范/800状态。Plus是训练后行为评测，不是这批训练数据。内部示范虽未参与本轮优化，却用于过研究决策，且未证明基础模型从未见过，不能称盲测或未见任务。[S1]

| 臂 | 物理时刻支持 | 每个状态两行 | 实际改变的因素 |
|---|---|---|---|
| C | 每demo两个固定分层中心，约有效起点的1/4、3/4 | clean+clean | 原条件选择实验的clean对照 |
| U | 与C相同的固定两点 | clean+随机非clean | 三个原生非clean条件的均匀hash选择 |
| L | 与C相同的固定两点 | clean+高loss非clean | 每epoch起点在线模型评分，三候选取最大 |
| T | 同demo、同早晚半区，每epoch轮换j；首轮保留C两点 | clean+clean | 冻结的时间重采样策略 |

U/L候选是`C15_Hbase/C0_Hoff/C15_Hoff`。L评分四格但clean不参加第二行竞争；评分是原H10×D32监督flow loss，使用在线参数而非EMA，同anchor四格共用评分key/标签，每epoch换key。L没有增加高loss task/anchor的抽样次数，也没有PGD或响应/attention评分。源码实核`multistep.py:104–145`、`selection_rules.py`、`cursor.py:32–46`、`runtime.py:158–175`；不能仅按“高loss”将它等同于已实现的梯度对抗训练。[S3]

共同模型为非LoRA π0.5-LIBERO；冻结视觉/语言前缀32叶、2,923,335,408参数，训练动作专家及action/time投影19叶、430,098,464参数。动作先按原quantile归一化，7维补零至32维；全部H10真实动作位置valid。普通stock图像增强、随机flow噪声和时间仍在。B32=16物理状态×2行，不是32独立示范。[S1,S2]

AdamW β=(.9,.95)、ε=1e−8、weight decay=1e−10、clip=1、EMA=.999，无梯度累计；前1000更新warmup至2.5e−5，10k附近降到2.5e−6，10k–20k维持下限。10k=50逻辑epoch=320k行，20k=100逻辑epoch=640k行。full-state接续保留optimizer、EMA、RNG、global step和cursor。损失为flow velocity MSE而非生成动作误差，`L32=(7L7+25L25)/32`；25维padding的目标仍含噪声。0.001不能单独作停止线或选checkpoint规则。[S1]

到20k，C只有3200个独特demo-j，每个选中100次；T有155832个demo-j，其中151664个选中2次、4168个选中4次，平均2.05次。T每50轮才覆盖一次冻结支持，20k第二段重放该支持。邻接H10重叠，不是15.58万独立示范。等更新比较是公平的预算对照，但不是等每点曝光或等优化充分性。[S4]

## 2. 已有曲线与行为支持到哪里

### 10k/20k

| 分支 | 末1000更新训练loss中位数，10k→20k | 共同基线环境EMA留出loss，10k→20k |
|---|---|---|
| C s0 | .001776→.001240 | .027489→.031017 |
| C s1 | .001665→.001305 | .027965→.031542 |
| T s0 | .011411→.011077 | .012256→.012404 |
| T s1 | .011311→.010978 | .012234→.012357 |

C拟合继续变好而留出恶化；T共同留出更低，但10k→20k仍略升。C/T在线loss来自不同支持，且online与EMA不同视图，不能直接相减为泛化gap。T20<C20在四格两seed的全部40个任务均值成立，但并非每个demo/状态成立。[S1,S4]

原C的固定EMA诊断已排除“退步完全由padding/未执行尾部造成”：C10内部前5×7监督误差也明显恶化，padding仅解释该次总loss增量约0.5%–0.6%。这不能补成T20或T30的生成动作/成功率证据。[S5]

### T20→T30：已训练与已可分析分开

`CURRENT.json`本地记录更新于2026-10-19 04:07:35 UTC：四条run均实际到达30000。floor-s0/reheat-s1有完整30k状态与曲线；floor-s1/reheat-s0归档尾部失败，后来只恢复出可供推理的EMA，不具备完整状态恢复资格。不得把它们写成未训练，也不得补写缺失曲线。[S6]

四条同seed分叉使用相同T20全状态和原50轮物理支持，新增10k更新。floor=2.5e−6；reheat=500更新升至2.5e−5，再9500更新cosine回2.5e−6。这识别的是**给定T20历史后的整段未来LR路径**，不是分别识别峰值、warmup、累计步长，也不是从原模型开始采用新日程的效果。[S6]

现有小表明确显示：[S7]

- 同seed的25k固定训练probe（EMA、前5×7）中，reheat在C_old与T_added两支持上均比floor更低；T_added差分别为−.002710/−.001781（s0/s1）。但25k内部基线环境，reheat−floor为+.000176（s0，更差）与−.0000075（s1，微小更好）；其余条件也非一致改善。
- 到30k只有floor-s0与reheat-s1完整曲线。相对各自restored20k，前者基线内部loss升1.57%（四格1.57%–2.65%），后者升6.47%（四格6.47%–10.41%）；两者训练probe继续降低。**不能把这两条跨seed曲线相减当作LR效应。**
- 末1000更新online loss中位数为floor-s0=.010570、reheat-s1=.009547；不是全四分支比较，也没有达到或证明必须达到0.001。
- 原T20与restored20k标量存在可测数值差异，分别保留；当前摘要的`paired_input_order=[]`不能冒称已做四run全程实际输入序列逐条同一验证。合同匹配与实测匹配是两个证据层级。

因此有“还能提高训练支持拟合”的证据；没有“更充分优化改善内部泛化/行为”的闭合证据，更没有全局收敛结论。

### 行为基线不可省略

Plus每模型固定10030案例；以下为缺失结局二元补全界，**不是置信区间**，也不是已排除runtime/q0差异的训练因果效应：[S8]

| 模型 | 闭合/计划 | 成功 | 缺失 | 总体补全界 |
|---|---:|---:|---:|---:|
| Original | 9907/10030 | 8346 | 123 | 83.21%–84.44% |
| C20 s0 | 9708/10030 | 7154 | 322 | 71.33%–74.54% |
| C20 s1 | 9720/10030 | 7198 | 310 | 71.76%–74.86% |
| T20 s0 | 9577/10030 | 7607 | 453 | 75.84%–80.36% |
| T20 s1 | 9295/10030 | 7455 | 735 | 74.33%–81.66% |

T20可局部优于C20，却仍低于原模型端点界限；只击败C不能声称新鲁棒性收益。T30当前仅floor-s0的Plus ordinal0–127闭合，128/128成功；不是随机代表样本、不是10030总体成功率、不是四分支胜负。[S6,S9]

U/L不能从叙事中消失。完整原LIBERO-10历史面板每模型300例：Original266、U0/U1=248/238、L0/L1=249/253；L−U净收益仅+1与+15，且全部低于Original。完整全40 C/U面板每模型1200例：Original1139，C0/C1=1071/1092、U0/U1=1081/1074；U−C一正一负。它们是自定义六条件面板，非标准clean500或Plus。[S10] Plus U/L在本次读到的10月17日清单仍各缺约2.8k–3.0k，属于有时间戳的历史覆盖，不当作今日实时结局；本次未拿完成子集排名。[S11]

## 3. 阻止第三篇独立结论的具体缺口

1. **采样处理的组成尚未拆开。** T同时改变物理状态、H10标签、阶段组成、重复率、窗口重叠及梯度相关性。这些是T策略的组成，不是隐藏超参违规；但它们阻止把全部效应单独归因为“更多帧/覆盖广度/低维流形”。等总曝光与等独特样本曝光不能在支持大小不同且固定预算时同时成立。
2. **现有C/U/L/T不是完整因素矩阵。** U/L只在固定时间支持上检验，T只在clean上检验；T与L直接比会同时改变时间采样和条件选择。T30只检验T的后续LR，不能代替U/L的充分优化或时间×选择器交互。
3. **L−U不是纯选择信息。** 实际L同时改变条件边际频率、跨epoch持续性及条件与状态匹配；评分开销也更大。L并未多采困难任务。早期bank的最大分数优势部分由argmax构造保证，不能当学习收益。[S3]
4. **指标与行为链条未闭合。** H5/H10监督残差不是生成chunk、issued前缀或closed-loop结局；固定demo probe不是评测episode中的同案状态。旧运行q0、collector、云/精度等差异必须保留，技术缺失不能算False，不能通过重跑正常结局来消除不利结果。
5. **重复层级有限。** 两训练seed共用split和一张T时间表；40任务、大量帧与重复条件不能当独立训练seed。验证已用于研究选择，需要新的确认人口才能称确认性结论。当前单模型/单配方结果不能推出一般VLA训练法则。

## 4. 最小补实验：分阶段，不一次开大网格

### A. 当前优先：关闭已存在的优化问题，不新增训练

以预定T30四个EMA为正式端点，T20及Original为必须基线，完成既定LIBERO-10/Plus清单；25k只作预先保存的过程诊断，不因其loss较好替换30k主终点。先接回已有结果、核现有可恢复标量；floor-s1/reheat-s0的30k缺失monitor/probe若不能从既存对象恢复，则保留缺失。如后续确需一次固定诊断，应使用已恢复EMA与原固定输入/key、另记新测量，不能声称补回原online日志。训练无需重跑。

主比较为每seed内`T30-floor−T20`、`T30-reheat−T20`与`T30-reheat−T30-floor`；另分别相对Original。两seed分别报告，禁止只取最佳LR×seed。当前恢复EMA可以用于行为，但不能继续训练时假装optimizer也完备。

**停止扩张条件：** 若训练拟合改善、内部留出恶化且行为不优于T20/Original，应接受“此接续不能修复行为”的结果；不追0.001、不追加LR网格。此结果仍有训练诊断价值，但未必足以独立成篇。

### B. 若第三篇走“时间采样×条件选择”路线，最小新训练是四个run

采用已经定义且有匹配旧对照的**10k原日程**，补齐下面2×3矩阵；这回答等更新预算下的策略与交互，不宣称每组已达到最优。

| 时间采样 / 条件行 | clean+clean | clean+uniform native | clean+loss-selected native |
|---|---|---|---|
| 固定两点 | 已有C10 s0/s1 | 已有U10 s0/s1 | 已有L10 s0/s1 |
| 冻结T轮换 | 已有T10 s0/s1 | **新增T-U10 s0/s1** | **新增T-L10 s0/s1** |

新增仅4×10000=40000优化更新；至少保留Original共享基线。固定原权重、1600demo、T同一时间表、每epoch16slot×2行批量、global-key规则、loss、归一化、dtype、参数范围、EMA和LR。T-U/T-L需要在**同一j**生成与原C/U/L相同的原生三非clean条件；现有T主要是clean缓存，不能声称这些条件已全部存在，必须先给出真实可复用清单、新渲染/字节预算及同状态标签资格。不能借Plus语义/机器人初态变化沿用不合法的原expert标签。

三个预声明效应：时间策略在各条件行上的差；L−U在两时间支持上的差；`(T-L−T-U)−(L−U)`的交互。这样不把时间覆盖与选择信息混成同一因素。旧10k证据若与新runtime不能满足同一数值合同，只能保留历史对照，不能强称严格配对；需要先资格而非悄悄重训旧臂。

这4个run是**等更新**，不是等GPU时间：L每epoch多做3200×4个逻辑B1评分，10k约640k额外评分行；U无这项评分。报告优化行数、评分调用、GPU时、数据生产成本。只有要声称计算效率时，才另冻结等计算预算的U对照；不以无用评分假装公平。

不要把T30选出的LR只给新T-U/T-L、再与旧U/L10比较。若要主张“各组充分优化后的优劣”，必须在相同开发预算下对各组使用同一候选LR集合/截止和验证规则，再用保留行为集；那是另一项更大研究，不能由此最小矩阵冒领。

确认阶段只对开发阶段保留的**一个主对比**增加预声明的新训练seed/独立时间排程及未用于选择的评测人口，并完整报告；不为了跨显著性门追加seed。两seed的已有证据保持开发/探索身份。

### C. 只有第二篇交付了可外推功能量，才讨论机制指导方法

先验证机制评分在留出任务/状态上对实际动作作用有增量，强比较包括监督loss、响应范数、阶段和简单幅度解释。通过后，在**固定同一T时间支持**上做候选M；它可以是选择、加权或有界正则，不预设必须PGD/对抗训练。

最小方法表保留`Original、T-clean、T-U、T-L、T-M、T-M-shuffle`，其中前四项在完全匹配合同下复用；新增M和shuffle才检验信息增量。shuffle优先在同task/预定时间层内置换完整demo条件历史，保持每epoch条件数量和跨epoch持续性、破坏状态匹配。若用冻结历史回放，该结果只识别这条历史的分配信息，不能自动等同在线自适应算法效应。M不得同时扩大时刻支持、改变可训练模块或新增loss再把全收益归于机制。

如果第二篇机制与M不可分，合成一篇更强的机制+方法论文优于为了第三篇硬拆。

## 5. 成功率评价与成篇判据

- **人口和端点先冻结。** 保留Original、可靠clean时间基线与同条件U/L。既有10k/20k/30k端点照原合同报告；新矩阵用共同10k正式终点，不能按训练loss/Plus成绩选checkpoint。旧成功、正常失败不重跑；剩余先按started/technical/never-started分类。
- **配对到真实输入。** case key应含task/variant、init/env seed、noise规则、实际暖机后q0/首输入、prompt、renderer/源与checkpoint身份。匹配的是初始比较条件；后继观测随各策略自然分叉。跨云未过原阈值时给单端点表/缺失补全界，不强拼训练因果效应，不放宽21例桥资格。
- **主结果为闭环成功与配对代价。** 每seed给success/failure/technical/missing、gain/harm/recovery和净Δ；同时给clean保持、扰动绝对SR及各官方类别。clean相对差中差只作辅助，不能把clean退步导致的相对数值叫鲁棒性提升。物理/语言语义变化与纯视觉条件分层；no-op headlight保留真实身份。
- **不确定性分层。** episode结果按共享初态/任务成簇，必要时suite内task bootstrap；训练seed独立列出，任务区间不得宣称包含seed总体变异。缺失补全界与统计区间分开。未完成人口不按完成子集排名，不把128/128外推为总体100%。
- **固定诊断回答原因，不能替代SR。** 同checkpoint分别测训练旧点、T新增点、内部demo留出；online/EMA分开，真实7维/padding25维、前5/H10和flow分项分开。生成动作误差、issued动作和成功率各有自己的证据人口。

第三篇若主张方法改善，应在预声明主总体相对**Original与强时间基线**有可复现收益，且clean保持/损害有明确边界；仅L>C或M>C不足。若主张训练分析，则允许阴性结论，但应由完整因素比较、固定预算与确认性复核得到超过“某组更好”的可检验原则。当前证据还没有达到这两个门槛。

## 证据索引

主checkout相对路径基于`/Users/researcher/Adv-VLA-π0.5`；研究树根为`/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5`，以下N表示其`docs/experiments/analysis-system-20260929/native-write-flow-20261006-v1/`。

- S1：主checkout `docs/experiments/existing-data-synthesis-20261018-v1/training-convergence-v1/REPORT.md`、`contract/CONTRACT-AUDIT-v1.md`、`contract/DATA-BENCHMARK-ADDENDUM-v1.md`。
- S2：同目录 `contract/CAUSAL-INTERPRETATION-REVIEW-v1.md`、`contract/SOURCE-LINES-v1.json`；实读N下`full40-training-reader-v1/payload.py:71–120`、`training-ul-execution-candidate-v1/frozen-multistep-v2/qualified-v1/ae_profile.py:11–82`、`temporal-support-clean-candidate-v1/formal-training-integration-candidate-v1/contract.py`。
- S3：N下`continuous-cul-trainer-v1/{protocol,cursor,runtime}.py`、`training-ul-execution-candidate-v1/frozen-multistep-v2/{selection_rules,multistep}.py`及`training-exposure-analysis-20261008-v1/REPORT-v1.md`。后者明确为早期bank的历史覆盖。
- S4：主checkout `docs/experiments/existing-data-synthesis-20261018-v1/training-convergence-v1/reassessment-0p001-v1/REPORT.md`与`INTERPRETATION.md`。
- S5：N下`cul-fixed-view-objective-diagnostic-20261008-v1/result-analysis-v1/REPORT-v1.md`。
- S6：主checkout `docs/experiments/t20-optimization-continuation-20261019-v1/{CURRENT.json,PLAN.md,SCIENTIFIC_DECISION-v1.json}`。CURRENT是执行回执路由；PLAN/DECISION内“preparing”历史字段不覆盖最新实际结果。
- S7：同目录 `curve-analysis-v1/technical-v2/ACTUAL-RESULT-v2.json`及`prepared-v2/run-v1/readback/science/{SUMMARY-v1.json,INTERNAL-CURVES.csv,PROBE-CHANGES.csv,PROBE-BRANCH-CONTRASTS.csv,TRAINING-WINDOWS.csv}`。本审查仅从小表摘录和作简单百分比换算，没有重算原始数组。
- S8：主checkout `docs/experiments/existing-data-synthesis-20261018-v1/behavior/actual-complete-v1/SINGLE-ENDPOINT-BASELINE-v2.json`及`SUMMARY-v1.json`。
- S9：主checkout `docs/experiments/t20-optimization-continuation-20261019-v1/evaluation/results/floor-s0-plus-u064-v1/ADOPTION-v1.json`，结合CURRENT里的前128例汇总。
- S10：N下`cul-final-L10-complete-analysis-20261009-v1/REPORT-v1.md`、`cul-final-CU-complete1200-analysis-20261009-v1/REPORT-v1.md`。
- S11：研究树 `docs/experiments/all-experiments-closeout-20261017-v1/inventory/CLOSEOUT-INVENTORY-v1.md`；10月17日历史覆盖，不是实时provider状态。
