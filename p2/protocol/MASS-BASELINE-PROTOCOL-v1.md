# A-20261019-paper2-prefix-mass-baselines-v1

状态：方法预声明、软件开发中，真实数值未运行。既有 CONF 已见；本项是事后诊断，绝不重新称盲确认。源/输入冻结以 cloud-intents 对应 manifest 为准。

## 问题与改变决定的证据

固定同一 query 的自然 prefix 内容差分 D，跨 flow 的读取变化究竟需要区分 main/wrist/language 的质量，还是每个 head 的 prefix 总质量变化已经足够？这是对既有“模态质量重加权”解释的简化检验。当前不建立新预测器，不进入模型或闭环。

10月16日 natural-reading-use-v1 已完成24锚点、8根、4任务的完整四角分析。其 C/L18/0→9/full10 根等权绝对误差与目标能量之比为 mass 0.060270、within 0.700003、目标可见单一 signed gain 0.178872；旧结果和反例全部复用。该 gain 只拟合一个整体标量，尚不能排除不同 head 的总 prefix 质量变化。新增加的三个质量缩放基线补这一缺口，不重跑旧P/V分解。

主竞争解释：不同 head 与动作位置的整体 prefix 读入强度变化足以解释已见优势，模态内比例变化无需另赋机制含义。反方是即使保留每个 head/位置各自的 prefix 质量，模态重分配仍有稳定、可量化的额外重构信息。两者皆是描述层解释；晚期质量是目标可见量，不是事前预测。

## 冻结输入与人口

只用 `jinwen-sru-natural-reading-20261016-v1` 的24个 `analysis/QUERY-00..23.npz` 与其 SOURCE/MANIFEST/RESULT身份链，不恢复原始 P/V、模型或24之外的数据。每根 q11/16/22；8根包括4任务的 C/T 访问父根。父轨迹角色与用于读 D 的 P_C/P_T 条件分别保存。

三层 L1/L9/L18，flow对0→5、5→9、0→9；8 heads，10候选位置。主比较预定 C/L18/0→9/full10，first5/tail5及所有其余条件保留为边界对照，不把432重复比较当独立样本。不存在新独立确认人口。三种时钟：真实物理 query 时间、flow j/τ、chunk 位置 p，不把后5候选位置称执行尾段。

四角记 F00=早质量×早分布、F10=晚质量×早分布、F01=早质量×晚分布、F11=晚质量×晚分布。它们是完整有效 prefix 内容差分在逐head V256读取空间的向量，head坐标拼接计能量，不能相加成W。Gram仅保留这些向量的内积，不能恢复任意 signed 通道向量或WO后的写入。

## 新比较的精确定义

对早/晚时刻 a/b，已有 m[a,h,p,g] 来自原完整softmax，g为 main/wrist/language。令 M[a,h,p]=sum_g m[a,h,p,g]。不把前缀质量重新归一为1，不修正BF16原行质量，不把suffix伪装成第四个固定内容组。

| 比较 | 晚期读取估计 | 新增描述自由度（full10） |
|---|---|---|
| zero-change | F00 | 0 |
| global-window mass | s F00，s=sum(h,p∈window) M[b]/sum(h,p∈window) M[a] | 每窗口1个质量比 |
| per-position mass | s[p] F00[h,p]，s[p]=sum_h M[b,h,p]/sum_h M[a,h,p] | 每位置1个质量比 |
| per-head-position mass | s[h,p] F00[h,p]，s[h,p]=M[b,h,p]/M[a,h,p] | 每head×位置1个质量比 |
| modality mass，旧对照 | F10 | 每head×位置×组的晚期质量，最多240个 |
| within，旧对照 | F01 | 使用晚期真实组内分配 |
| signed scalar oracle，旧参照 | a*F00，a*=<F00,F11>/||F00||²，各窗口独立 | 目标可见拟合1个系数 |

所有估计与F11比较；预测变化为估计−F00，故误差等于估计−F11。质量比不拟合响应或标签。更多质量比与更低SSE不是匹配容量胜利；基线复杂度按上表公开。这些比较也不一定构成误差单调下降的嵌套最优化。

设G[h,p]为F00/F10/F01/F11的4×4 Gram。新缩放SSE为 s²G00+G33−2sG03，变化能量为G00+G33−2G03。按已有 response/clean B3 的 parallel/perpendicular Gram 独立算同一s下的SSE。full空间是主分母；同时保留各part自身能量及part分母比，二者不混。正交内外SSE之和应复原full SSE。MAI signed交叉项从旧Gram复核闭合，仅用于数值账，不新称因果分解。

## 未定义、数值和聚合

早期总质量为零时相应缩放比未定义，即使早晚均零也不能填1。早晚均零的真实读取贡献可唯一确定为零，保留该零贡献但scale仍为NaN/未辨识，并单列ZERO_MASS_NOOP计数与状态；早零晚正的估计不可用。某group模板未定义时保留旧corner mask，不uniform填补。0目标能量时保留SSE与零目标标识，相对误差为null；不得从排名中隐式删例。近零正目标原样保留，主判断看绝对误差、绝对目标和逐案比率，不让大比率独立支配叙述。

scalar oracle只在full空间各窗口拟合一次，所得同一系数用于full与四个投影视图；不在B3/残余分别重拟后拼凑能量。F11对自身的零SSE仅作代数检查，不进入赢家比较。

FP64处理已存原生采集派生Gram；没有新模型精度选择。对舍入级微负二次型仅在明确相对误差界内标记后夹至0，较大负值为数值失败，禁止silent abs。逐字段shape/dtype/payload SHA、wholefile bytes/SHA及原SOURCE身份全部验收。不可用字段、缺文件、身份不符均fail closed，不换案例、不重跑采集。

先逐query对窗口内h,p累加绝对SSE/目标能量，再根内3q等权，再8根等权。任务另显示各自两根；根与任务均是描述分层，只有4任务不造token级CI。主相对量为等root绝对量的比；同时给逐query比率分布及所有根/任务，不能将两种统计量互换。若任一方法缺支持，提供完整缺失表与方法间共同支持比较，不在原计划人口上隐式重新分配权重。

## 如何改变科学判断

报告 head-position 与 modality 的配对绝对SSE差、除以共同full目标能量的差和逐root/task符号。没有预定实际等价界，因此不以小差或p值宣称等价。若head总量已接近旧modality，结论收缩为更粗粒度的读取强度变化；若modality在各根保有额外作用，才保留其额外描述价值。任一结果都不识别Q/K来源、不证明mixed角动作、receiver机制或训练收益。

本项无新超参数、rank、任务、模型、episode或训练。不由CPU结果自动触发GPU。后继 same E/same signed W需独立资格与新确认规则；原反事实修复v4.1六臂保持独立。

## 输出与图

共享数值表只生成一次：逐query和root/task/pop的SSE、目标能量、support、缩放范围、B3内外误差、closure/undefined/舍入记录；附输入及源码identity。图仅从该表渲染，不重读原数组。图意卡见 figures/F-MASS-BASELINE-v1.md。CPU读取、归档/释放由总控准入，准确bytes和deadline写在INTENT；软件测试不计真实结果。
