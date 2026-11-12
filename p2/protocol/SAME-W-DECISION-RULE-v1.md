# Same signed W：科学判据与资格候选

2026-10-19 · science_audit。状态：**候选、未提交；真实prefix质量CPU结果在本审查中未知；same-W新图、setter、精度资格均未执行。** 本文件不是云任务单，不新增训练、层/rank网格、行为采集或Goal。仅定义后续技术资格与科学取舍；总控负责精确资源准入和创建。已有v4.1反事实修复与第一篇保持独立。

## 1. CPU结果改变描述，不代替receiver识别

旧NR24已经证明：自然C/L18/0→9完整prefix读取中，modality mass固定早期空间模板的SSE比约0.06027；旧RI24则显示固定E、L9/j5换K时within解释更强。它们的固定变量、图、层和目标不同。当前新增CPU只问：NR中的细分模态质量是否比global/per-position/per-head-position的prefix总质量多保留描述信息。

| 当前CPU实际结果 | 应修改的解释 | 对same-W的含义 |
|---|---|---|
| head-position总质量已接近modality | 收缩“模态细分必要”的描述，优先更粗强度解释；没有等价界则不能称等价 | 不证明相同W之后无状态依赖，也不构成启动GPU的自动理由 |
| modality仍有完整支持的剩余增量 | 保留细分质量的条件重构价值、各根反例和自由度区别 | 不证明Q/K因果来源或receiver增益；仍需真正相同signed W |
| 两者均差、根/位置混合、低能量主导或字段失败 | 限定局部规律或保留未知，先正确读结果 | 不因结果“不漂亮”自动加实验，也不因此否定same-W问题 |

**same-W的独立问题：** 固定物理snapshot、输入prefix、模型、noise源与同一实际flow时间，在两条合法历史产生的生成状态中，逐坐标完全相同的原生pre-gate W增量，是否产生不同的瞬时速度／候选动作增量？已有同E换K和PVP不能隔离这一项。当前CPU结果无论正负都不单独决定该问题的go/no-go。

## 2. 一个预声明技术单位，先固定再看效应

候选资格只用一个已见锚点：`SDRU-CCONF-T06-V00700-I00-PC/q11`。这是技术开发输入，**不是新盲检**。接收输入始终该snapshot的T观测与T-prefix；C侧只用于恢复原pair身份，不把C receiver作为第二状态。

- history A：原none完整生成；history B：同一个原shared实际V编辑E，仅j5施加一次的pulse。相同原显式FP32 noise、输入、模型与solver。
- 候选切点固定 **j6的pre-update x_A/x_B**，层固定L9/index8；实际tau、weak_type、dynamic num_steps=10、dt和mask/positions从生成carry恢复，不手写理想0.4或把num_steps改成4。
- 两状态必须分别由其合法完整历史产生并可复放。若此固定锚点x_A=x_B或所选receiver相同，记`NO_DISTINCT_RECEIVER`，不换q、不改j追效应。
- 不用pulse与persistent的j6作为不同receiver：PVP已实证二者x6与L9读取前状态相同。选j7会引入另一份历史切断合同，本候选不采用。

从每个x_s恢复后，全部未来历史编辑关闭。对s∈{A,B}三种处理：none；同一实际E仅j6作用一次；same signed δW仅L9/j6作用一次。j7–9一律使用同一自然T-prefix，V/W编辑关闭；返回的suffix cache不变成下一步prefix。所有作用以**同state自己的none baseline**为差分参照。不能把B减A，也不能借旧R图或另一个runtime的baseline。

## 3. native BF16增量是执行前置门

PVP实际合同规定W是BF16 `[batch=1,position=10,channel=1024]`，位置在WO之后、gate/residual之前。相同请求δ不保证相同实际增量：`BF16(W_A+δ)−W_A`与`BF16(W_B+δ)−W_B`可能不同。例：round-to-nearest-even下W_A=1、W_B=2、δ=2^-7，前者可增2^-7，后者可舍入为0。此为数值反例，不是本项目已测结果。

必须分开保存：requested δ、各自W_before、各自native W_after、cast后的**实际消费增量**、shape/dtype/typed SHA与精确相等判定。实际δ相等才允许`same_W_qualified=true`。不把同范数、同FP32请求、近似cosine或同一replacement整W当作相同扰动。

两条技术路线只可选定一条后冻结，不能运行中反复换路线直到通过：

1. **首选严格自然剂量检查：** 取A状态same-E在L9/j6的实际`W_E−W_none`为δ_ref。只要δ_ref能在B的native W基线上逐坐标精确实现，就构造两个状态特定BF16目标W'_s=W_s+δ_ref，由native full-W选择器消费。两个target通常不同；共同的是其相对于各自baseline的增量。若任何必要坐标无法实现，保留`NATURAL_DELTA_NOT_COMMONLY_REPRESENTABLE`，不放宽epsilon、不升级模型计算FP32。
2. **仅作为另行冻结的技术诊断候选：** 每坐标求共同可表示增量集合`{u−W_A:u∈BF16_finite}∩{v−W_B:v∈BF16_finite}`。在云CPU以原BF16位模式的精确二进制有理值计算成员关系，构造状态特定native targets，不在模型路径做高精度加法。若要从δ_ref选δ*，选择规则、符号／剂量约束与可接受的变形标准必须先冻结，不能看δv/动作后挑点。δ*=0或大范围量化归零属于no-op/不可辨识；不以“交集总有0”宣布资格通过。改变δ_ref后必须单列方向、每位置剂量和支持变化，它只能说明该人工共同probe的作用，不能冒称与same-E完全匹配。

当前没有新的native additive W路径可直接授予资格。把δ旁路加在`x+y*gate`之后会改变干预位点；把W运算升FP32会改变原生图；均不能作为失败后的默认替代。若未来设计独立native additive路径，须证明实际消费site/dtype和zero/own-target identity后另冻结，本候选不包含它。

增量相等的判定应以BF16 before/after位模式对应的精确差为依据；FP32/FP64统计副本可作误差诊断，但不能仅凭一次有限精度相减相等就把被舍去的差冒充精确相等。不存在从观察结果调出的epsilon门。

## 4. 必须记录的因果边界

同state的none/E/W三臂在目标位点之前，应具有相同x/tau、receiver residual、Q、P、suffix K/V、mask/positions及gate；E只改原prefix V，W只改实际post-WO/pre-gate输出。跨A/B不要求Q/P或residual相同，那正是待研究的合法接收差异。跨A/B必须保同prefix、同实际tau及其time condition；本pi05接口的gate来自time MLP/adaRMS，不能称物理状态门控。

同tau的gate需要实际逐值相等回执，不能仅据源码推定此次运行相同。即便gate相同，BF16乘法和`x+y*gate`的舍入可依基线而异；因此记录W_before/W_used、gate、after-attention residual、post-block H及第一步v。若分歧首先表现为这些原生有限精度效应，应如实限定，不立即讲成学习到的高层“主动补偿”。

主要效应分别为`δv_s^E=v_s^E−v_s^0`、`δv_s^W=v_s^W−v_s^0`（同j6），以及完整剩余生成后`δa_s^E/δa_s^W`。receiver对照是`I_W=δv_B^W−δv_A^W`及末端对应量，不做W与动作跨坐标dot。raw32与adapter7分开；平移、旋转、夹爪分开，不合混合单位范数。first5仍是候选位置，本资格无Env.step、issued或Y。

## 5. 最小资格调用树（候选上界，不是实际回执）

新G†全路径／resume路径须使用同一动态site selector和相同记录结构；改变捕获或setter本身构成新图，旧G★资格不自动继承。下面明确区分完整10步、剩余continuation及单次suffix forward。**新接口还未实现，因此这些是待源码审查逐叶兑现的精确候选调用，不是已经完成或已批准的模型前向数。** 任一门失败停在该叶，已收费调用与未启动叶分别保留，不补warmup或重试。

| 顺序／叶 | 调用边界与数量 | 每次suffix调用 | 每次prefill合同 | 用途 |
|---|---|---:|---:|---|
| Q1-A/B | 原G★ none与pulse各1个完整10步，共2 full | 10 | 2 | 当前runtime技术参照；不是重做旧科学评分 |
| Q2-A/B | 新G†目标site关闭、两历史各1个完整10步，共2 full | 10 | 2 | 与Q1各自同历史逐值比较；形成合法x6与完整尾部参照 |
| Q3-A/B | 新G†从各自x6/tau恢复none，各1 continuation，共2 | 4（j6–9） | 2 | 与Q2各自剩余路径、raw32/adapter7对角资格 |
| Q4-A/B×V/W own-null | 每state各V own-null与W own-target一次，共4 continuation | 4 | 2 | 真正经过目标selector的无效编辑身份，不走stock bypass冒充 |
| Q5-A/B×first/repeat | same-E各state首次与延迟重复，共4 continuation | 4 | 2 | 核实际E相同、生成δ_ref和读取变化；重复不增人口 |
| Q6-A/B | 资格通过后，独立W setter单次suffix比较各1，共2 single forward | 1（只j6） | 1 | 比较相同W target的独立实现与G†第一步；不执行Euler尾部 |
| Q7-A/B×first/repeat | same-W各state首次与延迟重复，共4 continuation | 4 | 2 | 实际增量相等、第一步和末端效应重复性 |
| Q8-A/B | 原Q3 none各1延迟重复，共2 continuation | 4 | 2 | 排除mutable cache/hook残留与调用顺序污染 |

若全部叶运行：4 full、16 continuations、2 single forwards；按本表ABI为106次suffix求值和42次prefix prefill。**不能把22个不同边界统称22次等成本完整模型调用。** 图像编码包含在各次真实prefill准备中；无额外capture forward。若实现把cache移出调用边界、每次重跑clean参考、或另加stock comparator，必须更改逐叶预算后再冻结，不能沿用106/42。首次资格GPU候选1卡，具体墙钟/存储/云队列必须由实现和root intent登记；本文件不冒造已测吞吐。

## 6. Go/no-go的真实证据

| 门 | GO所需实际证据 | 未通过如何解释 |
|---|---|---|
| 现有CPU解释门 | 完整source/数值/support/失败读回，先正确收缩或保留质量解释 | 只影响叙述；不直接证明或否定same-W |
| 资格输入／图门 | 精确snapshot/noise/model/source；Q1↔Q2、Q2尾↔Q3、own-null和延迟重复逐字一致；合法不同receiver与完整原solver | 技术失败或no-distinct，不能记为receiver假说阴性；不调阈值/换例 |
| same-W精度门 | 实际native before/after增量逐坐标相同、非no-op、same site/tau，未来实际selector全off；所选δ路线冻结 | 不相等则不叫same-W；人工probe变形不可接受时不进入确认 |
| 局部可识别门 | 相同实际W下I_W及分项在重复中可分辨，并排除基线错误/历史处理残留；或稳定零作用 | 非零只证明此图／此probe条件异质性；零只限定此支持，不证明全局无receiver作用 |
| 科学确认门 | 未用于定规则的新任务/初态清单、精确调用／失败规则、效应单位和有意义的分辨标准先冻结；作用方向与幅度预测超过相同信息截止的简单反方 | 旧CONF不能重封盲测；仅能解释已见例或只有不可预测异质性时不升级成可外推机制 |

若same-E不同而合格same-W的差异没有可分辨增量，只支持在该支持/剂量中读取侧解释足够；若same-W仍有可靠差异，才支持目标节点以下的条件作用；两者都不能给唯一中介比例。没有预声明的实际等价／有意义效应标准时，不用小p值或接近零宣称普遍等价。

现阶段应继续必要的独立overlay准备与来源审查，等待真实CPU结果再作集中解释；**尚无same-W执行GO、真实资格或科学阳性/阴性结论**。实现证据与真缺口见同目录上一层`analysis/SAME-W-IMPLEMENTATION-GAP-v1.md`。
