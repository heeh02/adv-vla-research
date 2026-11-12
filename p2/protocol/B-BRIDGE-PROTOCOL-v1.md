# B侧none／自然d／共同c：原生首步功能桥接

2026-10-19。用户已直接授权本线程运行B侧桥接、先native资格和量化检查、百度CLI；允许多GPU不等于无限卡。**本协议固定科学支持与13个调用边界；此文件形成时仍是开发／合成资格，尚无本协议模型结果。** 本线程负责人唯一创建与收尾，子代理只开发审查；全账户活动／排队／准备合计≤8GPU，仅27w/5x1e或另有准入的JD，不干扰他人。旧源码／结果／第一篇及其他线程不改。

## 1. 问题、对象和不包含的后继

真实共同probe审计给出c相对自然d的范数比0.94857、向量误差23.30%、方向差13.40°，不能当作自然剂量保真。现在只问：**在同一个B计算背景、同图同位点同tau下，精确自然节点增量d与人工共同增量c，分别如何改变当前v6；c是否仍有可解释的自然作用对应？** 不用几何cosine替代功能桥接。

唯一支持：`SDRU-CCONF-T06-V00700-I00-PC/q11`；物理snapshot同旧T06 recovery-v3，query时钟policy env step55／absolute65。三臂模型receiver均为T观测／T-prefix；父轨迹标签PC不等于模型用clean prefix。B是从原同noise合法j5 pulse历史得到的pre-update x6，actual tau为原FP32 `0.3999999165534973`，连同weak_type按原carry恢复。位点L9/index8，post-WO/pre-gate，全10×1024 W。

自然d=`W_persistent(j6,L9)−W_pulse(j6,L9)`；c严格沿已经冻结并实际完成的共同格点规则及同一A/B原件重建。A=none原数据只用于固定c和来源核验，**不在A运行模型**。此次不做A/B receiver检验，不做j7–9编辑作用continuation，不执行任何机器人步、不新episode、不训练、不新增任务/rank/位点/剂量网格。

已有人工probe结果SHA `fcf6c7835d01a4e4eaeb12b4b96938e49d70182dc457c348f6bdc5084ee469b1`；原自然精度结果SHA `25b1c8b47745460277c329aa909e0f3fe3c177cf9fa5e751dd84b8e1313d9b30`。构造器须重现原623不可表示、612缩小仍非零、11归零、candidate nonzero10080等完整身份和精确整数结果；不得为新模型更易通过而改c。

## 2. 固定科学处理

| 处理 | 实际输入到L9 gate前的W | 含义 |
|---|---|---|
| none | B自己的原生W_B | 同state同图无W处理参照 |
| natural_d | 原生BF16目标W_B+d，即已保存/重新资格的persistent j6 W | 在B重现原same-E的局部写入；不是clean策略 |
| common_c | 原生BF16目标W_B+c | 固定人工共同probe的B侧作用 |

c本身不必能被单独BF16表示。模型消费的是状态特定whole-W BF16目标；CPU精确整数核`used−before`等于对应固定d/c，不能先cast c再加W，也不能把requested target或capture字典替换当实际消费。

单次运行所有prefix保持自然T；不再施加V编辑。历史pulse只通过保存的x6携带。`num_steps=10`仍作为原动态ABI传入，仅核原dt和时间，不设成1；**single只计算一次完整suffix forward，不作Euler更新。** 本次无未来日程／末端动作的科学效应；full10仅做下述资格参照。

## 3. 13个精确调用叶与成本

与native开发接口已对齐：`full_inactive`和原G★ full都包含T/C两次原prefill及10次suffix；`single`与`independent_single`每次也明确包含2prefill＋1suffix。所有recording均来自当次forward，不增加隐藏probe／warmup；工厂创建不算模型调用。完整树：**3 full＋10 single＝13调用边界，26 prefill＋40 suffix；Env.step=0，A模型调用=0。** 技术比较与延迟重复也计费，不另算成科学样本。

| 叶ID | 实际入口／处理 | prefill | suffix | 必需依赖／用途 |
|---|---|---:|---:|---|
| REF-01 | 旧冻结G★，原B pulse full10 | 2 | 10 | 当前runtime原模型/input/noise/parity参照，保存actual x6/tau6/W及原末端 |
| REF-02 | 旧冻结G★，原persistent full10 | 2 | 10 | 必须与REF-01同j5历史和x6、j6 L9 receiver/Q/P/gate等相同；核current native d与原归档d |
| CAN-01 | 新W选择器inactive、原B pulse full10 | 2 | 10 | 与REF-01逐原native字段／full状态／端点一致，所有W selector实际关闭 |
| S01 | candidate single，none_1 | 2 | 1 | 从同B实际x6/tau恢复；逐native与CAN-01/REF-01的j6对应；不更新Euler |
| S02 | candidate single，active own-target W_B | 2 | 1 | 真正经过新selector的identity，必须与S01全部被核字段/速度一致 |
| S03 | independent single，none | 2 | 1 | 独立site实现identity，须与S01一致；不能调用主setter函数自证 |
| S04 | independent single，natural_d | 2 | 1 | 完成前序技术门后读取固定目标，actual消费及输出作为独立实现参照；同j6还须匹配REF-02自然E下游 |
| S05 | independent single，common_c | 2 | 1 | 固定c目标独立参照；无依据输出选target |
| S06 | candidate single，natural_d_1 | 2 | 1 | 与S04及REF-02的j6实际W消费和下游对应逐字一致，方可用作d科学效应 |
| S07 | candidate single，common_c_1 | 2 | 1 | 与S05独立参照逐字一致，方可用作c科学效应 |
| S08 | candidate single，none_2延迟重复 | 2 | 1 | 与S01字节一致，排除处理残留／顺序污染 |
| S09 | candidate single，natural_d_2延迟重复 | 2 | 1 | 与S06字节一致 |
| S10 | candidate single，common_c_2延迟重复 | 2 | 1 | 与S07字节一致 |

REF-02是必要的当前runtime自然E comparator；不能以REF-01 pulse通过而推未运行persistent也必通过。两full是新图/source资格，不重算旧PVP科学评分或重做旧完整人口。

预算分stage：R=REF-01/02，4P/20S；C=CAN-01/S01…S10，22P/20S。任何额外调用、单步中隐式clean参考、prefill外置重算、独立capture或失败后重试都必须先更新调用账本；本协议不包含这些额外工作。

## 4. 两独立GPU阶段与实际依赖

架构允许R旧图资格与C新inactive full的独立准备并行，每stage至多1GPU，合计至多2GPU，账户总门仍8。**本次root为减少跨GPU等待与归档握手复杂度，选R先执行并归档/SHA冻结，C再准入执行；不为等待gate空占另一卡。** 科学叶数不因顺序改变，未执行叶不当失败或可随意补跑。

R通过输入/模型/source身份、旧pulse和persistent当前native parity、同receiver/E/实际d门后，输出小验收＋精确B x6/tau、目标及实际reference trace引用；归档完整SHA与provider/instance释放独立核。C只消费root接纳的同一R精确结果，不从目录“看起来有输出”自行GO。C首次先做CAN-01/S01/S02/S03资格，过门再进行S04/S05独立有处理参照与S06/S07；最后S08…10验证延迟重复。

每个stage在Create前登记独占run/queue/resource/source/input SHA、所有调用叶、科学／归档／退出绝对截止。总新输出上限**2GiB跨R/C分账**，不是每卡2GiB；root运输明确分配及临时盘预算后方可提交。预留不能按两个线程各8GPU；只持有本阶段实际资格卡，失败／完成及时归档并核释放，不碰他人作业。源码和输入冻结原件保持不可变；技术失败不通过换硬件／放宽精度择优。

## 5. 不可跳过的native门

1. **源与输入：** 同固定snapshot/prompt/编码、noise完整字节、decoder state（仅身份，本轮single不decode）、模型参数/source/runtime ABI；原PG/prefix mask/positions和实际KT/VT/E保持来源。B来自合法pulse轨迹，不插值生成state、不换query。
2. **参考当前parity：** REF-01/02分别核旧source相应记录；full G★之间同j5、x6与j6 L9 receiver应按旧合同exact。原d必须在当前reference重现；若不同，不能用旧target冒充current natural d。
3. **新图inactive与恢复：** CAN-01与REF-01比较共有真实PG、flow x/v/time/weak/dt、三代表层记录和raw32末端；S01实际x6/tau/weak/dt匹配full carry且suffix v6/相关中间量exact。新trace增加字段不自动继承parity。
4. **own-target及独立实现：** S02选target=W_B必须通过actual selector并identity；S03独立实现none一致；S06/07分别与S04/05对相同实际target一致。S06的natural d还须匹配REF-02 j6下游W/y/H/v；不要求它的V输入与REF-02相同，因为此门正是在W节点替代natural E。
5. **target消费：** candidate所有W处理臂目标之前的residual/Q/P/suffixKV/gate/mask/x/time等必须与其S01 baseline一致，非L9桥selector为false；实际`bridge_W_used`应与固定BF16 target逐坐标字节一致，实际增量由exact dyadic复核。记录`bridge_W_before`与`bridge_W_used`，不沿旧capture字段名误把before当used。
6. **延迟重复：** S08/09/10对对应首轮的native outputs、carry身份与source exact。合法真实零效应保留；zero-target/零能量比率为null，无效或未知不填0。

所有严格门失败保留已charge、已生成、已归档和未启动叶；不改成科学阴性，不运行余下科学叶，不隐式retry。字节parity失败不通过epsilon、升FP32、改solver或删尾坐标修成PASS。

## 6. 科学读出：仅v6，明确单位

主要三组signed差：`Δv_d=v_d−v_none`、`Δv_c=v_c−v_none`、`v_c−v_d`，全10候选位置逐坐标保留。模型输出native BF16 raw32分开报告：索引0:3、3:6、6分别是平移相关、旋转相关、夹爪相关模型通道；7:32作为padding另列，不混入实控解释。每组按first5/tail5/full10保signed向量、零／反号支持、能量、RMS、c/d效应范数比、relative bridge error、cosine及零分母。

**覆盖此前“首步adapter7”模糊说法：本次single没有完整生成终点，不报告adapter7动作、已发命令、运动或任务结局。** v6前7个通道不是已经经过decoder的adapter7，不冒称米/弧度。full10 raw32只用资格，不拿原persistent的最终动作代替单次j6未来关闭的科学结果。

不因c的W范数只减5.14%而预设其v作用接近；不混七维单位形成一个“好坏分数”。两个technical repeats不是独立样本；一个已见query也没有跨任务置信区间或盲检身份。

## 7. 量化读出和最强简单反方

只消费真实保存的`W_used`、gate、receiver residual r、after-attention residual y，保full10×1024及位置/窗口。BF16位模式转换为`2^-133`整数单位，理想乘积为`2^-266`，给精确combined关系：

`Δy_actual − g·ΔW = [y_arm−(r+W_arm·g)] − [y_none−(r+W_none·g)]`。

这些是实际native叶与理想算术的差值描述，不分配“量化因果百分比”。不增加gW捕获、不重写原`x+y*gate`以改变compiled graph；未观察到真实被消费的product时，不能分开认领乘法和加法份额。

在原B pulse/persistent与新none/d/c上，预声明两种CPU候选预测逐坐标比实际y：

- 理想单次舍入：`RN16(r+Wg)`，其中乘加先按精确dyadic计算。
- 分步舍入：`RN16(r+RN16(Wg))`。

RN-even用整数完成，保signed zero、次正规、finite overflow（不饱和）及全部mismatch坐标。**实际GPU可能先FP32 FMA再cast BF16，该双舍入候选不在本两模型内。** 两模型都匹配时路径不识别；仅一模型匹配只说明它在所测数组上匹配、另一模型不匹配，不认定唯一compiled path。两者均不匹配仍报告combined真实关系，不追加精度／算子网格或为了归因改模型。

同tau的gate实际已核相同，不把舍入影响写成物理state gating；相同W变化在native gate乘法／residual加法中可能吸收或改变有效差。`analysis/quantization.py`提供上述纯CPU读出，可在GPU科学进程结束后同一实例CPU阶段运行，不需要额外模型调用或单独CPU作业。

## 8. 验收、解释与后继决定

技术完成要求全部13叶按各阶段实际执行/归档，所有必须native门通过，无未知被填PASS；科学结果需量化读出和signed v桥接同source。在某stage资格失败时，全树不complete，但已产生reference和失败证据保留。数据归档、科学complete、job终态、instance释放分别记录。

- 若c与d的v6作用明显不同或反号，只能降低该人工probe作为自然作用替身的价值；不为寻求A/B阳性继续自动扩展，也不称c“有害”。
- 若B侧作用对应且简单量化解释可核，说明此probe在该B工作点有局部功能桥，才由root根据完整结果判断是否值得A/B；不自动GO，不证明A侧自然same-E也对应。
- 若二者效应都零／很小，保实际分辨与精度限制；不能说读取足够或receiver全局无效。若只有量化候选匹配，仍不是机制因果定位。
- 非零到最低有效位不等于有意义；目前未预设功能等价界，故不输出“等价PASS”。后续若需要等价主张，须另冻结有科学依据的标准。

旧“same-W零差异则读取充分”的推断继续撤回。本次B-only没有I(c;A,B)的实际计算，不可写成跨state receiver实验。第一篇／第三篇、原v4.1修复和训练都不由本结果自动改变。

## 9. 实现接口和合成验证

`implementation/b_bridge_v1/native/query.py`：`full_inactive`、`single`、`independent_single`、`restore_time`、`wait`；single真实输出W字段`reads.bridge_W_before/bridge_W_used`形状`[3,1,10,1024]`，active形状`[3]`；取代表层slot1、batch0，gate取`[1,0]`为`[1,1024]`。full对应多一层j5–9轴。

`analysis/acceptance.py`提供`extract_site`（旧full需明确legacy_reference，其他实际selector不能伪造）、`validate_site`、`require_exact_fields`与`velocity_readout`。within-candidate source身份必须相同；旧／新图parity另按各自精确source身份比较共有实际字段，不把两图source hash强行改相等。

`analysis/quantization.py`提供`analyze_native_triplet`和`analyze_rounding_candidates`，无网络或模型。对应测试目前12项通过，覆盖全部finite BF16往返、RN-even两符号tie/subnormal/overflow、fused-vs-split精确反例、零号／路径不可识别、actual site/own-target/mask/time/source负例及signed v分组。合成测试不替代本次真实native资格。最终source manifest由统一执行包绑定，绝对资源／输出分账由root在提交前固定。
