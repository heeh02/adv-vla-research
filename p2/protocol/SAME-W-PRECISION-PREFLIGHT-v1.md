# Same-W精度前检：固定PVP B参考增量

2026-10-19。**reference、支持和判据冻结；软件候选，未提交、未读真实数组。** 本文件是既有PVP的CPU技术前检，不扩基础prefix-mass任务或已封存postread图任务；不调用模型、仿真、训练，不扫描case/层/flow/方向择优。执行运输需另按本批精确输入和源码登记，本文件不创建云意图或资源预留。

## 1. 为什么现在可先做CPU

基础prefix质量CPU已实际完成，主全空间head-position比值约0.451335324826、modality约0.060270384187，8个root聚合均modality误差更低。这只保留模态细分的局部描述价值，不能直接推出或否定receiver作用。

已保存PVP证据另支持技术前检：同一个query，j6的pulse与persistent拥有相同x6、L9接收侧Q/P/gate/suffix等；两者此刻仅有是否继续施加原E的差异。none的x6则是另一个合法状态。因此可从既有三CALL得到B的实际原生W增量，先检查其在A上的finite BF16可表示性，避免为取得该量先新增模型资格调用。

这里将先前候选的“从A新same-E取δ”明确改为**从既有B取δ**。这是来源复用推动的技术定义变化，不根据模型结果挑更易通过的增量。原same-W执行资格和科学确认仍独立，不能由本CPU结果自动触发GPU或训练。

## 2. 唯一支持、身份与定义

- query：`SDRU-CCONF-T06-V00700-I00-PC/q11`；实际G★来源为成功T06 recovery-v3，receiver始终T，父轨迹角色C不等于C receiver。
- 位点：L9/index8、flow j6 pre-gate/post-WO W；原tensor轴`[5flows,3layers,1batch,10positions,1024channels]`取`[1,1,0,:,:]`，保全10,240坐标。
- A baseline=`baseline_none_1` / PVP-001。
- B baseline=`shared_pulse` / PVP-097；B after=`shared_persistent` / PVP-098。
- 固定`d = W_B_after − W_B_before`，检查`W_A+d`。不换符号、不改reference、不跳过失败坐标、不生成共同格点probe。

`implementation/same_w_precision_v1/INPUTS.json`列6个精确对象，总 **9,151,111 B**：实际PVP-SOURCE 29,761 B、PLANNED 44,783 B、实际input_ref 567,887 B、三个CALL合计8,508,680 B。来源由实际SOURCE JSON的root/query/input_ref/selected_calls绑定，并与原PLANNED实际call_id/role/schema核验；不得从000002或其他文件序号推断身份。原SOURCE SHA为`8c105a44b18fd3bf89c7417071e42866de4248d2132e98e62dfdeff37e87c511`。

继续逐字复用已冻结native_reader与PVP numeric.validate：wholefile和各typed leaf SHA、shape、storage/logical dtype、有限值、G★而非R、三角色、PG/input/noise、实际selectors及原native gates均须过。新增全三臂j6 mask、suffix positions、query_defined、adarms condition、actual time/weak_type/dt与gate相同检查；A/B必须为不同合法x6。未知或不合格不能视为精度阴性。

## 3. Exact dyadic算法

输入必须是原native BF16的uint16位模式；每个finite值都表示为以`2^-133`为单位的Python整数。sign、exponent、fraction从位模式提取：

- exponent=0：magnitude=fraction；保±0位模式。
- 1≤exponent≤254：magnitude=`(128+fraction) << (exponent−1)`。
- 根据sign取正负；exponent=255（Inf/NaN）拒绝。

对每坐标用无界整数算`d=B_after−B_before`和`target=A+d`。逆映射要求target恰是finite BF16值：subnormal范围按整数精确映射；normal必须在相应指数的步长格点上；超过finite最大值`255<<253`拒绝。不使用FP32／FP64近似相等、epsilon、round、clip或饱和。

B的原目标必须由同精确运算重建为原finite B_after；该事实不证明A目标可表示。A目标若存在精确表示也只说明数学上可构造两个状态特定native targets，尚未实际通过W setter消费。

signed zero单列：纯+0↔−0变化的numeric d仍为0，不作为非no-op。保存A/B-before/B-after负零数量、B零符号转换数、target为0及由非零d到0的数量。精确实数加法不唯一决定零的native符号，本模块不生成新target tensor，不偷偷选择一种zero sign后称native资格。

## 4. 输出与判据

唯一输出`precision-result.json`，上限256KiB；不输出W／δ／target大数组。保输入wholefile、full leaf和固定切片SHA；总10,240及每位置1,024完整分母；native参考增量非零／零数量；A目标representable/nonrepresentable/overflow数量；全部不可表示channel位置和overflow位置；零/负零/no-op状态；实际clock和原输入/PG身份。

| 状态 | 固定判据 | 允许解释 |
|---|---|---|
| `PASS_EXACT_REPRESENTABILITY_ONLY` | d有非零坐标，全部10,240个A目标恰为finite BF16，输入与receiver门通过 | 归档G★的该剂量通过精确可表示性前置；不等于新setter或same-W实验通过 |
| `NO_GO_NATURAL_SAME_DELTA_UNREPRESENTABLE` | 至少1个必要A坐标不可表示 | 该固定native增量不能按完全相同逐坐标剂量转移；技术no-go，不叫receiver假说科学阴性 |
| `NO_GO_ZERO_REFERENCE_DELTA` | d全为0，包括仅零符号变换 | 本参考是numeric no-op，没有非零共同扰动可识别 |
| `NO_GO_NO_DISTINCT_RECEIVER` | A/B x6实际相同 | 当前固定支持没有不同receiver；不换例搜索 |
| `INPUT_OR_NUMERIC_FAILURE_NOT_SCIENTIFIC_NEGATIVE` | 身份/字段/codec/时间/完整性/数值实现失败 | 未完成技术分析，保失败／未知，不能按零或不成立填补 |

所有状态均`same_W_setter_qualified=false`、`new_probe_generated=false`、`scientific_receiver_hypothesis_rejected=false`。任何不可表示坐标即严格natural same-delta no-go，不按通过比例挑subset；但完整坐标计数仍保留供理解精度限制。

## 5. 执行与后续界限

API为`adapter.run_stage(stage,out,source_manifest_sha256=...,deadline_epoch=...)`；真实输入入口Linux-only，原数组留云；精确6对象由外层运输只GET一次，模块只消费已stage原件，不拥有SDK/STS/Create/Stop能力。绝对截止、归档、release由后继运输负责，不能将本模块结果作为自行启动GPU的许可。

如果新图或新后端改变W基线，归档可表示性不能自动迁移；仍须新source自己的own-null、actual consumed W_before/W_after、同τ与future-off、重复性／恢复资格。旧persistent的j7–9继续施加，最终动作不能充当“只在j6一次编辑”的末端结果。本前检不补齐这些缺口，不拟合预测器，不重封旧CONF为盲检。
