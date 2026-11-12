# 本轮直接核读的相关方法边界

2026-10-19。这是本负责人直接打开的一手资料与设计判断，未把交接 review 当论文原文。当前只读取与首项分析和后继干预有关的章节，不声称完成穷尽综述。

Kobayashi 等，EMNLP 2020《Attention is Not Only a Weight》，已核正文§2.1–3，特别是式(1)–(4)：输出是attention权重乘经过V/WO变换的内容向量再求和；P本身不足以代表输出。因此我们不把P/V区分或普通线性分解列作新贡献。[正式全文](https://aclanthology.org/2020.emnlp-main.574.pdf)。

《VLA-Trace》v2，已核§3.2与附录A.3：用softmax前的logit mask做模态路径消融，并分全层、单层、窗口和prefill/生成期；该方法改变允许路径。我们的质量基线仅分解原已存读取，不是路径必要性干预，也不能从其结果认领同样因果结论。[全文](https://arxiv.org/html/2605.30117v2)。

《Not All Features Are Created Equal》作者当前47页PDF，已核§4.7、附录C.3与O：已有等范数随机方向、真机激活转移及matched initial flow noise讨论；附录O明确披露模拟driver未完整控制初始噪声项，而实机记录并复用了该噪声。same-W候选应保持自己的同snapshot、同初始noise、同τ和合法接收历史，不以“任意激活能改变动作”作为新意。[本次作者全文](https://cwru-aism.github.io/vla-interp-page/paper.pdf)。未把作者当前PDF擅自当作新的arXiv版本。

本轮新增质量基线本身是一个必要反方检查，尚不是第二篇独立贡献。科学价值取决于其是否收缩解释，并为后继可预测的same-signed-W条件作用提供清楚的比较对象；不能由软件完工或数据再分析数量认领论文完成。
