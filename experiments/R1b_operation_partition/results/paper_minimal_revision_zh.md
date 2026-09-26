# 论文改句建议：最小修改版

本版取代上一版改句建议；实验结果未变；修改已于 2026-09-26 应用并编译。表中行号以修改前源文件为准。12 个原建议位置中，7 处需修改，5 处保留。目标是保留原句论证结构，仅更新结果值、过时表述及必要定义。

摘要、引言和结论不加入样本量、分组过程、bootstrap 细节或实验时间顺序；不使用 supplementary study、补充实验、R1b 等内部过程称谓。正文用 on the annotated tasks 限定结果范围，24 个任务和每任务 12 条回复只在方法/附录统一说明。这一简短限定不能完全删除，否则会将有限标注样本的估计写成完整任务集结果。

术语：coverage 译为“覆盖率”；estimated at 译为“估计为”。摘要前半句原有的 Under conditionally independent honest labels and a common view 可译为“在诚实标签条件独立且各参与者具有共同视图的假设下”，不扩写为实验流程叙述。主结果仍为答案加运算细化的 0.477，摘要依照原数值精度写 0.48。

| TeX 行号 | 原句/待改片段（中文） | 建议（中文） | 建议（英文/TeX） | 修改评价 |
|---|---|---|---|---|
| [L44](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:44) | 在人工评定的决定性运算粒度下，该覆盖率只能界定在 0.06–0.79 之间。 | 在人工评定的决定性运算粒度下，标注任务上的覆盖率估计为 0.48。 | at the human-judged grain of the decisive operation, coverage is estimated at 0.48 on the annotated tasks. | 只替换摘要后半句，前半句及其独立性、共同视图条件保持原样；按摘要原有精度保留两位小数。用 on the annotated tasks 限定样本，不写实验过程、区间或“补充”。 |
| [L73](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:73) | 人工决定性运算粒度的覆盖率只能界定在 0.06–0.79 之间， | 人工决定性运算粒度下，标注任务上的覆盖率估计为 0.477， | at the human decisive-operation grain, coverage is estimated at $0.477$ on the annotated tasks, | 仅替换过时的“只能给界”及数值。保留原句答案 0.843 和完整集词法 0.004；此处不新增跨样本比值或三粒度同样本比较，配对比较只放结果段。 |
| [L90](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:90) | 我们将两种测量的划分称为答案粒度和词法粒度。 | 我们将这两种划分称为答案粒度和词法粒度。人工决定性运算粒度将归一化答案相同、标注运算相同的回复归为一类。 | We call these the \emph{answer grain} and the \emph{lexical grain}. The human decisive-operation grain groups replies with the same normalised answer and the same annotated operation. | 保留原定义，只补一条必要的粒度定义。分组共识算法不放问题定义段；不引入 R1b 内部实验编号。 |
| [L244](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:244) | 在 n=10、f=3 时，由此得到界 [0.06,0.79]（外侧 95% 界 [0.02,0.87]），对比答案 0.843 和词法 0.004。若每个答案类按相同比例拆分，覆盖率至多为 0.23–0.55。对答案相等应用该估计器得到 [0.67,0.86]，包含精确值 0.843。 | 在 n=10、f=3 时，直接划分标注得到的覆盖率估计为 0.477（95% 单元 bootstrap 区间 [0.319,0.644]）；在相同标注任务上，答案粒度和词法粒度分别为 0.843 和 0.00142。 | At $n=10$, $f=3$, direct partition annotation estimates coverage at $0.477$ (95\% cell-bootstrap interval $[0.319,0.644]$), against $0.843$ at the answer grain and $0.00142$ lexically on the same annotated tasks. | 保留前面的数学关系与界的推导，只将末尾旧界及两项估计器检查换成直接结果。旧界和检查已经在附录，正文不必重复。标题 Bounding certification 改为 Certification；Pairwise rates still bound the certificate 保留。这是结果段唯一一次报告区间；区间的条件范围在附录说明。 |
| [L260](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:260) | 决定性运算粒度只有界、尚未测量：f=3 时的覆盖率区间 [0.06,0.79] 很宽， | 决定性运算粒度的估计限于 24 个标注任务； | The decisive-operation estimates are limited to 24 annotated tasks; | 只替换已过时的“未测量”及旧宽区间；保留后文原有自动仪器局限和逐对标注一致性数据。另将本段 and the coverage bounds it implies 改为 and the partition-based coverage estimates；routes to measuring it 改为 measurement limitations。不增加 ARI、闭包新增对数等过程诊断。 |
| [L263](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:263) | 而人工决定性运算粒度的覆盖率只能界定在 0.06–0.79 之间。 | 而人工决定性运算粒度下，标注任务上的覆盖率估计为 0.477。 | and at the human decisive-operation grain it is estimated at $0.477$ on the annotated tasks. | 只替换结论中的过时短语和数值，保留整个理论结论及其余原句，不再重复实验名称、方法、区间或新的解释。 |
| [L299](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:299) | 诚实标签律是基于约 30 条回复的代入多项分布；表 e5 给出 split-half 敏感性检查，评估的是训练半样本选出的类别，而非全样本经验众数。 | 诚实标签律是基于约 30 条回复的代入多项分布；表 e5 给出 split-half 敏感性检查，评估的是训练半样本选出的类别，而非全样本经验众数。 | The honest law is the plug-in multinomial from about $30$ replies; Table~\ref{tab:e5} reports a split-half sensitivity check, which evaluates the class selected on the training half rather than the full-sample empirical mode. | 保留原句。这段专门描述原有约 30 条回复的认证模拟，人工划分的方法另置附录，无需在此插入另一实验。 |
| [L357](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:357) | 更强的模型在回答什么上趋同，而不是在如何陈述为什么上趋同。 | 更强的模型在回答什么上趋同，而不是在如何陈述为什么上趋同。 | Stronger models converge on \emph{what} to answer, not on how they state \emph{why}. | 保留原句。原文明确限定 how they state why，讨论的是表达形式，且前文已说明 KEY 是词法粒度。仅更新本次结果不需要重写能力讨论。 |
| [L484](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:484) | 这些逐对估计不能恢复划分、p_max 或 h≥3 的 A(h)，但能给认证覆盖率提供界，如下所述。 | 这些逐对估计不能恢复划分、p_max 或 h≥3 的 A(h)，但能给认证覆盖率提供界，如下所述。 | they do not recover a partition, $\pmaxx$ or $A(h)$ for $h\ge3$; they do bound certification coverage, as follows. | 保留原句。其主语是逐对估计，不能恢复划分的数学局限仍然成立，不因直接分组结果而改变。 |
| [L488](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:488) | 在人工答案加运算关系下 n=3f+1、h=2f+1 的指定众数覆盖率：同模型校准对的界、同质答案内估计与答案粒度精确值。 | 在人工答案加运算关系下 n=3f+1、h=2f+1 的指定众数覆盖率：同模型校准对的界、同质答案内估计与答案粒度精确值。 | \caption{Designated-mode coverage at $n=3f+1$ ($h=2f+1$) under the human answer+operation relation: bounds from same-model calibration pairs (95\% outer limits in brackets), the homogeneous within-answer-class estimate, and the exact answer-grain value.} | 保留原图注与表格数字。这张表报告逐对校准导出的界，不能把其中数字改成直接分组点估计；直接划分结果另列一张表。 |
| [L513](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:513) | 两条路线可能使决定性运算粒度可测： | 两条路线可能扩展对决定性运算粒度的测量： | Two routes could extend measurement of the decisive-operation grain: | 只把 make ... measurable 改为 extend measurement，保留原文的两条研究方向、解释和引用。无需另写一段未来实验计划。 |
| [L539](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:539) | 清单第 1 项：粒度及仪器。答案与答案加 KEY；归一化器、词法规则、嵌入与评判模型；遗漏会导致同样回复给出 0.843 或 0.004 的覆盖率。 | 清单第 1 项：粒度及仪器。答案与答案加 KEY；归一化器、词法规则、嵌入与评判模型；遗漏会导致同样回复给出 0.843 或 0.004 的覆盖率。 | (1) grain and instrument & answer vs.\ answer+KEY; normaliser, lexical rule, embedding, judges & the same replies give coverage $0.843$ or $0.004$\\ | 保留原行。0.843 与 0.004 是原有完整共同集的同样本比较，仍然有效；报告清单不必罗列所有实验结果。 |

摘要目标句（保留原句结构）：

> Under conditionally independent honest labels and a common view, the honest mode on three models is certified at the classical budget with mean probability 0.84 even when only final answers are compared; at the human-judged grain of the decisive operation, coverage is estimated at 0.48 on the annotated tasks.

中文：在诚实标签条件独立且各参与者具有共同视图的假设下，三个模型在经典故障预算下的诚实众数认证概率平均为 0.84，即使仅比较最终答案也是如此；在人工评定的决定性运算粒度下，标注任务上的覆盖率估计为 0.48。

L244 段落的数学推导保持原样，标题由 Bounding certification at the decisive-operation grain 改为 Certification at the decisive-operation grain，仅替换表中列出的末尾结果句。附录保留原逐对上下界，另用一段简短方法及结果表交代人工划分。原稿摘要中“even when only final answers are compared”的修辞并非本次结果更新必需修改，故保留；若之后统一润色，可另行评估。

L260 除表中替换外，两项同段措辞同步更新：`and the coverage bounds it implies` → `and the partition-based coverage estimates`；`routes to measuring it` → `measurement limitations`。此前逐对 κ 和未解决比例属于原校准实验，保留；不得将其冒充本次划分标注的一致性。

正文并列比较必须区分两个范围：完整集的词法覆盖率仍为 0.004；相同标注任务上的词法覆盖率为 0.00142。只在 L244 使用后者作同样本比较，其余原有完整集比较保留，不为更新人工值而全篇替换词法数值。

实际应用时，L90 的定义进一步压缩为一句：`The human decisive-operation grain refines answer classes by annotated operation.`，以保持 9 页正文。正文新增表格引用，附录结果表已插入。最终记录见 `grain-iclr15/revisions/r1b_minimal_revision/`。
