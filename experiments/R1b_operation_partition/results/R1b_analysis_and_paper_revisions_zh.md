# R1b 标注实验结果分析与论文修改建议

完成日期：2026-09-26。论文基准：[iclr-15.tex](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex)。**没有修改论文源文件或三个原始标注文件**；下面是可审阅的修改建议。文档中的标注流程被用于理解实验定义，未被当作新的用户操作请求。

## 1. 已完成的剩余分析

- 三份标注各 288 行，共 864 条标注，覆盖 24 个不同任务对应的模型–任务单元（各基准 8 单元、各模型 8 单元），每单元 12 条回复。
- 检查了缺失、重复、任务/条目集合、vague 标记和跨表只读字段；依据 admin/key.json 回查原始 samples.jsonl / samples2.jsonl，答案类别与归一化 KEY 一致。T02-R07 的 answer 多一个末尾空格，去空白后与源回复相同，未修改原表。
- A、C 为 UTF-8；B 为 GB18030，已修正读取器，原文件字节保持不变，SHA256 记录在结果中。
- 修正主分析粒度：先按原定“至少两人同组→连通分量”生成运算划分，再与归一化答案类别取交集，得到论文理论要求的答案细化划分。**此交集是本次分析明确添加的修正，不能描述为原脚本已预先实现。** 原始纯运算结果也保留。
- VAGUE 按原规则在每位标注者的划分中为单例；多数 VAGUE 的 13/288 条（4.5%）在共识中必为单例。剔除分析只剔除多数 VAGUE；它是改变样本/估计对象的敏感性检查。
- 计算 r_max、无放回 U-statistic A(3)、h=3/5/7 的代入覆盖率；补充同单元配对差、基准/模型分组、逐一剔除标注者、全体划分交集、传递闭包审计。无需再收集模型调用。
- selftest 的答案/词法替代检查均通过。原自测会保留真实表中的 VAGUE，本次已清空合成标签的 VAGUE，避免自测污染。回复字段核验与逐单元细化断言通过。

重现：在本目录运行 `python3 code/score_partitions.py`、`python3 code/score_partitions.py --selftest`、`python3 code/analyze_results.py`。**不要重新运行 build_sheets.py，它会重写已完成标注表。**

## 2. 最终结果

均为选定 24 单元等权均值。覆盖率是把 12 条回复的经验类别频率当作诚实标签律、条件于共同视图和诚实标签独立的理论事件概率；并非在线协议成功率、准确率或人为攻击实测成功率。

| 粒度 | 平均 r_max | A(3)，U-statistic | f=1,h=3 覆盖率 | f=2,h=5 | f=3,h=7 | f=3 描述性 95% 单元 bootstrap 区间 |
|---|---:|---:|---:|---:|---:|---|
| 答案 | 0.9479 | 0.8890 | 0.8937 | 0.8639 | 0.8432 | [0.7208, 0.9430] |
| 答案＋人工运算，主分析 | 0.8090 | 0.5998 | 0.6186 | 0.5287 | **0.4767** | **[0.3192, 0.6439]** |
| 答案＋人工运算，剔除多数 VAGUE | 0.8462 | 0.6587 | 0.6768 | 0.5885 | 0.5346 | [0.3822, 0.6922] |
| 词法：答案＋KEY | 0.2014 | 0.01477 | 0.02339 | 0.005174 | 0.001423 | [0.000071, 0.003566] |
| 纯运算（不要求答案相同） | 0.8438 | 0.6578 | 0.6757 | 0.5888 | 0.5355 | [0.3830, 0.6932] |

**主结果应使用 0.477，不是原始纯运算的 0.536。** T01、T02、T10、T17 的共识运算组含不同答案，共计 73 对；纯运算组不能保证是答案的细化，因而不能直接用于论文的该项比较。剔除 VAGUE 后的 0.535 与纯运算 0.536 接近只是数值巧合，两者含义不同。

平均每单元 r_max 的七次方才是覆盖率；不能把平均 r_max 再取七次方。A(3) 对应任意类的三人同类碰撞，而 r_max^h 是指定经验众数事件，两者不能混用。

同单元配对比较（10,000 次单元 bootstrap）：

| f=3 对比 | 平均差 | 描述性 95% 区间 |
|---|---:|---|
| 答案减答案＋人工运算 | 0.3665，即 36.65 个百分点 | [0.2196, 0.5176] |
| 答案＋人工运算减词法 | 0.4752，即 47.52 个百分点 | [0.3157, 0.6396] |

人工运算细化使覆盖率相对答案下降约 43.5%；词法覆盖率比人工联合粒度低约 335 倍。后一比例易受接近零的分母影响，论文更宜报告绝对值与百分点差。答案到人工的 A(3) 下降约 32.5%，人工 A(3) 仍是词法的约 40.6 倍。

## 3. 结果好不好，是否支持原论点

**结果有用，且方向符合中心论点；它也要求收紧一部分叙述。**

| 论点 | 新证据与判断 | 可写程度 |
|---|---|---|
| 比较粒度影响 validity 激活概率 | 同一回复下 A(3) 从答案 0.889 降至人工联合 0.600；词法为 0.0148 | 支持限定子集的方向性结论。细化单调性由定理保证，实验展示幅度；并非独立证明定理 |
| 即使排除词法噪声，离散度仍降低指定众数认证覆盖率 | 0.843→0.477，配对差区间为正；三基准都同方向 | 支持。不能将其解释为任意协议无法达成一致或实际容错预算降到某一数值 |
| 词法严重过度拆分人工等价运算 | 人工联合 0.477、词法 0.00142；A(3) 同样大幅分离 | 支持与既有逐对 97% false-rejection 结果一致的解释。**本实验没有重新估计 97% 的误拒率** |
| 人工粒度无法测量，只能给界 | 现在获得明确的有限样本划分与点估计 | 旧措辞已过时，必须改；不能升级成“可靠测量客观语义/推理等价” |
| 细粒度认证几乎完全崩溃 | 人工联合仍为 0.477，剔除 VAGUE 为 0.535 | 若指人工运算，则证据不足；接近零的现象只能限定为词法仪器 |
| 原逐对界 [0.06,0.79] 合理 | 新点估计 0.477 数值处于旧区间内 | 数值相容。样本、k 和聚合对象不同，不能声称新实验验证或收紧了完整共同集的理论界 |
| LLM judges 主要测答案相等 | 新实验未运行 judge | 继续由原 E1 仪器校准支持，不能把 R1b 写成新的 judge 验证 |
| 提示攻击者依从性、强制攻击、外部 checker | 新实验没有涉及 | 这些段落不需要为 R1b 修改；也不能把本实验当作其新增证据 |

## 4. 稳健性与不能忽略的限制

| 检查 | 结果 | 含义 |
|---|---|---|
| 原始运算划分一致性 | 平均 ARI 0.523；成对共指一致率 82.3%；AB/AC/BC 的 ARI 为 0.652/0.477/0.440 | 一致性中等。82.3% 未校正偶然一致，不能单独当作高可靠性；与旧 Fleiss κ=0.532 不同定义，不可直接比较提高/降低 |
| 各标注者联合覆盖率 | A 0.509；B 0.429；C 0.517 | 有标注者差异；B 的 VAGUE 更多（29），A 12、C 6 |
| 逐一剔除标注者，仅两人都同组才合并 | 0.391、0.420、0.403 | 方向仍然一致；这是更严格规则，不能与主分析混为同一估计 |
| 三人划分与答案类别全交集 | 0.387 | 严格替代聚合降低绝对值；与主结果差约 9 个百分点，不是主值的统计置信下限 |
| 多数票传递闭包 | 5 单元新增 27 对未经直接多数支持的同组关系：T03、T04、T17、T18、T19 | 连通分量保证划分，却可能合并没有直接多数支持的回复。不能称所有同组对都获至少两人支持。它按原规则构造，不宜静默改用另一主规则 |
| VAGUE 剔除 | 主 0.477→0.535 | 主结论方向保留；剔除结果条件于可识别回复，不是同一总体的更准确值。单例处理只相对于合并模糊条目的选择更保守，不是人口覆盖率的保证下界 |
| 基准内覆盖率，人工联合 | GSM8K 0.385；MMLU 0.455；MBPP 0.589 | 各基准均低于同子集答案、远高于词法。各 8 单元，不能夸大基准差异 |
| 基准分层 bootstrap | 人工覆盖率区间 [0.317,0.637] | 与普通单元 bootstrap [0.319,0.644] 接近；仅检查重采样分层是否改变描述性区间 |
| 子集代表性 | 12 条回复答案覆盖率 0.843；选中单元完整回复池答案覆盖率 0.830；论文完整共同集为约 0.843 | 一个答案均值接近不能证明人工粒度代表性。模型轮换、每基准均衡、逐次候选池的分位抽样不是 369 单元等概率样本；不能当作全体总体估计 |
| 每单元只有 12 条回复 | 经验众数选择与频率的七次方可有有限样本偏差 | 当前区间只重采样单元，未传播回复采样、标注者和共识规则的不确定性，不能称完整人口置信区间 |
| KEY 而非完整推理 | 人工比较的是短摘要里写出的操作 | 不能推出内在推理链、全部推理步骤或因果机制是否相同 |
| 三个文件不等于独立性证据 | 三人完成，但无独立操作记录 | 描述材料要求独立，与“已有证据证明独立完成”分开；不补造独立性或公开预注册声明 |

不同模型各 8 个单元的人工覆盖率为 nano 0.527、4o-mini 0.311、4.1-mini 0.592；任务分配不同，不能以这些值论证模型能力排名或更强模型必然更一致。

## 5. 论文修改表（已按最小修改原则修订）

本表取代上一版扩写建议。7 处修改、5 处保留；只更新数值、过时措辞和必要粒度定义。正文不使用“补充实验”或实验内部编号，方法统一放附录。准确英文原文与待替换片段在 CSV 中，完整说明见 `paper_minimal_revision_zh.md`。

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

## 6. 附录的简短方法与表格

在原逐对 bounds 表后增加一段划分标注方法及直接结果表，见 `r1b_paper_insert.tex`。旧 bounds 表与原校准统计保留；L299、L357、L484、L488、L539 不再建议扩写。详细审计与实验诊断保留在本分析报告和结果文件，不堆入论文正文。

## 7. 结果文件

- `r1b_partition_results.json`：主结果、每单元结果、输入哈希、纯运算敏感性。
- `r1b_diagnostics.json`：配对差、标注者/模型/基准分析、闭包和子集审计。
- `r1b_cells.csv`：24 单元逐行结果。
- `paper_revision_table.csv`：完整改句表，包括英文原句。
- `r1b_paper_insert.tex`：附录方法和结果表候选。
- `r1b_selftest.json`：两项流水线自测。
