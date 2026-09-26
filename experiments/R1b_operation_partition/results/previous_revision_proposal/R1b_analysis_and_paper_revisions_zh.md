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

## 5. 论文修改表

行号基于当前未修改的 `paper-source/iclr-15.tex`；同一行常包含多个句子，以下指出待替换的句子或片段。CSV 另含精确英文原文，便于复核。英文中的 LaTeX 保留为可粘贴候选，并非已应用修改。

| TeX 行号 | 原始句子/片段（中文翻译） | 建议句子（中文翻译） | 建议句子（英文/TeX） | 改得好不好及理由 |
|---|---|---|---|---|
| [L44](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:44) | 在诚实标签条件独立且视图共同的条件下，三个模型在经典故障预算下，即使仅比较最终答案，诚实众数的平均认证概率也为 0.84；人工决定性运算粒度的覆盖率只能界定在 0.06–0.79。 | 在诚实标签条件独立且视图共同的条件下，完整共同任务集上的答案粒度覆盖率为 0.843；在 24 个选定模型–任务单元的补充人工划分实验中，答案加运算粒度的代入覆盖率为 0.477（描述性 95% 单元 bootstrap 区间 0.319–0.644），同一子集的答案粒度为 0.843。 | Under conditionally independent honest labels and a common view, answer-grain coverage at the classical budget is 0.843 on the full common set; a supplementary partition study of 24 selected model--task cells estimates answer+operation plug-in coverage at 0.477 (descriptive 95% cell-bootstrap interval 0.319--0.644), against 0.843 at the answer grain on the same subset. | 必须改。把“仅有界”更新为受限样本的直接估计，并避免把子集结果当作完整共同集结论。摘要偏长，可把区间留到正文。 |
| [L73](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:73) | 在 n=10、f=3 时，答案粒度的平均每任务认证概率为 0.843；人工决定性运算粒度只能界定在 0.06–0.79，而精确字符串匹配所得 0.004 主要是仪器伪影。 | 在 n=10、f=3 时，完整共同集的答案粒度覆盖率为 0.843。补充划分实验的同一 24 单元中，答案、答案加运算和词法粒度的代入覆盖率分别为 0.843、0.477 和 0.00142，表明词法匹配夸大了人工运算粒度的下降。 | At $n=10$, $f=3$, answer-grain coverage on the full common set is $0.843$. In the supplementary 24-cell partition study, plug-in coverage is $0.843$, $0.477$ and $0.00142$ at the answer, answer+operation and lexical grains, respectively, showing that lexical matching exaggerates the reduction observed under human operation grouping. | 必须改。三数全部使用同一子集；0.004 仍可在完整集实验处保留。证据与措辞匹配，不暗示人工粒度也接近归零。 |
| [L90](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:90) | 我们将两种测量的划分称为答案粒度和词法粒度。 | 主要自动化实验采用答案和词法划分；补充人工实验将运算共识连通分量与归一化答案类别取交集，定义答案加运算划分，保留原始运算分组作为敏感性结果。 | The main automated experiments use the answer and lexical partitions. The supplementary human study defines an answer+operation partition by intersecting majority-connected operation components with normalised answer classes; operation-only results are retained as a sensitivity analysis. | 必须改。指南允许不同答案共用运算组，纯运算划分不是答案划分的细化；必须显式定义与理论一致的联合粒度。这是分析修正，不能声称原脚本事先已实现。 |
| [L244](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:244) | 决定性运算粒度的覆盖率：假设人工关系构成答案划分的细化，由 A_r(2)≤r_max≤p_max 和 r_max²≤A_r(2) 导出覆盖率上下界。n=10、f=3 时为 0.06–0.79（外侧 95% 界 0.02–0.87），对比答案 0.843 和词法 0.004；同质拆分假设给上限 0.23–0.55；答案自检区间包含 0.843。 | 决定性运算粒度的测量：早期逐对标注在细化假设下给出了 0.06–0.79 的覆盖率界。补充研究对 24 单元各 12 条回复进行直接分组，并与答案类别取交集。n=10、f=3 时，该子集答案加运算覆盖率为 0.477（描述性区间 0.319–0.644），答案为 0.843、词法为 0.00142。剔除多数判为模糊的回复后为 0.535；三人分组与答案类别全交集为 0.387。新旧研究样本与估计对象不同，不能把新值当作完整共同集上下界的精确替代。 | \textbf{Measuring certification at the decisive-operation grain.} The earlier pairwise study bounded coverage by $[0.06,0.79]$ under a refinement assumption (Appendix~\ref{app:calib}). A supplementary study directly groups 12 replies in each of 24 selected model--task cells and intersects operation components with answer classes. At $n=10$, $f=3$, answer+operation plug-in coverage on this subset is $0.477$ (descriptive 95\% cell-bootstrap interval $[0.319,0.644]$), versus $0.843$ for answers and $0.00142$ lexically. Dropping majority-vague replies gives $0.535$; intersecting all three annotator partitions with answer classes gives $0.387$. These studies use different samples and estimands, so the subset estimate does not replace the full-frame bounds. | 必须改。正文直接报告补充测量，旧界移作背景而非删除。保留敏感性；删去正文的同质假设上限可减少篇幅，附录继续保留。 |
| [L260](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:260) | 答案加 KEY 是词法粒度；它拆分了人工认可等价对中的 97%。可辩护的细粒度结论是细化排序、仪器分歧、人工逐对区间及其覆盖率界。决定性运算粒度尚未测量，0.06–0.79 的覆盖率界很宽，自动仪器未恢复该关系，人工一致性仅中等；上下界也假设人工关系是划分。 | 词法数字测量的是指定字符串仪器下的离散度。新的答案加运算划分为选定 24 单元提供直接估计，但依赖仅 12 条回复、共识构造与有限样本。运算分组平均 ARI 为 0.523；多数票关系的传递闭包在 5 单元新增 27 对合并。这些限制禁止把 0.477 当作完整共同集或客观推理等价的精确测量。 | The answer+KEY partition remains lexical: its activation and certification values measure dispersion under a string instrument. The new answer+operation partitions provide direct estimates on 24 selected cells, conditional on 12 replies per cell and the consensus construction. Mean operation-partition ARI is $0.523$, and transitive closure adds 27 co-assigned pairs in five cells beyond direct majority support. These limitations preclude interpreting $0.477$ as exact full-set coverage or as a measurement of objective reasoning equivalence. | 必须改。旧“尚未测量”已过时，但不能改成“已经可靠解决”。ARI 与此前 Fleiss κ 不是同一指标；27 对是闭包诊断，不是证明标注错误。 |
| [L263](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:263) | 在 n=10、f=3 时，答案粒度平均每任务认证概率为 0.843，而人工决定性运算粒度只能界定在 0.06–0.79。 | 在 n=10、f=3 时，完整共同集的答案粒度覆盖率为 0.843；补充人工划分子集的答案加运算覆盖率为 0.477，表明即使剔除词法过度拆分的影响，认证覆盖率仍有下降。 | at $n=10$, $f=3$, answer-grain coverage is $0.843$ on the full common set, while the supplementary human-partition subset gives answer+operation coverage of $0.477$, showing a reduction that persists after accounting for lexical over-splitting. | 必须改。结论聚焦方向性证据，保留子集限定；不声称反驳或证明任意协议的容错能力。 |
| [L299](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:299) | 诚实标签律是基于约 30 条回复的代入多项分布；表 e5 给出 split-half 敏感性检查，评估的是训练半样本选出的类别，而非全样本经验众数。 | 完整共同集模拟采用约 30 条回复的代入多项分布；split-half 检查仍评估训练半样本选择的类别。独立 R1b 子集分析每单元仅有 12 条已标注回复，使用 r_max 的 h 次方闭式计算指定众数覆盖率。 | For the full-common-set simulations, the honest law is the plug-in multinomial from about $30$ replies; Table~\ref{tab:e5} reports a split-half sensitivity check that evaluates the class selected on the training half rather than the full-sample empirical mode. The separate R1b subset analysis uses 12 annotated replies per cell and computes designated-mode coverage directly as $r_{\max}^{h}$. | 建议改。区分完整集 Monte Carlo 与 R1b 的闭式计算，避免读者误以为人工划分也有约 30 条回复或经历相同模拟；原敏感性数字与方法保留。 |
| [L357](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:357) | 更强的模型在回答什么上趋同，而不是在如何陈述为什么上趋同。 | 较强模型在答案粒度有更高的碰撞率，而词法 KEY 碰撞率仍较低；这些测量本身不能确定决定性运算的趋同程度。 | Stronger models have higher answer-grain collision rates while lexical KEY collisions remain rare; these measurements alone do not establish how strongly decisive operations converge. | 建议改。原句虽强调“陈述”，仍容易被读作推理不趋同。补充分组显示大量运算等价，弱化该推论更严谨；不能根据每模型不同任务的 8 单元均值推断能力效应。 |
| [L484](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:484) | 这些逐对估计不能恢复划分、p_max 或 h≥3 的 A(h)，但能给认证覆盖率提供界，如下所述。 | 早期逐对估计本身不恢复划分、p_max 或 h≥3 的 A(h)，但能给覆盖率提供界；独立的 R1b 直接分组研究在下文为有限子集提供划分估计。 | the pairwise estimates alone do not recover a partition, $\pmaxx$ or $A(h)$ for $h\ge3$, but they bound coverage as follows; a separate R1b grouping study below supplies partition estimates for a limited subset. | 建议改。逐对标注的数学局限仍成立，补充分组并没有推翻它；新增实验与旧估计须分开。 |
| [L488](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:488) | 在人工答案加运算关系下 n=3f+1、h=2f+1 的指定众数覆盖率：同模型校准对的界、同质答案内估计与答案粒度精确值。 | 早期逐对校准给出的覆盖率界（含外侧 95% 界）及附加同质假设下的估计；答案列是完整共同集经验代入法下的参考值。后续 R1b 子集的直接划分结果单独报告。 | \caption{Earlier pairwise-calibration bounds on designated-mode coverage at $n=3f+1$ ($h=2f+1$), with 95\% outer limits in brackets and estimates under an additional within-answer homogeneity assumption. The answer column is the empirical plug-in reference on the full common set; direct R1b subset estimates are reported separately.} | 建议改。原表数字保留；明确“精确”仅相对于经验分布，避免读者把旧界和新子集当同一估计对象。 |
| [L513](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:513) | 两条路线可能使决定性运算粒度可测：围绕少量人工锚点融合仪器的 Dawid–Skene 潜在类别模型，以及前提与结论的判断聚合；其不可能性结果帮助解释运算共识难以定义。 | R1b 对有限子集通过人工直接分组测量了答案加运算划分；推广还需要更多单元与回复、独立性文档，以及对共识聚合规则的敏感性检查。潜在类别融合和前提与结论的判断聚合仍可作为未来研究方向，而非本研究已验证的方法。 | R1b measures an answer+operation partition on a limited subset through direct human grouping. Generalisation requires more cells and replies, documentation of annotator independence, and sensitivity checks for consensus aggregation. Latent-class instrument fusion and premise-versus-conclusion judgment aggregation \citep{list2002} remain possible research directions rather than validated solutions in this study. | 必须改。把“未来才能测量”改为“已有限测量、推广仍有条件”。不能根据三个文件推断标注者彼此独立。 |
| [L539](/Users/zhuyiming/Desktop/BFT/grain-iclr15/paper-source/iclr-15.tex:539) | 清单第 1 项：粒度及仪器。答案与答案加 KEY；归一化器、词法规则、嵌入与评判模型；遗漏会导致同样回复给出 0.843 或 0.004 的覆盖率。 | 清单第 1 项：粒度及仪器。答案、答案加 KEY 和 R1b 答案加运算；同一 R1b 回复子集给出的覆盖率分别为 0.843、0.00142、0.477。 | (1) grain and instrument & answer, answer+KEY, R1b answer+operation & the same R1b replies give coverage $0.843$, $0.00142$, $0.477$\\ | 建议改。清单展示三种粒度与同一回复比较，更能服务中心论点；相邻其他完整集行可保留并标明各自样本。 |

## 6. 建议新增，而非直接覆盖旧表

在 Appendix G 的旧 pairwise bounds 表后（当前 L496）添加 R1b 的方法段与一张独立表。候选 LaTeX 在 `results/r1b_paper_insert.tex`。该表把新子集的答案、人工联合、词法数值放在一起，避免把完整集 0.004 和子集 0.00142 混排。旧 bounds 表及旧 300 对校准统计不应因 R1b 被删除。

L299 的完整集模拟过程与 Dirichlet/split-half 数字继续保留，但新增 R1b 方法段要明确直接使用 r_max^h 闭式计算，k=12，不是完整集约 30 条回复的 30,000 次模拟。L543 的“答案粒度约 16% 不覆盖”来自完整集 0.843，仍成立，不能改成 R1b 人工 52.3% 不覆盖来冒充同一实验。

总体判断：这些修改让新增实证补上此前人工粒度仅有界的缺口，也把论证从“所有细粒度都几乎归零”校准为“人工运算细化存在实质下降，词法仪器又明显夸大下降”。论文中心的 validity informativeness gap 保留，证据链更完整；绝对数值和总体推广仍应保守。

## 7. 结果文件

- `r1b_partition_results.json`：主结果、每单元结果、输入哈希、纯运算敏感性。
- `r1b_diagnostics.json`：配对差、标注者/模型/基准分析、闭包和子集审计。
- `r1b_cells.csv`：24 单元逐行结果。
- `paper_revision_table.csv`：完整改句表，包括英文原句。
- `r1b_paper_insert.tex`：附录方法和结果表候选。
- `r1b_selftest.json`：两项流水线自测。
