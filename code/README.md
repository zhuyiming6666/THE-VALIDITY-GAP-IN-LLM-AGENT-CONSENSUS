# v5 复算与构建

2026-09-18：用户授权补齐剩余 TODO，新增 `complete_todos.py`（KEY 前缀敏感性、图 2/3）、`run_matched_prompted.py`（完整匹配提示攻击组）、`analyse_matched_prompted.py`（完整性审计与任务 bootstrap）。从 llmbft 运行 `python3 v5/code/build_revision.py --compile`；此构建为离线，不会触发模型调用。详情见 `v5/analysis/todo_completion_20260918/README.md`。

以下保留原修订记录；其中 v4 路径和“其余 TODO 待办”描述为历史状态。

# v4 离线修订与复现

本目录是当前 `iclr-05` 的复算和构建入口；v3 的论文、原始实验及历史脚本保持不变。

2026-09-16 范围收窄：论文只保留用户指定的 11 点及必要联动修正。原稿的其他 TODO 恢复待办，不再自动补齐。此前扩大修改的版本保存在 `v4/analysis/before_scope_restore_20260916/`。

从 `llmbft` 目录运行：

```sh
python3 -B v4/code/build_revision.py --compile
```

依赖：Python、NumPy、Matplotlib、可用的 `pdflatex`/`bibtex`。构建依赖相邻 v3 目录和仓库原始数据，不是脱离仓库的独立数据包。v4/paper 已补齐会议样式、参考文献及本稿需要的图片/表格。

## 复算范围

- `revision_analysis.py`：使用现有评分和记录加载器，但重新实现异构统计和 800 对仪器比较。零碰撞保留为 0；样本不足才视为缺失；平均分布通过多项分配展开计算 U 统计量，不使用小数伪计数。
- 按当前评分器重新判断 800 对输出的答案相等和词面相等，保留历史 embedding/judge 返回值；任务簇 bootstrap 共 5,000 次，种子 20260915。
- 复核 123 任务主表、缺失 KEY 的完整案例敏感性、288 场 none/forced 记录和 300 场 checker 记录。checker 只读取存档结果，不执行候选程序。
- 原认证守卫不改；复用 `submission_certification_check.json`，验证其输入哈希以及当前记录与主分析一致。定理有限枚举、CV、重标记、零值、整数计数和平均分布 U 统计量无偏性均有回归检查。
- `build_revision.py`：恢复原图 2、3、4，只更新图 5 必须与当前 judge/词面重算一致的数字；不生成额外表格、面板或置信区间，不改后续另行更新的图 1。编译英文论文并记录输入/输出 SHA-256。
- 分析 JSON 中保留的敏感性结果、bootstrap 等属于离线复核记录，不代表获准插入论文；当前构建不会把这些扩展结果写入稿件。

## 不应使用的历史入口

v3 的 `run_all.sh`、`threshold_analysis.py` 和 `instruments.py` 中旧的异构/archived-label 分析不是本次修订的复现入口。保留它们是为了历史追溯，不表示修正后的论文采用其旧统计口径。v4 仍复用 v3 经核验的评分器、加载器和认证函数；所读取代码的哈希记入 manifest。

## 产物

- `v4/results/revision_analysis.json`：逐任务、逐对和汇总结果，含测试结果、输入哈希。
- `v4/results/revision_build_manifest.json`：构建来源及输出哈希。
- `v4/paper/fig/`：当前稿件使用的图片；之前额外生成的六张表已移至备份的 `withdrawn_tables/`，不被当前论文引用。
- `v4/paper/iclr-05.pdf`：修订后的英文 PDF。

所有步骤都是离线分析；没有新增模型调用，没有补造人工标签、完整 prompted 对照或 KEY 长度实验。

## 2026-09-16 人工校准材料修订

新增独立的 `score_annotations.py` 与冻结的 `annotation_normalization.py`，默认读取 v4/annotation，避免误读 v3。前者执行双轴多数、缺失 KEY 词面类比较、加权校准与覆盖率统计，并校验表格源文本和标签；后者是现有论文评分器的原样快照，其哈希保存在隐藏 key 中。

```sh
python3 -B v4/code/score_annotations.py --validate-only
python3 -B v4/code/tests/test_annotations.py
python3 -B v4/code/package_annotations.py
```

正式评分必须等三份真实独立标注齐备；结果路径是 v4/results/annotation_scoring.json。`package_annotations.py` 从空白母版生成总包及 A/B/C 个人盲标包，不含身份、gold、参考标签或 admin 文件。方案和发放说明位于 v4/annotation/；旧材料备份位于 v4/analysis/before_annotation_revision_20260916/。
