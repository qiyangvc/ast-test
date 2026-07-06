# AST 对抗垃圾文本实验补充方案

本文档说明本项目新增的 AST（Adversarial Spam Text）实验管线。早期补充阶段只完成代码、脚本和实验协议；当前提交版已经通过 `scripts/submission_pipeline.py` 完成 Word2Vec、MLP/CNN/RNN、clean test、AST test、UCI 外部测试、置信度搜索攻击和 AST 样本质量抽查。最终提交指标见 `docs/SUBMISSION_REPORT.md`。

## 目标

补齐项目在对抗样本训练与鲁棒性评测上的能力：

1. 支持先切分、再生成 AST，避免同源变体泄漏到测试集。
2. 支持 clean test 与 AST test 双测试集。
3. 支持按攻击类型统计效果，而不是只报 overall accuracy。
4. 支持文本级 AST 与 embedding 级 FGM/PGD 组合实验。
5. 支持输出可写入 PPT/报告的指标文件。

## 新增文件

```text
src/adversarial_text.py       # 中文垃圾文本规则攻击器
src/ast_dataset.py            # AST 数据集构建、切分、manifest 输出
src/adversarial_training.py   # embedding 级 FGM/PGD 对抗训练
src/ast_metrics.py            # clean/AST/robustness 指标
src/ast_experiment.py         # AST 实验 dry-run/execute 运行器
scripts/build_ast_dataset.py  # 构建 AST 数据集
scripts/run_ast_experiment.py # 规划或执行 AST 实验，默认 dry-run
scripts/prepare_external_datasets.py # 下载并规范化外部数据集
scripts/submission_pipeline.py # 提交版完整训练、评估、攻击与报告流水线
scripts/run_strong_ast_experiment.py # strong AST独立构建与训练流水线
scripts/evaluate_ast_cross.py # 跨AST扰动分布测试
scripts/serve_submission_ui.py # 浏览器图形化测试界面
```

`src/config.py` 新增：

```text
AST_DATASET_CONFIG
STRONG_AST_DATASET_CONFIG
AST_TRAINING_CONFIG
AST_EXPERIMENT_CONFIG
```

## 标签约定

本项目原训练代码使用：

```text
0 = spam
1 = normal
```

新增 AST 管线内部使用可读标签：

```text
spam
normal
```

写入 legacy 数据时仍保持原项目文件名：

```text
msgspam.log.seg  -> spam
msgpass.log.seg  -> normal
```

## 正确实验顺序

必须按以下顺序执行：

```text
原始 clean 数据
  -> 去重与冲突样本剔除
  -> 分层划分 train / val / test
  -> 在每个 split 内部生成 AST
  -> 训练只使用 train_clean 或 train_clean_ast
  -> 最终评估 test_clean 和 test_ast
```

不要先生成 AST 再随机切分。这样会导致同一条原始短信及其变体同时进入训练集和测试集，指标会虚高。

## 数据构建

先准备外部中文数据源：

```bash
python scripts/prepare_external_datasets.py
```

默认会准备三份中文数据：

```text
tensorlayer_text_antispam  原项目兼容 msglog 数据
spam_messages_lr           中文短信 spam/normal 数据
fbs_sms_dataset            伪基站真实垃圾短信，spam-only
```

本次已实际准备的数据规模：

```text
tensorlayer_text_antispam: 18,782 条，normal 9,801，spam 8,981
spam_messages_lr:          10,929 条，normal 3,939，spam 6,990
fbs_sms_dataset:           14,074 条，spam-only
```

`SpamMessagesLR` 原始文件中有 44 行为空文本，已跳过；不是可恢复样本。

如果只用项目原始格式，默认读取：

```text
data/msglog/msgspam.log.seg
data/msglog/msgpass.log.seg
```

构建 AST 数据集：

```bash
python scripts/build_ast_dataset.py \
  --input-dir tensorlayer_text_antispam=data/external/raw/tensorlayer_text_antispam/msglog \
  --canonical-jsonl spam_messages_lr=data/external/canonical/spam_messages_lr.jsonl \
  --canonical-jsonl fbs_sms_dataset=data/external/canonical/fbs_sms_dataset.jsonl \
  --output-dir data/ast_experiment \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2 \
  --seed 42 \
  --max-variants-spam 2 \
  --max-variants-normal 1
```

本次已构建的 AST 数据集统计：

```text
loaded: 43,785
kept_after_dedup: 39,562
dropped_duplicates: 4,218
dropped_conflicting_labels: 3

train_clean: spam 18,578，normal 9,115
val_clean:   spam 2,654，normal 1,302
test_clean:  spam 5,308，normal 2,605

train_ast: spam 37,092，normal 9,109
val_ast:   spam 5,297，normal 1,302
test_ast:  spam 10,593，normal 2,603
```

## 较强扰动 AST 扩展实验

当前完整提交结果 `output/submission_full_20260706_full/` 使用的是 `ast_strength=mild`，它强调标签稳定和训练稳定。为进一步验证模型在真实规避写法下的鲁棒性，新增独立 strong AST 实验组，不覆盖已有 mild 结果。

strong AST 生成策略：

```text
ast_strength: strong
max_variants_spam: 4
max_variants_normal: 1
```

strong 组在 spam 样本上额外引入：

```text
semantic_rewrite            语义级改写
strong_mixed                多策略组合扰动
pinyin_abbreviation         拼音/首字母缩写，如 微信 -> wx/vx
multi_keyword_obfuscation   多关键词同时插符号/缩写
contact_split               联系方式拆分，如 微信 -> w x / v/x
url_obfuscation             URL/入口规避
amount_obfuscation          金额/福利表达规避
strong_phrase_variant       更激进关键词替换
```

normal 样本仍只使用保守扰动，避免把正常短信改写成垃圾短信。

已构建的 strong AST 数据集：

```text
data/ast_experiment_strong

train_ast: spam 74,237，normal 9,109
val_ast:   spam 10,603，normal 1,302
test_ast:  spam 21,205，normal 2,603
```

一键构建并训练 strong AST 完整实验：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full
```

该脚本默认训练 `baseline`、`text_ast`、`embedding_fgm`、`text_ast_fgm` 四组模式和 `mlp/cnn/rnn` 三类模型，训练参数与完整提交版一致：`vector_size=200`、`w2v_epochs=20`、`clf_epochs=10`、`max_len=64`、`max_vocab=50000`、`batch_size=512`。

已额外完成两个跨扰动测试：

```bash
# mild训练模型 -> strong AST测试集
.venv_submit/bin/python scripts/evaluate_ast_cross.py \
  --output-dir output/submission_full_20260706_full \
  --dataset-dir data/ast_experiment_strong \
  --name mild_on_strong

# strong训练模型 -> mild AST测试集
 .venv_submit/bin/python scripts/evaluate_ast_cross.py \
  --output-dir output/submission_strong_ast_20260706_full \
  --dataset-dir data/ast_experiment \
  --name strong_on_mild
```

结果文件：

```text
output/submission_full_20260706_full/metrics/mild_on_strong_test_ast.json
output/submission_strong_ast_20260706_full/metrics/strong_on_mild_test_ast.json
```

关键结果：

```text
mild训练模型 -> strong AST:
  best = text_ast/cnn
  acc = 0.9406
  spam_recall = 0.9390
  text_ast_fgm/cnn acc = 0.8999

strong训练模型 -> mild AST:
  best = embedding_fgm/rnn
  acc = 0.9666
  spam_recall = 0.9733
  text_ast_fgm/rnn acc = 0.9629
```

这些结果说明：模型不是简单“扰动越强越好”。strong 训练组在 strong AST 原生测试上最稳，但跨回 mild AST 时不同模型表现有分化，因此最终报告应把 mild AST、strong AST 和 cross-AST 三类结果一起呈现。

输出结构：

```text
data/ast_experiment/
  manifest.json
  canonical/
    train_clean.jsonl
    train_ast.jsonl
    train_clean_ast.jsonl
    val_clean.jsonl
    val_ast.jsonl
    val_clean_ast.jsonl
    test_clean.jsonl
    test_ast.jsonl
    test_clean_ast.jsonl
  legacy/
    train_clean/
    train_ast/
    train_clean_ast/
    val_clean/
    val_ast/
    val_clean_ast/
    test_clean/
    test_ast/
    test_clean_ast/
```

`canonical/*.jsonl` 保留元数据：

```json
{
  "id": "ast_train_clean_project_msglog_spam_xxx_0",
  "source": "project_msglog",
  "label": "spam",
  "label_id": 0,
  "text": "加我薇信领取福利",
  "segmented": "加 我 薇信 领取 福利",
  "split": "train",
  "is_adversarial": true,
  "attack_type": "phrase_variant",
  "parent_id": "clean_project_msglog_spam_xxx",
  "operations": ["微信->薇信"]
}
```

`legacy/*/msg*.log.seg` 用于兼容原项目训练代码。

## 支持的文本级攻击类型

```text
phrase_variant       关键词变体，如 微信 -> 薇信 / V信 / vx
glyph_variant        字形近似替换，如 微 -> 薇
phonetic_variant     字音近似替换，如 信 -> 心 / 新
symbol_insertion     插空格、横线、点号等
traditional_variant  简繁/异体混用，如 奖 -> 獎
digit_letter_mix     数字字母混写，如 500 -> 5oo
contact_obfuscation  联系方式绕写，如 QQ -> 扣扣
mixed                多种扰动组合
```

## 实验模式

后续实验建议至少跑四组：

```text
baseline
  train_clean 训练
  test_clean / test_ast 评估

text_ast
  train_clean_ast 训练
  test_clean / test_ast 评估

embedding_fgm
  train_clean 训练，训练过程中启用 embedding 级 FGM
  test_clean / test_ast 评估

text_ast_fgm
  train_clean_ast 训练，训练过程中启用 embedding 级 FGM
  test_clean / test_ast 评估
```

查看 dry-run 计划，不训练、不测试：

```bash
python scripts/run_ast_experiment.py \
  --dataset-dir data/ast_experiment \
  --work-dir output/ast_runs \
  --mode text_ast_fgm \
  --models rnn cnn
```

真正执行时才加：

```bash
python scripts/run_ast_experiment.py \
  --dataset-dir data/ast_experiment \
  --work-dir output/ast_runs \
  --mode text_ast_fgm \
  --models rnn cnn \
  --execute
```

如果只是查看计划，不要添加 `--execute`；提交版完整结果已通过 `scripts/submission_pipeline.py` 单独执行完成，产物位于 `output/submission_full_20260706_full/`。

## 指标

新增评估指标包括：

```text
Clean Accuracy
AST Accuracy
Robust Drop = Clean Accuracy - AST Accuracy
Spam Precision / Recall / F1
Normal Precision / Recall / F1
False Positive Rate
Macro F1
Weighted F1
Attack Success Rate = 1 - AST Spam Recall
Per-attack metrics
```

垃圾文本检测中应优先看：

```text
Spam Recall
False Positive Rate
AST Spam Recall
Robust Drop
```

原因是：漏掉垃圾文本与误杀正常文本都比 overall accuracy 更关键。

## 与 PPT 实践部分的补齐关系

PPT 最后实践部分已有：

```text
无字符相似性网络 vs 有字符相似性网络
相似度阈值分析
样例分析
基准算法对比
```

项目新增部分补齐：

```text
明确 train/val/test 划分
防止 AST 同源样本泄漏
clean test 与 AST test 分开评估
按攻击类型拆分指标
支持 text-level AST + embedding-level FGM/PGD
输出 JSON/Markdown 报告
```

## 可扩展数据集

建议后续以 source 维度合并数据，不要直接混成一份失去来源的数据：

```text
project_msglog                 当前项目原始数据
SpamMessagesLR                 中文垃圾短信补充
FBS_SMS_Dataset                伪基站真实垃圾短信，只适合作为 spam 或外部测试
SpamMessage                    中文垃圾短信大规模补充，需清洗 ham 质量
chinese_conversation_and_spam  中文聊天垃圾信息，适合外部泛化测试
UCI SMS Spam Collection        英文短信，不建议直接混入中文主训练
SMS Phishing / Smishing        英文钓鱼短信，适合跨语言外部测试
```

当前脚本已内置的下载源：

```text
TensorLayer text-antispam:
https://raw.githubusercontent.com/tensorlayer/text-antispam/master/word2vec/data/msglog.tar.gz

SpamMessagesLR:
https://raw.githubusercontent.com/x-hacker/SpamMessagesLR/master/train.txt

FBS_SMS_Dataset:
https://github.com/Cypher-Z/FBS_SMS_Dataset.git

UCI SMS Spam Collection（可选英文源）:
https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
```

FBS_SMS_Dataset 来自 CCS'20 论文 *Lies in the Air: Characterizing Fake-base-station Spam Ecosystem in China*。使用该数据时应保留来源说明并按其 README 要求引用。

导入外部数据时建议先转为 canonical JSONL：

```json
{"source":"SpamMessagesLR","text":"短信内容","label":"spam"}
{"source":"SpamMessagesLR","text":"短信内容","label":"normal"}
```

然后：

```bash
python scripts/build_ast_dataset.py \
  --input-dir project_msglog=data/msglog \
  --canonical-jsonl SpamMessagesLR=data/external/spam_messages_lr.jsonl \
  --output-dir data/ast_experiment
```

## 已完成事项与限制

本次提交版已经完成：

```text
下载并规范化可访问的外部数据集
构建 AST 数据集
训练 Word2Vec
训练 MLP/CNN/RNN
运行 clean test
运行 AST test
运行 UCI 英文外部测试
生成模型指标
运行基于模型置信度搜索的文本攻击
生成 AST 自动质量检查文件
```

当前仍需如实说明的限制：

```text
Hugging Face gated 数据集需要有权限的 HF token，本机未能下载
AST 样本质量检查是自动检查，不等同于课程签字式人审
文本级 AST 以规则候选生成为主，置信度搜索已用于攻击评估，但不是覆盖所有模型的最强白盒攻击
尚未做多随机种子 mean ± std
尚未做显著性检验
```

如果要进一步严格，可以补：

```text
基于模型置信度的黑盒攻击
基于梯度/重要性的字符替换
人工审核 AST 样本语义保持率
多随机种子 mean ± std
显著性检验
```
