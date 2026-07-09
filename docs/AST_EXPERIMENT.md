# AST 鲁棒文本检测实验手册

本文档记录当前独立项目的 AST 数据构建、训练、测试和复现实验协议。项目保留的是可提交作业的最终管线。

## 实验目标

1. 构建 clean、mild AST、strong AST 三类可复现实验数据。
2. 比较 `baseline`、`text_ast`、`embedding_fgm`、`text_ast_fgm` 四组核心训练策略，并支持扩展到 Focal Loss、BiLSTM-Attention 和 soft voting 集成矩阵。
3. 同时评估 clean test、AST test、UCI 英文外部测试和跨扰动泛化。
4. 使用基于模型置信度的搜索攻击，而不是只随机生成一条扰动样本。
5. 输出作业报告、完整指标 JSON、模型权重和本地图形化测试程序。

## 核心文件

```text
src/adversarial_text.py       # AST 扰动候选生成
src/ast_dataset.py            # 去重、切分、AST 构建
src/ast_metrics.py            # 指标计算
src/submission_serving.py     # 训练后模型加载和推理

scripts/prepare_external_datasets.py
scripts/build_ast_dataset.py
scripts/submission_pipeline.py
scripts/run_strong_ast_experiment.py
scripts/evaluate_ast_cross.py
scripts/serve_submission_ui.py
scripts/audit_ast_lexicon.py       # 外部 AST 词库审计
scripts/audit_data_sources.py      # 数据源、去重、切分审计
scripts/audit_attack_candidates.py # AST 候选空间审计
scripts/audit_ast_quality.py       # AST 质量审计
scripts/audit_vocab_oov.py         # 词表/OOV 审计

lexicons/chinese_spam_ast_lexicon.json
```

## 标签约定

```text
0 = spam
1 = normal
```

JSONL 中保留可读标签 `spam` / `normal`，模型训练和评估统一使用上面的二分类 ID。

## 数据协议

必须遵守如下顺序：

```text
原始 clean 数据
  -> 去重与冲突标签剔除
  -> 分层划分 train / val / test
  -> 只在各自 split 内生成 AST
  -> 训练使用 train_clean 或 train_clean_ast
  -> 最终只报告 test_clean 和 test_ast
```

不能先生成 AST 再随机切分，否则同一原始样本及其变体可能同时进入训练集和测试集，指标会虚高。

## 数据来源

已使用数据：

```text
TensorLayer text-antispam: 18,782 条，normal 9,801，spam 8,981
SpamMessagesLR:            10,929 条，normal 3,939，spam 6,990
FBS_SMS_Dataset:           14,074 条，spam-only
UCI SMS Spam Collection:   英文外部测试集
```

Hugging Face gated 数据集已通过脚本尝试访问，但当前环境未提供授权 token，因此记录为 `GatedRepoError`，没有纳入训练。

准备数据：

```bash
.venv_submit/bin/python scripts/prepare_external_datasets.py
```

构建 mild AST：

```bash
.venv_submit/bin/python scripts/build_ast_dataset.py \
  --input-dir tensorlayer_text_antispam=data/external/raw/tensorlayer_text_antispam/msglog \
  --canonical-jsonl spam_messages_lr=data/external/canonical/spam_messages_lr.jsonl \
  --canonical-jsonl fbs_sms_dataset=data/external/canonical/fbs_sms_dataset.jsonl \
  --output-dir data/ast_experiment \
  --ast-strength mild \
  --max-variants-spam 2 \
  --max-variants-normal 1
```

本次 mild 数据规模：

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

## AST 强度

`mild` 组强调标签稳定，主要扰动包括符号插入、同音/近形替换、繁简变体、金额和链接轻度规避。

`strong` 组用于更强鲁棒性评估，在 spam 样本上增加：

```text
semantic_rewrite
strong_mixed
pinyin_abbreviation
multi_keyword_obfuscation
contact_split
url_obfuscation
amount_obfuscation
strong_phrase_variant
```

normal 样本仍只做保守扰动，避免把正常短信改成垃圾短信。

## AST 外部词库

AST 词库已从 Python 代码中拆出，统一放在：

```text
lexicons/chinese_spam_ast_lexicon.json
```

词库包含：

```text
phrase_variants: 180
strong_phrase_variants: 84
pinyin_abbreviations: 53
char_variants: 42
traditional_variants: 53
multi_keyword_obfuscation_keywords: 166
benefit_hints: 40
semantic_templates: 20
```

覆盖域包括联系渠道、银行金融、电商促销、房产、教育、博彩、发票办证、运营商服务、退订伪装、匿名化占位符等。训练集构建、置信度搜索和图形化界面均使用同一个 `ChineseSpamTextAttacker` 加载该词库，因此不会出现训练和网页 AST 规则不一致。`segment_for_project` 会把 AST 领域词注册进 jieba，Word2Vec、模型输入和网页演示使用同一套分词口径。

只读审计命令：

```bash
.venv_submit/bin/python scripts/audit_ast_lexicon.py \
  --dataset-dir data/ast_experiment_strong \
  --sample-size 1000 \
  --strength strong \
  --out output/ast_lexicon_audit_20260708.json
```

最近一次审计结果：

```text
spam lexicon coverage: 0.9191
normal lexicon coverage: 0.2679
sampled strong-AST generation rate: 1.0000
```

审计剩余未覆盖的高频词主要是数据集中特定新闻/政治/品牌噪声词和泛化程度较低的实体词，不再为了刷覆盖率全部塞入 AST 词库。

## 数据与候选空间审计

补充审计命令：

```bash
.venv_submit/bin/python scripts/audit_data_sources.py \
  --dataset-dir data/ast_experiment_strong \
  --out output/data_source_audit_20260708.json

.venv_submit/bin/python scripts/audit_attack_candidates.py \
  --dataset-dir data/ast_experiment_strong \
  --sample-size 200 \
  --seed-count 4 \
  --out output/attack_candidate_audit_20260708_after_semantic.json

.venv_submit/bin/python scripts/audit_ast_quality.py \
  --dataset-dir data/ast_experiment_strong \
  --out output/ast_quality_audit_20260708.json \
  --review-jsonl output/ast_quality_suspicious_20260708.jsonl

.venv_submit/bin/python scripts/audit_vocab_oov.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full \
  --out output/vocab_oov_audit_20260708.json
```

审计摘要：

```text
external canonical: 49,359 条，spam 30,792，normal 18,567
dedup kept: 43,745，dropped_duplicates: 5,610，conflicts: 2
strong clean split overlap: 0
strong AST parent_id mismatch: 0
strong candidate generation rate: 1.0000
avg unique candidates per sampled spam text: 26.00
semantic rewrite quick uniqueness: 994 / 1000
text_ast_fgm vocab OOV:
  train_clean_ast: 0.0192
  test_ast:        0.0520
  UCI English:     0.2626
```

`audit_ast_quality.py` 对旧 strong AST 文件标记了约 16.4% 长度异常样本，主要来自旧语义改写模板过短和模板坍缩。本轮已扩展默认 benefit/contact/urgency、增加 `{context}` 语义槽位，并保留 `output/ast_quality_suspicious_20260708.jsonl` 供人工复核。注意：已有 `data/ast_experiment_strong` 仍是旧 AST 文件，若采用新生成器作为最终版本，必须重建 AST 数据集后再训练和评估。

strong 数据规模：

```text
train_ast: spam 74,237，normal 9,109
val_ast:   spam 10,603，normal 1,302
test_ast:  spam 21,205，normal 2,603
```

## 训练协议

mild 默认完整训练，用于复现当前已完成的 MLP/TextCNN/BiLSTM 四模式基础模型结果，并额外输出三模型 `ensemble_vote`：

```bash
.venv_submit/bin/python scripts/submission_pipeline.py \
  --output-dir output/submission_full_20260706_full \
  --vector-size 200 \
  --max-vocab 50000 \
  --max-len 64 \
  --w2v-epochs 20 \
  --clf-epochs 10 \
  --batch-size 512 \
  --confidence-attack-limit 0 \
  --review-sample-size 0
```

mild 扩展矩阵训练，用于补齐旧 PPT 中的 Focal Loss、模型集成和 BiLSTM-Attention 对照：

```bash
.venv_submit/bin/python scripts/submission_pipeline.py \
  --full-matrix \
  --output-dir output/submission_full_matrix \
  --vector-size 200 \
  --max-vocab 50000 \
  --max-len 64 \
  --w2v-epochs 20 \
  --clf-epochs 10 \
  --batch-size 512 \
  --confidence-attack-limit 0 \
  --review-sample-size 0
```

strong 默认完整训练：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full
```

strong 扩展矩阵训练：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --full-matrix \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_full_matrix
```

训练配置：

```text
Word2Vec: Gensim skip-gram, vector_size=200, epochs=20, negative sampling
Default classifiers: PyTorch MLP, TextCNN, BiLSTM
Full-matrix classifiers: MLP, TextCNN, BiLSTM, BiLSTM-Attention
Default modes: baseline, text_ast, embedding_fgm, text_ast_fgm
Full-matrix modes: baseline, focal, text_ast, text_ast_focal, embedding_fgm, embedding_fgm_focal, text_ast_fgm, text_ast_fgm_focal
Ensemble: ensemble_vote, probability soft voting over configured base models
Classifier epochs: 10
Batch size: 512
Max sequence length: 64
Max vocabulary: 50,000
Focal gamma: 2.0
```

## 评估协议

每个模型组合输出：

```text
clean metrics
AST metrics
robust_drop = clean_accuracy - ast_accuracy
spam_recall_drop = clean_spam_recall - ast_spam_recall
by_attack metrics
UCI English metrics
```

扩展矩阵中，每个 mode 还会在 base model 全部训练后输出 `ensemble_vote`，其概率为多个模型 softmax 概率的平均值，再按平均概率取最终类别。

跨扰动评估：

```bash
.venv_submit/bin/python scripts/evaluate_ast_cross.py \
  --output-dir output/submission_full_20260706_full \
  --dataset-dir data/ast_experiment_strong \
  --name mild_on_strong

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

## 已完成核心结果

```text
mild AST 最佳: text_ast_fgm/cnn, AST Acc = 0.9731
strong AST 最佳: text_ast_fgm/cnn, AST Acc = 0.9824
mild->strong 最佳: text_ast/cnn, Acc = 0.9406
strong->mild 最佳: embedding_fgm/rnn, Acc = 0.9666
strong 置信度攻击: 364/5308, ASR = 0.0686
strong AST 自动质检: 23807/23808 通过
```

注意：上面的训练指标来自已生成的默认矩阵 `data/ast_experiment*` 和 `output/submission_*` 产物。若修改 `lexicons/chinese_spam_ast_lexicon.json`，或采用新增的 Focal Loss、BiLSTM-Attention、`ensemble_vote` 扩展矩阵，必须重新构建 AST 数据集并重新运行训练、clean/AST/UCI/跨扰动/置信度攻击评估后，才能更新这些指标。

完整表格见 [SUBMISSION_REPORT.md](/Users/cjc/Documents/big_data/-111/docs/SUBMISSION_REPORT.md)。

## 图形化界面

```bash
.venv_submit/bin/python scripts/serve_submission_ui.py \
  --output-dir output/submission_full_20260706_full \
  --host 127.0.0.1 \
  --port 7860
```

界面支持单条预测、模型横向比较、AST 候选搜索和指标查看。若加载 `--full-matrix` 输出目录，页面会自动显示 `bilstm_attn` 和 `ensemble_vote`。

## 可继续扩展的创新点

Focal Loss、BiLSTM-Attention 和 soft voting 集成已经补进代码。后续仍可作为报告未来工作的是：

1. Query-based 黑盒搜索攻击：在候选 AST 上做 beam search、遗传搜索或贝叶斯优化。
2. 语义一致性过滤：用句向量相似度、NLI 或轻量判别器过滤语义漂移样本。
3. 逃逸样本主动学习：把置信度搜索成功样本加入复审池，形成二轮训练闭环。
4. 校准评估：增加 ECE、Brier Score 和阈值曲线，服务真实业务阈值选择。
5. 多语言规避评估：扩展中英混写、拼音、缩写和英文 spam 的联合鲁棒性。
