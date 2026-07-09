# AST 鲁棒垃圾短信检测项目报告

## 项目定位

本项目已整理为独立的 AST 鲁棒文本检测作业项目。当前仓库只保留数据准备、AST 构建、PyTorch 训练评估、置信度搜索攻击、跨扰动测试和图形化演示相关代码。

项目核心问题是：垃圾短信模型在面对规避式改写时是否仍能稳定识别 spam，而不是只在 clean test 上取得高准确率。

## 完成范围

已完成：

- 外部数据下载与 canonical JSONL 标准化。
- clean 数据去重、冲突标签剔除、分层切分。
- mild AST 与 strong AST 数据构建。
- AST 词库外置为独立 JSON，并补充词库覆盖率审计脚本。
- 已补充训练集动态 AST 词表，默认与外部固定词库合并使用。
- Word2Vec 真实训练。
- MLP、TextCNN、BiLSTM 三类模型真实训练。
- 已补充可重训的 BiLSTM-Attention、Focal Loss 变体和 soft voting 模型集成代码。
- `baseline`、`text_ast`、`embedding_fgm`、`text_ast_fgm` 四组核心实验；扩展矩阵支持 8 组 mode。
- clean test、AST test、UCI 英文外部测试。
- 按攻击类型统计 AST 指标。
- 基于模型置信度搜索的攻击评估。
- mild/strong 双向跨扰动评估。
- AST 自动质量检查。
- 本地图形化测试界面。

## 数据集

中文训练与测试数据：

```text
TensorLayer text-antispam: 18,782 条，normal 9,801，spam 8,981
SpamMessagesLR:            10,929 条，normal 3,939，spam 6,990
FBS_SMS_Dataset:           14,074 条，spam-only
```

外部测试：

```text
UCI SMS Spam Collection: 英文 spam/ham 测试集
```

Hugging Face gated 数据集：

```text
paulkm/chinese_conversation_and_spam: GatedRepoError
reatiny/chinese-spam-10000: GatedRepoError
```

当前环境没有授权 token，因此 gated 数据没有纳入训练；失败原因已写入输出目录的 `external_access/huggingface_gated_attempts.json`。

## 数据构建协议

AST 数据构建遵守：

```text
clean 原始数据
  -> 去重和冲突标签剔除
  -> train/val/test 分层切分
  -> 只从 train split 挖掘动态 AST 词表
  -> split 内生成 AST
  -> train_clean 或 train_clean_ast 训练
  -> test_clean 与 test_ast 评估
```

这样避免同源 clean 样本和 AST 变体同时进入训练与测试。

AST 关键词、强扰动短语、拼音缩写、多关键词扰动项和语义改写提示词已外置到 `lexicons/chinese_spam_ast_lexicon.json`。最近一次只读审计显示：

```text
phrase_variants: 180
strong_phrase_variants: 84
pinyin_abbreviations: 53
multi_keyword_obfuscation_keywords: 166
semantic_templates: 20
spam lexicon coverage: 0.9191
sampled strong-AST generation rate: 1.0000
```

该词库同时被 AST 数据构建、confidence-search、图形化界面和 jieba 分词入口使用，避免训练/演示/审计规则不一致。本版还新增 `src/ast_vocabulary.py`：从 train split 自动挖掘 spam 区分度高的关键词，生成拼音、首字母、符号插入、繁体和可选 Word2Vec 相似词扰动，并保存为 `dynamic_vocab.json`。最终 AST 生成默认使用“外部固定词库 + 动态词表”，如需消融可用 `--no-dynamic-vocab` 关闭动态扩展。

审计命令：

```bash
.venv_submit/bin/python scripts/audit_ast_lexicon.py \
  --dataset-dir data/ast_experiment_strong \
  --sample-size 1000 \
  --strength strong \
  --out output/ast_lexicon_audit_20260708.json
```

补充数据审计：

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

审计结论：

```text
external canonical: 49,359 条，spam 30,792，normal 18,567
strong train_clean_ast: 111,039
strong test_clean: 7,913
strong test_ast: 23,808
clean split overlap: 0
AST parent_id mismatch: 0
strong candidate generation rate: 1.0000
avg unique candidates per sampled spam text: 26.00
semantic rewrite quick uniqueness: 994 / 1000
text_ast_fgm vocab OOV: train_clean_ast 0.0192, test_ast 0.0520, UCI 0.2626
```

质量审计额外发现：旧 strong AST 文件约 16.4% 样本被程序标记为长度异常，主要来自旧语义改写模板过短或多条原文坍缩到同一模板。本轮已扩展语义模板到 20 条、增加 `{context}` 槽位，并输出 `output/ast_quality_suspicious_20260708.jsonl` 作为人工复核清单。该修改不改变旧训练产物，最终采用时需要重建 AST 数据集并重跑训练/评估。

数据规模：

```text
loaded: 43,785
kept_after_dedup: 39,562
dropped_duplicates: 4,218
dropped_conflicting_labels: 3

clean:
  train: 27,693
  val:    3,956
  test:   7,913

mild AST:
  train_ast: 46,201
  val_ast:    6,599
  test_ast:  13,196

strong AST:
  train_ast: 83,346
  val_ast:   11,905
  test_ast:  23,808
```

## 训练配置

```text
Word2Vec: Gensim skip-gram, vector_size=200, epochs=20, negative sampling
Classifier: PyTorch MLP / TextCNN / BiLSTM
Modes: baseline / text_ast / embedding_fgm / text_ast_fgm
Classifier epochs: 10
Batch size: 512
Max sequence length: 64
Max vocabulary: 50,000
FGM epsilon: 0.5
```

为贴近旧 PPT 中的方法对照，代码已支持扩展矩阵：

```text
Classifier: MLP / TextCNN / BiLSTM / BiLSTM-Attention
Modes:
  baseline / focal
  text_ast / text_ast_focal
  embedding_fgm / embedding_fgm_focal
  text_ast_fgm / text_ast_fgm_focal
Ensemble: ensemble_vote, probability soft voting
Focal gamma: 2.0
```

完整命令：

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

strong AST 完整命令：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full
```

扩展矩阵重训命令：

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

.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --full-matrix \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_full_matrix
```

## Mild AST 结果

Precision / Recall / F1-Score 均以 spam 类作为正类。

| Mode | Model | Clean Acc | Clean P | Clean R | Clean F1 | AST Acc | AST P | AST R | AST F1 | Robust Drop | UCI Acc | UCI F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | mlp | 0.9621 | 0.9721 | 0.9714 | 0.9717 | 0.9469 | 0.9659 | 0.9680 | 0.9669 | 0.0152 | 0.8947 | 0.6836 |
| baseline | cnn | 0.9736 | 0.9824 | 0.9781 | 0.9803 | 0.9557 | 0.9687 | 0.9764 | 0.9725 | 0.0178 | 0.8746 | 0.6548 |
| baseline | rnn | 0.9711 | 0.9822 | 0.9746 | 0.9783 | 0.9577 | 0.9804 | 0.9667 | 0.9735 | 0.0133 | 0.9038 | 0.7112 |
| text_ast | mlp | 0.9592 | 0.9761 | 0.9627 | 0.9694 | 0.9560 | 0.9866 | 0.9582 | 0.9722 | 0.0032 | 0.9112 | 0.6950 |
| text_ast | cnn | 0.9702 | 0.9821 | 0.9732 | 0.9777 | 0.9681 | 0.9889 | 0.9712 | 0.9799 | 0.0021 | 0.9150 | 0.7310 |
| text_ast | rnn | 0.9697 | 0.9814 | 0.9732 | 0.9773 | 0.9679 | 0.9889 | 0.9708 | 0.9798 | 0.0018 | 0.9311 | 0.7653 |
| embedding_fgm | mlp | 0.9678 | 0.9800 | 0.9717 | 0.9759 | 0.9521 | 0.9703 | 0.9700 | 0.9702 | 0.0157 | 0.8945 | 0.6872 |
| embedding_fgm | cnn | 0.9778 | 0.9901 | 0.9766 | 0.9833 | 0.9617 | 0.9779 | 0.9742 | 0.9761 | 0.0161 | 0.8854 | 0.6748 |
| embedding_fgm | rnn | 0.9766 | 0.9849 | 0.9802 | 0.9825 | 0.9682 | 0.9844 | 0.9758 | 0.9801 | 0.0084 | 0.8945 | 0.7063 |
| text_ast_fgm | mlp | 0.9688 | 0.9814 | 0.9719 | 0.9766 | 0.9656 | 0.9879 | 0.9690 | 0.9784 | 0.0032 | 0.9056 | 0.7042 |
| text_ast_fgm | cnn | 0.9770 | 0.9884 | 0.9772 | 0.9828 | 0.9731 | 0.9928 | 0.9736 | 0.9831 | 0.0039 | 0.9151 | 0.7326 |
| text_ast_fgm | rnn | 0.9745 | 0.9887 | 0.9731 | 0.9808 | 0.9726 | 0.9927 | 0.9731 | 0.9828 | 0.0018 | 0.9324 | 0.7849 |

结论：

- mild AST 最佳 AST Acc 为 `text_ast_fgm/cnn = 0.9731`。
- `text_ast_fgm/cnn` 相比 `baseline/cnn`，AST Acc 从 `0.9557` 提升到 `0.9731`。
- 最低 Robust Drop 为 `text_ast/rnn` 和 `text_ast_fgm/rnn`，均为 `0.0018`。
- UCI 英文外部测试最佳为 `text_ast_fgm/rnn = 0.9324`。

## Strong AST 结果

Precision / Recall / F1-Score 均以 spam 类作为正类。

| Mode | Model | Clean Acc | Clean P | Clean R | Clean F1 | Strong AST Acc | Strong AST P | Strong AST R | Strong AST F1 | Robust Drop | UCI Acc | UCI F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | mlp | 0.9612 | 0.9786 | 0.9633 | 0.9709 | 0.8467 | 0.9815 | 0.8439 | 0.9075 | 0.1145 | 0.9055 | 0.6884 |
| baseline | cnn | 0.9736 | 0.9842 | 0.9763 | 0.9802 | 0.8720 | 0.9805 | 0.8736 | 0.9240 | 0.1016 | 0.8988 | 0.6974 |
| baseline | rnn | 0.9709 | 0.9829 | 0.9736 | 0.9782 | 0.8413 | 0.9879 | 0.8320 | 0.9033 | 0.1296 | 0.9116 | 0.7275 |
| text_ast | mlp | 0.9532 | 0.9777 | 0.9520 | 0.9647 | 0.9715 | 0.9955 | 0.9724 | 0.9838 | -0.0182 | 0.9142 | 0.6726 |
| text_ast | cnn | 0.9656 | 0.9807 | 0.9678 | 0.9742 | 0.9811 | 0.9954 | 0.9833 | 0.9893 | -0.0154 | 0.8970 | 0.6818 |
| text_ast | rnn | 0.9636 | 0.9808 | 0.9646 | 0.9726 | 0.9800 | 0.9952 | 0.9823 | 0.9887 | -0.0164 | 0.9453 | 0.7857 |
| embedding_fgm | mlp | 0.9661 | 0.9780 | 0.9714 | 0.9747 | 0.8764 | 0.9799 | 0.8793 | 0.9269 | 0.0897 | 0.8950 | 0.6840 |
| embedding_fgm | cnn | 0.9791 | 0.9912 | 0.9776 | 0.9843 | 0.8701 | 0.9813 | 0.8708 | 0.9227 | 0.1090 | 0.8986 | 0.6987 |
| embedding_fgm | rnn | 0.9766 | 0.9882 | 0.9768 | 0.9825 | 0.8000 | 0.9890 | 0.7842 | 0.8748 | 0.1766 | 0.9087 | 0.7339 |
| text_ast_fgm | mlp | 0.9583 | 0.9829 | 0.9544 | 0.9685 | 0.9742 | 0.9966 | 0.9743 | 0.9853 | -0.0159 | 0.9221 | 0.6814 |
| text_ast_fgm | cnn | 0.9711 | 0.9888 | 0.9678 | 0.9782 | 0.9824 | 0.9974 | 0.9828 | 0.9900 | -0.0113 | 0.9232 | 0.7500 |
| text_ast_fgm | rnn | 0.9687 | 0.9913 | 0.9618 | 0.9763 | 0.9802 | 0.9978 | 0.9799 | 0.9888 | -0.0116 | 0.9616 | 0.8558 |

结论：

- strong AST 原生测试最佳为 `text_ast_fgm/cnn = 0.9824`。
- strong AST spam recall 最佳为 `text_ast/cnn = 0.9833`。
- UCI 英文外部测试最佳为 `text_ast_fgm/rnn = 0.9616`。
- 仅 embedding FGM 在 strong AST 上效果较弱，说明文本级 AST 训练是主要增益来源。

## 跨扰动评估

| Eval | Best Model | Acc | Precision | Recall | F1-Score | 结果文件 |
|---|---|---:|---:|---:|---:|---|
| mild 训练模型 -> strong AST | text_ast/cnn | 0.9406 | 0.9940 | 0.9390 | 0.9657 | `output/submission_full_20260706_full/metrics/mild_on_strong_test_ast.json` |
| strong 训练模型 -> mild AST | embedding_fgm/rnn | 0.9666 | 0.9849 | 0.9733 | 0.9791 | `output/submission_strong_ast_20260706_full/metrics/strong_on_mild_test_ast.json` |

补充观察：原 mild 训练的 `text_ast_fgm/cnn` 在 strong AST 上 Acc 为 `0.8999`，而 strong 训练的 `text_ast_fgm/cnn` 在 strong AST 原生测试上 Acc 为 `0.9824`。这说明新增 strong AST 训练显著提升了对强规避写法的适应能力，但强扰动也会带来分布偏移，因此报告中保留 mild、strong、cross 三类结果更完整。

## 置信度搜索攻击

mild 组：

```text
目标模型: text_ast_fgm/cnn
搜索样本: 5297
成功数: 246
ASR: 0.0464
```

strong 组：

```text
目标模型: text_ast_fgm/cnn
搜索样本: 5308
成功数: 364
ASR: 0.0686
```

攻击过程为每条 clean spam 生成多个 AST 候选，并选择模型 normal 类置信度最高的候选，因此属于基于模型置信度的搜索攻击。

## AST 质量检查

```text
mild AST:   13,196 条，13,195 条通过，1 条需复核，通过率 0.9999
strong AST: 23,808 条，23,807 条通过，1 条需复核，通过率 0.99996
```

限制说明：该检查是程序辅助质检，不等同于学生本人或第三方人工签名审核。如果课程要求人审，需要学生打开 `review/ast_quality_review.jsonl` 进行最终确认。

## 产物

```text
output/submission_full_20260706_full/
  SUBMISSION_REPORT.md
  metrics/all_results.json
  models/
  word2vec/
  attacks/
  review/

output/submission_strong_ast_20260706_full/
  SUBMISSION_REPORT.md
  metrics/all_results.json
  metrics/strong_on_mild_test_ast.json
  models/
  word2vec/
  attacks/
  review/
```

`data/` 和 `output/` 被 `.gitignore` 忽略，提交作业时如需附带训练产物，需要额外打包。

## 重要说明

当前表格中的 mild/strong 训练指标来自已完成的默认矩阵训练产物。若继续使用本次扩展后的外部 AST 词库、启用动态词表，或将 Focal Loss、BiLSTM-Attention、`ensemble_vote` 作为最终实验版本，应重新执行：

```text
1. rebuild AST dataset
2. train Word2Vec
3. train MLP/TextCNN/BiLSTM/BiLSTM-Attention under the full matrix when needed
4. run clean / AST / UCI / cross-perturbation evaluation
5. rerun confidence-search attack and AST quality review
```

没有重新跑完上述流程前，不能把旧指标称为“固定词库 + 动态词表后的最终指标”。

## 图形化测试程序

```bash
.venv_submit/bin/python scripts/serve_submission_ui.py \
  --output-dir output/submission_full_20260706_full \
  --host 127.0.0.1 \
  --port 7860
```

浏览器打开 `http://127.0.0.1:7860` 后，可以进行单条文本预测、置信度展示、AST 候选搜索、模型横向对比和完整指标查看。若加载扩展矩阵输出目录，界面会自动显示 `bilstm_attn` 和 `ensemble_vote`。

## 可进一步创新

Focal Loss、BiLSTM-Attention 和 soft voting 集成已经补进代码。为避免继续扩大重训范围，以下方向只作为报告未来工作：

- 更强黑盒搜索：beam search、遗传算法或贝叶斯优化替代当前候选枚举。
- 语义一致性约束：加入句向量相似度或 NLI 过滤，减少 AST 语义漂移。
- 主动学习闭环：将置信度搜索成功的逃逸样本送入复审和二次训练。
- 模型校准：增加 ECE、Brier Score 和阈值曲线，服务真实业务阈值选择。
- 多语言/混写鲁棒性：将拼音、英文 spam、中英混写、符号规避统一纳入评估。
