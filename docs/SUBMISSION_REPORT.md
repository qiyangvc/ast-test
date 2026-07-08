# AST 鲁棒垃圾短信检测项目报告

## 项目定位

本项目已整理为独立的 AST 鲁棒文本检测作业项目。当前仓库只保留数据准备、AST 构建、PyTorch 训练评估、置信度搜索攻击、跨扰动测试和图形化演示相关代码。

项目核心问题是：垃圾短信模型在面对规避式改写时是否仍能稳定识别 spam，而不是只在 clean test 上取得高准确率。

## 完成范围

已完成：

- 外部数据下载与 canonical JSONL 标准化。
- clean 数据去重、冲突标签剔除、分层切分。
- mild AST 与 strong AST 数据构建。
- Word2Vec 真实训练。
- MLP、TextCNN、BiLSTM 三类模型真实训练。
- `baseline`、`text_ast`、`embedding_fgm`、`text_ast_fgm` 四组实验。
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
  -> split 内生成 AST
  -> train_clean 或 train_clean_ast 训练
  -> test_clean 与 test_ast 评估
```

这样避免同源 clean 样本和 AST 变体同时进入训练与测试。

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

## Mild AST 结果

| Mode | Model | Clean Acc | Clean Spam Recall | AST Acc | AST Spam Recall | Robust Drop | UCI Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | mlp | 0.9621 | 0.9714 | 0.9469 | 0.9680 | 0.0152 | 0.8947 |
| baseline | cnn | 0.9736 | 0.9781 | 0.9557 | 0.9764 | 0.0178 | 0.8746 |
| baseline | rnn | 0.9711 | 0.9746 | 0.9577 | 0.9667 | 0.0133 | 0.9038 |
| text_ast | mlp | 0.9592 | 0.9627 | 0.9560 | 0.9582 | 0.0032 | 0.9112 |
| text_ast | cnn | 0.9702 | 0.9732 | 0.9681 | 0.9712 | 0.0021 | 0.9150 |
| text_ast | rnn | 0.9697 | 0.9732 | 0.9679 | 0.9708 | 0.0018 | 0.9311 |
| embedding_fgm | mlp | 0.9678 | 0.9717 | 0.9521 | 0.9700 | 0.0157 | 0.8945 |
| embedding_fgm | cnn | 0.9778 | 0.9766 | 0.9617 | 0.9742 | 0.0161 | 0.8854 |
| embedding_fgm | rnn | 0.9766 | 0.9802 | 0.9682 | 0.9758 | 0.0084 | 0.8945 |
| text_ast_fgm | mlp | 0.9688 | 0.9719 | 0.9656 | 0.9690 | 0.0032 | 0.9056 |
| text_ast_fgm | cnn | 0.9770 | 0.9772 | 0.9731 | 0.9736 | 0.0039 | 0.9151 |
| text_ast_fgm | rnn | 0.9745 | 0.9731 | 0.9726 | 0.9731 | 0.0018 | 0.9324 |

结论：

- mild AST 最佳 AST Acc 为 `text_ast_fgm/cnn = 0.9731`。
- `text_ast_fgm/cnn` 相比 `baseline/cnn`，AST Acc 从 `0.9557` 提升到 `0.9731`。
- 最低 Robust Drop 为 `text_ast/rnn` 和 `text_ast_fgm/rnn`，均为 `0.0018`。
- UCI 英文外部测试最佳为 `text_ast_fgm/rnn = 0.9324`。

## Strong AST 结果

| Mode | Model | Clean Acc | Strong AST Acc | Robust Drop | Strong AST Spam Recall | UCI Acc |
|---|---|---:|---:|---:|---:|---:|
| baseline | mlp | 0.9612 | 0.8467 | 0.1145 | 0.8439 | 0.9055 |
| baseline | cnn | 0.9736 | 0.8720 | 0.1016 | 0.8736 | 0.8988 |
| baseline | rnn | 0.9709 | 0.8413 | 0.1296 | 0.8320 | 0.9116 |
| text_ast | mlp | 0.9532 | 0.9715 | -0.0182 | 0.9724 | 0.9142 |
| text_ast | cnn | 0.9656 | 0.9811 | -0.0154 | 0.9833 | 0.8970 |
| text_ast | rnn | 0.9636 | 0.9800 | -0.0164 | 0.9823 | 0.9453 |
| embedding_fgm | mlp | 0.9661 | 0.8764 | 0.0897 | 0.8793 | 0.8950 |
| embedding_fgm | cnn | 0.9791 | 0.8701 | 0.1090 | 0.8708 | 0.8986 |
| embedding_fgm | rnn | 0.9766 | 0.8000 | 0.1766 | 0.7842 | 0.9087 |
| text_ast_fgm | mlp | 0.9583 | 0.9742 | -0.0159 | 0.9743 | 0.9221 |
| text_ast_fgm | cnn | 0.9711 | 0.9824 | -0.0113 | 0.9828 | 0.9232 |
| text_ast_fgm | rnn | 0.9687 | 0.9802 | -0.0116 | 0.9799 | 0.9616 |

结论：

- strong AST 原生测试最佳为 `text_ast_fgm/cnn = 0.9824`。
- strong AST spam recall 最佳为 `text_ast/cnn = 0.9833`。
- UCI 英文外部测试最佳为 `text_ast_fgm/rnn = 0.9616`。
- 仅 embedding FGM 在 strong AST 上效果较弱，说明文本级 AST 训练是主要增益来源。

## 跨扰动评估

| Eval | Best Model | Acc | Spam Recall | 结果文件 |
|---|---|---:|---:|---|
| mild 训练模型 -> strong AST | text_ast/cnn | 0.9406 | 0.9390 | `output/submission_full_20260706_full/metrics/mild_on_strong_test_ast.json` |
| strong 训练模型 -> mild AST | embedding_fgm/rnn | 0.9666 | 0.9733 | `output/submission_strong_ast_20260706_full/metrics/strong_on_mild_test_ast.json` |

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

## 图形化测试程序

```bash
.venv_submit/bin/python scripts/serve_submission_ui.py \
  --output-dir output/submission_full_20260706_full \
  --host 127.0.0.1 \
  --port 7860
```

浏览器打开 `http://127.0.0.1:7860` 后，可以进行单条文本预测、置信度展示、AST 候选搜索、12 个模型横向对比和完整指标查看。

## 可进一步创新但本轮未修改

为避免破坏当前已完成实验的可复现性，以下方向只作为报告未来工作，不在本轮改代码：

- 更强黑盒搜索：beam search、遗传算法或贝叶斯优化替代当前候选枚举。
- 语义一致性约束：加入句向量相似度或 NLI 过滤，减少 AST 语义漂移。
- 主动学习闭环：将置信度搜索成功的逃逸样本送入复审和二次训练。
- 模型校准：增加 ECE、Brier Score 和阈值曲线，服务真实业务阈值选择。
- 多语言/混写鲁棒性：将拼音、英文 spam、中英混写、符号规避统一纳入评估。
