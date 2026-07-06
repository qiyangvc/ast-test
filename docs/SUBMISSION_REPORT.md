# 文本检测实践提交版报告

## 完成范围

本次补齐了从数据准备、Word2Vec 训练、MLP/CNN/RNN 分类训练、clean test、AST test、外部 UCI 英文测试、模型指标汇总、AST 样本质量检查，到基于模型置信度搜索攻击的完整流程。

完整运行命令：

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

其中 `--confidence-attack-limit 0` 表示对 clean spam 测试集做完整置信度搜索攻击，`--review-sample-size 0` 表示对 AST 测试集做完整自动质量检查。本次流水线耗时约 123.31 分钟，完整本地产物位于 `output/submission_full_20260706_full/`，目录大小约 505M。`output/` 和 `data/` 被 `.gitignore` 忽略，提交作业时需要额外打包该输出目录或抽取其中报告、指标和关键样例。

## 数据集

- 中文主数据：TensorLayer text-antispam、SpamMessagesLR、FBS_SMS_Dataset。
- 英文外部测试：UCI SMS Spam Collection，已下载并转为 canonical JSONL。
- Hugging Face gated 数据：已通过脚本尝试访问 `paulkm/chinese_conversation_and_spam` 和 `reatiny/chinese-spam-10000`，当前环境未提供有权限的 HF 登录/token，因此返回 `GatedRepoError`。失败原因已记录到 `output/submission_full_20260706_full/external_access/huggingface_gated_attempts.json`。

AST 数据构建采用先去重、再分层切分、最后在 split 内生成对抗变体，避免同源变体泄漏到测试集。

数据规模：

```text
loaded: 43,785
kept_after_dedup: 39,562

train_clean: spam 18,578，normal 9,115
val_clean:   spam 2,654，normal 1,302
test_clean:  spam 5,308，normal 2,605

train_ast: spam 37,092，normal 9,109
val_ast:   spam 5,297，normal 1,302
test_ast:  spam 10,593，normal 2,603
```

## 训练配置

- Word2Vec：Gensim skip-gram，`vector_size=200`，`epochs=20`，negative sampling。
- 分类器：PyTorch MLP、TextCNN、BiLSTM。
- 实验模式：`baseline`、`text_ast`、`embedding_fgm`、`text_ast_fgm`。
- 分类器训练：每个模型 `10` epoch，`batch_size=512`，`max_len=64`，`max_vocab=50000`。

说明：原项目 TensorFlow/TensorLayer 版本与当前 Python 3.12 环境不兼容，因此提交流水线使用 PyTorch 复现实训目标并保留真实训练、真实测试和可复现实验指标。

## 指标摘要

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

- 最佳 AST 准确率为 `text_ast_fgm/cnn` 的 `0.9731`。
- `text_ast_fgm/cnn` 相比 `baseline/cnn`，AST 准确率从 `0.9557` 提升到 `0.9731`，Robust Drop 从 `0.0178` 降到 `0.0039`。
- 最低 Robust Drop 为 `text_ast/rnn` 和 `text_ast_fgm/rnn`，均为 `0.0018`。
- 最佳英文 UCI 外部测试结果为 `text_ast_fgm/rnn`，准确率 `0.9324`。

## 较强扰动 AST 扩展组

当前主表格对应 `ast_strength=mild` 的训练结果。为增强实验完整性，已新增并完整跑通 `ast_strength=strong` 的独立数据构建、训练和评估组：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full
```

strong 组采用 `max_variants_spam=4`、`max_variants_normal=1`，在 spam 样本上增加语义改写、多关键词混淆、拼音/首字母缩写、联系方式拆分、URL/金额规避和强组合扰动。已构建的 strong AST 测试集规模为 `23,808` 条，其中 spam `21,205` 条、normal `2,603` 条。

strong 组完整训练耗时约 `155.91` 分钟，输出目录为 `output/submission_strong_ast_20260706_full/`。关键结果如下：

- strong AST 原生测试最佳：`text_ast_fgm/cnn`，AST Acc `0.9824`，AST Spam Recall `0.9828`。
- strong 组英文 UCI 外部测试最佳：`text_ast_fgm/rnn`，UCI Acc `0.9616`。
- strong 组基于模型置信度搜索攻击：目标 `text_ast_fgm/cnn`，搜索 `5308` 条 clean spam，成功 `364` 条，攻击成功率 `0.0686`。
- strong AST 自动质量检查：`23808` 条，`23807` 条通过，`1` 条需复核，通过率 `0.99996`。

已完成两项跨扰动评估：

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

跨扰动结果：

- mild 训练模型 -> strong AST：最佳 `text_ast/cnn`，Acc `0.9406`，Spam Recall `0.9390`；原 `text_ast_fgm/cnn` 在 strong AST 上 Acc `0.8999`。
- strong 训练模型 -> mild AST：最佳 `embedding_fgm/rnn`，Acc `0.9666`，Spam Recall `0.9733`；`text_ast_fgm/rnn` Acc `0.9629`。

结论：strong AST 组显著提高了对更强文本规避写法的原生鲁棒性，但跨回 mild AST 时并非所有模型都优于 mild 训练组，说明强扰动会引入一定分布偏移；报告中应同时呈现 mild、strong 和 cross-AST 三组指标。

## 置信度搜索攻击

- 攻击目标模型：`text_ast_fgm/cnn`
- clean spam 测试样本数：`5308`
- 实际搜索样本数：`5297`
- 成功数：`246`
- 攻击成功率：`0.0464`
- 输出文件：`output/submission_full_20260706_full/attacks/confidence_search_text_ast_fgm_cnn.jsonl`

该攻击不是只随机生成一个 AST 样本，而是为每条 spam 样本生成候选扰动，并选择模型 normal 类置信度最高的候选，属于基于模型置信度搜索的文本攻击。

## AST 样本质量检查

- 检查样本数：`13196`
- 通过数：`13195`
- 需要复核：`1`
- 通过率：`0.9999`
- 输出文件：`output/submission_full_20260706_full/review/ast_quality_review.jsonl`

限制说明：该文件是 Codex-assisted 自动质量检查，不等同于学生本人或第三方真实人审签名。如果课程严格要求人审，需要学生再打开该 JSONL 对样本进行确认。

## 关键产物

- 自动生成报告：`output/submission_full_20260706_full/SUBMISSION_REPORT.md`
- Word2Vec：`output/submission_full_20260706_full/word2vec/`
- 模型权重：`output/submission_full_20260706_full/models/`
- 全量指标：`output/submission_full_20260706_full/metrics/all_results.json`
- 单模型指标：`output/submission_full_20260706_full/metrics/<mode>/<model>.json`
- 置信度攻击：`output/submission_full_20260706_full/attacks/`
- AST 质检：`output/submission_full_20260706_full/review/`

## 图形化测试程序

本项目新增了一个浏览器图形界面，入口为：

```bash
.venv_submit/bin/python scripts/serve_submission_ui.py \
  --output-dir output/submission_full_20260706_full \
  --host 127.0.0.1 \
  --port 7860
```

打开 `http://127.0.0.1:7860` 后，可以进行：

```text
单条文本 spam/normal 预测
spam 与 normal 置信度展示
分词、未知词和截断信息查看
AST 扰动候选生成与模型置信度搜索
12 个模型横向对比
完整 Clean / AST / UCI 指标表查看
```

该测试程序使用 `output/submission_full_20260706_full/` 中的真实 PyTorch 权重、vocab 和指标文件，不需要重新训练。
