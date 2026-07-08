# AST 鲁棒垃圾短信检测项目

这是一个面向课程作业提交的独立文本检测项目，目标是构建、训练并评估具备对抗鲁棒性的垃圾短信分类系统。项目重点不是简单训练一个 spam/normal 分类器，而是系统性比较 clean 数据、轻量 AST、较强 AST、embedding FGM 和跨扰动泛化能力。

## 项目特点

- 数据准备：自动下载并规范化 TensorLayer text-antispam、SpamMessagesLR、FBS_SMS_Dataset 和 UCI SMS Spam Collection。
- AST 数据构建：先去重、再分层切分、最后在各 split 内生成对抗变体，避免同源样本泄漏。
- 模型训练：Gensim Word2Vec + PyTorch MLP/TextCNN/BiLSTM。
- 实验矩阵：`baseline`、`text_ast`、`embedding_fgm`、`text_ast_fgm` 四组模式。
- 鲁棒评估：clean test、AST test、按攻击类型统计、英文 UCI 外部测试、跨扰动测试。
- 攻击评估：基于模型 normal 类置信度搜索最容易逃逸的 AST 候选。
- 图形界面：本地浏览器页面支持单条预测、模型对比、AST 候选搜索和指标查看。

## 目录结构

```text
src/
  adversarial_text.py       # 中文垃圾短信 AST 生成器
  ast_dataset.py            # 数据清洗、切分、AST 构建和 manifest
  ast_metrics.py            # clean/AST/鲁棒性指标
  submission_serving.py     # PyTorch 提交模型加载与推理
  config.py                 # 当前项目路径和实验默认参数

scripts/
  prepare_external_datasets.py  # 下载并规范化外部数据
  build_ast_dataset.py          # 构建 mild/balanced/strong AST 数据集
  submission_pipeline.py        # 完整训练、评估、攻击、报告流水线
  run_strong_ast_experiment.py  # strong AST 独立完整实验
  evaluate_ast_cross.py         # mild/strong 跨扰动评估
  serve_submission_ui.py        # 图形化测试界面服务

docs/
  AST_EXPERIMENT.md         # 实验协议和复现实验说明
  SUBMISSION_REPORT.md      # 作业提交报告

web_ui/
  index.html
  static/
```

当前仓库只保留本次 AST 作业所需的代码路径和复现实验产物说明。

## 环境

当前实验使用 Python 3.12 虚拟环境 `.venv_submit`。如需重新安装依赖：

```bash
python -m venv .venv_submit
.venv_submit/bin/python -m pip install -U pip
.venv_submit/bin/python -m pip install -r requirements.txt
```

核心依赖见 [requirements.txt](/Users/cjc/Documents/big_data/-111/requirements.txt)。

## 数据准备

```bash
.venv_submit/bin/python scripts/prepare_external_datasets.py
```

默认输出到 `data/external/`，并生成 canonical JSONL。`data/` 默认被 `.gitignore` 忽略。

构建 mild AST 数据集：

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

构建 strong AST 数据集可直接运行 strong 实验脚本，它会复用 mild manifest 中的数据源。

## 完整训练

mild AST 完整流水线：

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

strong AST 完整流水线：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full
```

这两个输出目录都被 `.gitignore` 忽略。提交作业时如需附带训练产物，需要额外打包 `output/submission_full_20260706_full/` 和 `output/submission_strong_ast_20260706_full/`。

## 交叉评测

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

## 已完成结果

mild AST：

```text
最佳 AST Acc: text_ast_fgm/cnn = 0.9731
最低 Robust Drop: text_ast/rnn = 0.0018, text_ast_fgm/rnn = 0.0018
最佳 UCI Acc: text_ast_fgm/rnn = 0.9324
置信度搜索攻击: 5297 条，成功 246 条，ASR = 0.0464
AST 自动质检: 13196 条，通过 13195 条，通过率 = 0.9999
```

strong AST：

```text
最佳 strong AST Acc: text_ast_fgm/cnn = 0.9824
最佳 strong UCI Acc: text_ast_fgm/rnn = 0.9616
strong 置信度搜索攻击: 5308 条，成功 364 条，ASR = 0.0686
strong AST 自动质检: 23808 条，通过 23807 条，通过率 = 0.99996
跨扰动: mild->strong 最佳 text_ast/cnn = 0.9406；strong->mild 最佳 embedding_fgm/rnn = 0.9666
```

完整说明见 [docs/SUBMISSION_REPORT.md](/Users/cjc/Documents/big_data/-111/docs/SUBMISSION_REPORT.md)。

## 图形化测试程序

```bash
.venv_submit/bin/python scripts/serve_submission_ui.py \
  --output-dir output/submission_full_20260706_full \
  --host 127.0.0.1 \
  --port 7860
```

打开：

```text
http://127.0.0.1:7860
```

如果端口被占用，换一个端口即可，例如 `--port 7861`。界面使用标准库 HTTP 服务，不依赖 Flask、Streamlit 或 Gradio。

## 后续创新点

以下是更有价值、但本轮没有修改代码实现的后续方向：

- Query-based 黑盒攻击：在现有置信度搜索基础上加入 beam search 或遗传搜索。
- 语义一致性约束：引入句向量相似度或小模型判别，过滤语义偏移过大的 AST。
- 主动学习闭环：把高置信逃逸样本加入二次标注池，再做迭代训练。
- 多语言鲁棒性：中文 AST、英文 UCI 和混合中英规避写法统一评估。
- 校准与阈值优化：增加 ECE/Brier Score，给出业务可用阈值而不是只报 accuracy。

这些点适合作为报告中的“未来工作”，不建议在当前已完成实验上临时硬改，以免破坏可复现性。
