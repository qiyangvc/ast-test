# AST 鲁棒垃圾短信检测项目

这是一个面向课程作业提交的独立文本检测项目，目标是构建、训练并评估具备对抗鲁棒性的垃圾短信分类系统。项目重点不是简单训练一个 spam/normal 分类器，而是系统性比较 clean 数据、轻量 AST、较强 AST、embedding FGM 和跨扰动泛化能力。

## 项目特点

- 数据准备：自动下载并规范化 TensorLayer text-antispam、SpamMessagesLR、FBS_SMS_Dataset 和 UCI SMS Spam Collection。
- AST 数据构建：先去重、再分层切分、最后在各 split 内生成对抗变体，避免同源样本泄漏。
- 模型训练：Gensim Word2Vec + PyTorch MLP/TextCNN/BiLSTM/BiLSTM-Attention。
- 实验矩阵：默认复现 `baseline`、`text_ast`、`embedding_fgm`、`text_ast_fgm` 四组模式；`--full-matrix` 会额外加入 Focal Loss 变体、BiLSTM-Attention 和 soft voting 集成评估。
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
  audit_ast_lexicon.py          # AST 外部词库覆盖率和生成率审计
  audit_data_sources.py         # 外部数据、去重、切分泄漏审计
  audit_attack_candidates.py    # 训练前 AST 候选空间审计
  audit_ast_quality.py          # 已生成 AST 样本质量审计
  audit_vocab_oov.py            # 训练词表/OOV 覆盖审计

docs/
  AST_EXPERIMENT.md         # 实验协议和复现实验说明
  SUBMISSION_REPORT.md      # 作业提交报告

lexicons/
  chinese_spam_ast_lexicon.json # 独立 AST 词库，不再把关键词硬塞在代码里

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

默认输出到 `data/external/`，并生成 canonical JSONL。默认来源包含三组中文训练/AST 数据和 UCI 英文外部测试数据；`data/` 默认被 `.gitignore` 忽略。

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

## AST 词库与审计

AST 关键词、短语变体、强扰动短语、拼音缩写、语义改写模板和领域提示词已外置到：

```text
lexicons/chinese_spam_ast_lexicon.json
```

当前词库覆盖联系渠道、银行金融、电商促销、房产、教育、博彩、发票办证、运营商服务、退订伪装和匿名化占位符等场景。`src/adversarial_text.py`、数据集构建、置信度搜索和 Web UI 都通过同一个 `ChineseSpamTextAttacker` 加载该词库；`segment_for_project` 也会把该词库注册进 jieba，避免攻击规则认识领域词、分词器却拆碎领域词。

审计命令：

```bash
.venv_submit/bin/python scripts/audit_ast_lexicon.py \
  --dataset-dir data/ast_experiment_strong \
  --sample-size 1000 \
  --strength strong \
  --out output/ast_lexicon_audit_20260708.json
```

最近一次只读审计结果：

```text
phrase variants: 180
strong phrase variants: 84
pinyin abbreviations: 53
multi-keyword obfuscation terms: 166
semantic templates: 20
spam lexicon coverage: 0.9191
sampled strong-AST generation rate: 1.0000
```

注意：如果修改词库，必须重新构建 AST 数据集并重新训练/评估，旧的 `output/submission_*` 指标不会自动代表新词库结果。

## 训练前审计

这些脚本只读数据和词表，不训练模型：

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

最近一次审计摘要：

```text
external canonical: 49,359 条，spam 30,792，normal 18,567
strong train_clean_ast: 111,039
strong test_clean: 7,913
strong test_ast: 23,808
clean split overlap: 0
AST parent mismatch: 0
strong candidate generation rate: 1.0000
avg unique candidates per sampled spam text: 26.00
semantic rewrite quick uniqueness: 994 / 1000
text_ast_fgm vocab OOV: train_clean_ast 0.0192, test_ast 0.0520, UCI 0.2626
```

`audit_ast_quality.py` 对旧 strong AST 文件标记了约 16.4% 长度异常样本，主要来自旧语义改写模板过短；本轮已扩展语义模板和上下文槽位，但已有 AST 数据文件不会自动更新，需要重新运行 `scripts/build_ast_dataset.py` 或 strong 实验脚本的构建阶段。

## 完整训练

mild AST 默认完整流水线，复现当前已完成的 MLP/TextCNN/BiLSTM 四模式基础模型结果，并额外输出三模型 `ensemble_vote`：

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

如果要与旧 PPT 中的 Focal Loss、模型集成和 BiLSTM-Attention 尽量对齐，使用扩展矩阵。该命令会训练 8 组模式、4 类模型，并在每个 mode 下输出 `ensemble_vote` soft voting 结果：

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

扩展矩阵的模式为：

```text
baseline / focal
text_ast / text_ast_focal
embedding_fgm / embedding_fgm_focal
text_ast_fgm / text_ast_fgm_focal
```

模型为：

```text
mlp / cnn / rnn / bilstm_attn / ensemble_vote
```

strong AST 默认完整流水线：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full
```

strong AST 扩展矩阵同样加 `--full-matrix`：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --full-matrix \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_full_matrix
```

## WSL2 一键训练

Windows/WSL2 下可以直接运行脚本完成环境准备、数据准备和 32 个基础模型训练：

```bash
bash scripts/run_wsl2_full_matrix.sh
```

默认执行 strong AST 扩展矩阵：

```text
8 个 mode × 4 个模型 = 32 个训练
models: mlp / cnn / rnn / bilstm_attn
额外评估: ensemble_vote
```

脚本默认输出到带时间戳的新目录，避免旧 checkpoint 导致训练被跳过，例如：

```text
output/submission_strong_full_matrix_20260709_120000
```

常用参数：

```bash
# 跑 mild AST 的 32 个训练
PROFILE=mild bash scripts/run_wsl2_full_matrix.sh

# 指定固定输出目录
OUTPUT_DIR=output/submission_strong_full_matrix bash scripts/run_wsl2_full_matrix.sh

# 已经准备好 data/，只训练
SKIP_DATA_PREP=1 SKIP_DATA_BUILD=1 bash scripts/run_wsl2_full_matrix.sh
```

建议把项目放在 WSL2 的 Linux 文件系统下，例如 `~/big_data/-111`，不要放在 `/mnt/c/...`，否则大量小文件和模型读写会明显变慢。

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

以下结果来自当前已经训练完成的默认矩阵。新增的 Focal Loss、BiLSTM-Attention 和 `ensemble_vote` 需要按上面的 `--full-matrix` 命令重训后再更新指标。

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

Focal Loss、BiLSTM-Attention 和 soft voting 集成已经补进训练/评估/网页推理代码。仍可继续扩展的方向：

- Query-based 黑盒攻击：在现有置信度搜索基础上加入 beam search 或遗传搜索。
- 语义一致性约束：引入句向量相似度或小模型判别，过滤语义偏移过大的 AST。
- 主动学习闭环：把高置信逃逸样本加入二次标注池，再做迭代训练。
- 多语言鲁棒性：中文 AST、英文 UCI 和混合中英规避写法统一评估。
- 校准与阈值优化：增加 ECE/Brier Score，给出业务可用阈值而不是只报 accuracy。

这些点适合作为报告中的“未来工作”，不建议在当前已完成实验上临时硬改，以免破坏可复现性。
