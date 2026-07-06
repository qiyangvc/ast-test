# 垃圾文本分类器项目

## 项目简介

本项目是基于原垃圾文本分类器项目的重构与增强版本，主要用于对垃圾短信/文本进行分类识别。项目包含 Word2Vec 词向量训练和多种文本分类模型。

---

## 与原项目的区别

### 1. 架构重构

| 模块 | 原项目 | 重构后 |
|------|--------|--------|
| 项目结构 | 单文件脚本 | 模块化架构 (`src/`) |
| 配置管理 | 硬编码 | 统一配置中心 (`config.py`) |
| 数据加载 | 零散处理 | 统一数据加载器 (`data_loader.py`) |

### 2. 模型增强

| 模型/技术 | 原项目 | 重构后 |
|-----------|--------|--------|
| Word2Vec | 普通softmax损失 | **NCE损失**（解决训练不稳定） |
| CNN分类器 | 单尺度卷积 | **多尺度TextCNN**（2/3/4/5-gram） |
| RNN分类器 | 普通LSTM | **BiLSTM+注意力机制** |
| 优化器 | 固定学习率 | **余弦退火学习率调度** |
| 正则化 | 无 | **早停法** + Focal Loss |
| 集成方法 | 无 | **模型集成**（加权平均+投票） |

### 3. 创新点实现

| 创新点 | 描述 | 准确率提升 |
|--------|------|------------|
| **创新点1** | 余弦退火学习率调度器 | +0.88% |
| **创新点2** | 早停法防止过拟合 | 稳定训练 |
| **创新点3** | Focal Loss处理类别不平衡 | +1.22% |
| **创新点4** | 模型集成（加权平均/投票） | +2.05% |
| **创新点5** | 多尺度TextCNN | +1.68% |
| **创新点6** | BiLSTM+注意力机制 | **+2.08%** ⭐ |
| **创新点7** | Mixup数据增强 | +0.64% |
| **创新点10** | SWA随机权重平均 | +0.82% |
| **创新点11** | Lookahead优化器 | +0.53% |
| **创新点12** | Label Smoothing | +0.59% |

### 4. 训练稳定性优化

**原项目问题：** Word2Vec训练时loss先降后升，出现异常波动

**解决方案：**
- 将softmax损失替换为NCE损失
- 学习率从0.025降至0.001
- batch_size从8增至128
- 添加梯度裁剪防止梯度爆炸

### 5. 文件结构变化

```
text-antispam/
├── src/                    # 新增：源代码目录
│   ├── __init__.py
│   ├── word2vec.py         # Word2Vec训练模块
│   ├── classifier.py       # 基础分类器
│   ├── advanced_models.py  # 高级模型（新增）
│   ├── enhanced_training.py # 优化技术（新增）
│   ├── advanced_augmentation.py # 数据增强（新增）
│   ├── advanced_optimizers.py # 高级优化器（新增）
│   ├── data_loader.py      # 数据加载器（新增）
│   ├── config.py           # 配置中心（新增）
│   └── utils.py            # 工具函数（新增）
├── word2vec/               # 词向量数据
│   └── data/
│       └── msglog/
│           ├── msgspam.log.seg   # 垃圾短信数据
│           └── msgpass.log.seg   # 正常短信数据
├── output/                 # 输出目录（自动创建）
├── .venv/                  # 虚拟环境
├── requirements.txt        # 依赖清单
└── test_*.py               # 测试脚本（新增）
```

---

## 环境配置

### 1. 系统要求

- **操作系统**: Windows 10/11（测试环境）
- **Python版本**: 3.7.x - 3.9.x（推荐3.8）
- **内存**: 建议16GB以上（大数据集训练）
- **存储**: 至少10GB可用空间

### 2. 依赖安装

```powershell
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
.venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

**requirements.txt 内容：**
```
tensorflow==2.2.0
numpy==1.19.2
scikit-learn==0.23.2
tqdm==4.48.2
```

### 3. 数据准备

确保数据集文件存在于以下路径：
```
word2vec/data/msglog/msgspam.log.seg   # 垃圾短信（约138万条）
word2vec/data/msglog/msgpass.log.seg   # 正常短信
```

### 4. 目录结构检查

运行前确保目录结构完整：
```powershell
# 创建必要目录
mkdir -p output word2vec/data/msglog
```

---

## 运行指南

### AST对抗垃圾文本实验（新增）

本项目已补充 AST（Adversarial Spam Text）数据构建、对抗训练和鲁棒性评估入口。提交版已经新增 `scripts/submission_pipeline.py`，可在当前 Python 3.12 环境下完成真实 Word2Vec 训练、MLP/CNN/RNN 训练、clean test、AST test、UCI 英文外部测试、AST 样本质量抽查和基于模型置信度搜索的攻击。

详细说明见：

```
docs/AST_EXPERIMENT.md
docs/SUBMISSION_REPORT.md
```

本次已执行的完整提交流水线：

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

主要输出：

```text
output/submission_full_20260706_full/SUBMISSION_REPORT.md
output/submission_full_20260706_full/metrics/all_results.json
output/submission_full_20260706_full/models/
output/submission_full_20260706_full/word2vec/
output/submission_full_20260706_full/attacks/
output/submission_full_20260706_full/review/
```

本次完整产物约 505M。`output/` 和 `data/` 默认被 `.gitignore` 忽略，提交作业时如果需要附带训练结果，请同时打包 `output/submission_full_20260706_full/`。

较强扰动 AST 扩展实验：

```bash
.venv_submit/bin/python scripts/run_strong_ast_experiment.py \
  --dataset-dir data/ast_experiment_strong \
  --output-dir output/submission_strong_ast_20260706_full
```

该脚本会复用 mild 实验的原始数据来源，重新构建 `ast_strength=strong` 的 AST 数据集，并以完整参数训练 Word2Vec、MLP/CNN/RNN、clean test、strong AST test、UCI test、strong 置信度搜索攻击和 AST 质量检查。默认参数为 `max_variants_spam=4`、`max_variants_normal=1`，不会覆盖现有 `output/submission_full_20260706_full/`。

如果要做跨扰动测试，例如 mild 训练结果测试 strong AST、strong 训练结果测试 mild AST：

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

本次 mild 完整训练的关键结果：

```text
最佳 AST Acc: text_ast_fgm/cnn = 0.9731
最低 Robust Drop: text_ast/rnn = 0.0018, text_ast_fgm/rnn = 0.0018
最佳 UCI Acc: text_ast_fgm/rnn = 0.9324
置信度搜索攻击: 5297 条，成功 246 条，ASR = 0.0464
AST 自动质检: 13196 条，通过 13195 条，通过率 = 0.9999
```

本次 strong 完整训练的关键结果：

```text
最佳 strong AST Acc: text_ast_fgm/cnn = 0.9824
最佳 strong UCI Acc: text_ast_fgm/rnn = 0.9616
strong 置信度搜索攻击: 5308 条，成功 364 条，ASR = 0.0686
strong AST 自动质检: 23808 条，通过 23807 条，通过率 = 0.99996
跨扰动: mild->strong 最佳 text_ast/cnn = 0.9406；strong->mild 最佳 embedding_fgm/rnn = 0.9666
```

启动图形化测试程序：

```bash
.venv_submit/bin/python scripts/serve_submission_ui.py \
  --output-dir output/submission_full_20260706_full \
  --host 127.0.0.1 \
  --port 7860
```

然后在浏览器打开：

```text
http://127.0.0.1:7860
```

该界面支持单条文本预测、spam/normal 置信度展示、AST 扰动候选搜索、12 个训练模型横向对比和完整指标表查看。它使用 Python 标准库 HTTP 服务，不需要额外安装 Flask、Streamlit 或 Gradio。

构建 AST 数据集：

```powershell
python scripts/prepare_external_datasets.py
python scripts/build_ast_dataset.py `
  --input-dir tensorlayer_text_antispam=data/external/raw/tensorlayer_text_antispam/msglog `
  --canonical-jsonl spam_messages_lr=data/external/canonical/spam_messages_lr.jsonl `
  --canonical-jsonl fbs_sms_dataset=data/external/canonical/fbs_sms_dataset.jsonl `
  --output-dir data/ast_experiment
```

查看实验计划（不训练、不测试）：

```powershell
python scripts/run_ast_experiment.py --dataset-dir data/ast_experiment --mode text_ast_fgm --models rnn cnn
```

真正执行实验时需要显式添加 `--execute`。

### 1. 训练Word2Vec词向量

```powershell
# 使用虚拟环境Python执行
.venv\Scripts\python.exe train_word2vec.py
```

**预期训练时间：约1小时**（取决于硬件配置）

### 2. 训练文本分类器

```powershell
# 训练基线CNN模型
.venv\Scripts\python.exe train_classifier.py --model cnn

# 训练BiLSTM-Attention模型（推荐）
.venv\Scripts\python.exe train_classifier.py --model bilstm-attention
```

**预期训练时间：约4小时**（完整训练集）

### 3. 测试高级模型

```powershell
# 测试高级模型架构
.venv\Scripts\python.exe test_advanced_models.py

# 测试增强训练技术
.venv\Scripts\python.exe test_enhanced_training.py

# 测试高级优化器
.venv\Scripts\python.exe test_advanced_optimizers.py
```

### 4. 运行完整测试

```powershell
# 运行所有测试
.venv\Scripts\python.exe test_all.py
```

---

## 模型性能对比

| 模型 | 测试准确率 | 训练时间 |
|------|------------|----------|
| 基线CNN（原项目） | 94.65% | ~4小时 |
| Focal Loss CNN | 95.87% | ~4小时 |
| 多尺度TextCNN | 96.33% | ~4.5小时 |
| BiLSTM-Attention ⭐ | **96.73%** | ~5小时 |
| 模型集成（投票） | 96.70% | ~10小时（含训练多个模型） |

---

## 配置说明

### 配置文件位置

配置项集中在 `src/config.py`：

```python
# Word2Vec配置
WORD2VEC_CONFIG = {
    'embedding_dim': 128,      # 词向量维度
    'window_size': 5,          # 窗口大小
    'min_count': 5,            # 最小词频
    'negative': 5,             # 负采样数
    'learning_rate': 0.001,    # 学习率
    'batch_size': 128,         # 批次大小
    'epochs': 10               # 训练轮数
}

# 分类器配置
CLASSIFIER_CONFIG = {
    'max_seq_len': 20,         # 序列最大长度
    'test_size': 0.1,          # 测试集比例
    'val_size': 0.15,          # 验证集比例
    'batch_size': 16,          # 批次大小
    'epochs': 15,              # 训练轮数
    'learning_rate': 0.001     # 学习率
}
```

### 关键配置项说明

| 配置项 | 作用 | 建议值 |
|--------|------|--------|
| `embedding_dim` | 词向量维度 | 128/256 |
| `max_seq_len` | 文本序列长度 | 20-50 |
| `learning_rate` | 初始学习率 | 0.001 |
| `batch_size` | 训练批次 | 16-64 |
| `epochs` | 训练轮数 | 10-20 |

---

## 常见问题

### Q1: 训练时出现内存不足

**解决方案：**
- 减小 `batch_size` 参数（如从128降至64）
- 增加虚拟内存（Windows）
- 使用更小的 `max_seq_len`

### Q2: Word2Vec训练loss异常波动

**解决方案：**
- 确认使用NCE损失而非softmax
- 降低学习率至0.001以下
- 增加训练数据量

### Q3: 模型准确率不达标

**解决方案：**
- 检查数据集路径是否正确
- 确保词向量模型已正确训练
- 尝试使用BiLSTM-Attention模型

---

## 项目贡献

### 代码规范

- Python代码遵循PEP8规范
- 使用类型注解
- 函数/类需添加文档字符串

### 新增功能流程

1. 在 `src/` 目录下创建新模块
2. 在 `src/config.py` 中添加配置项
3. 编写对应的测试脚本
4. 更新README文档

---

## 许可证

MIT License

---

## 联系方式

如有问题，请提交Issue或联系项目维护者。
