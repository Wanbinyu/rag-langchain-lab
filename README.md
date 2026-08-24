# RAG LangChain Lab

一个面向可验证检索的本地 RAG 实验项目。它不只演示“切分—向量化—问答”，还覆盖增量索引、混合召回、跨语言重排、证据分级、拒答和离线评测，适合用来研究 RAG 的检索质量与工程取舍。

[![Tests](https://github.com/Wanbinyu/rag-langchain-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Wanbinyu/rag-langchain-lab/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 解决什么问题

普通 RAG Demo 往往只展示一次向量检索，难以回答“为什么召回这段”“没有证据时怎么办”“改动一篇文档是否需要全量重建”等问题。本项目围绕这些问题实现了一条可观察、可评测的检索链路：

- **内容哈希增量索引**：新增、修改、删除文档时只更新受影响的片段。
- **混合候选召回**：组合向量检索、精确词匹配与 BM25，使用 RRF 合并候选。
- **跨语言 Cross-Encoder 重排**：使用多语言 Reranker 对候选进行二阶段排序；不可用时显式回退。
- **受控 Query Rewrite / Retry**：仅在证据不足且仍有召回希望时触发二次检索，限制额外延迟。
- **证据分级与拒答**：输出 `证据充分 / 需复核 / 证据不足`，证据不足时拒绝生成无依据答案。
- **可追溯引用**：展示来源、chunk、分数差距以及检索和生成耗时。
- **离线评测**：统计 Recall@K、MRR、证据状态、拒答表现，并支持失败样本归因。
- **CLI + 本地 Web 控制台**：既可脚本化运行，也可在浏览器中检查文档、索引和召回结果。

## 检索流程

```text
本地文档
  -> 内容哈希增量索引 -> Chroma
  -> Vector / Exact-term / BM25 候选
  -> RRF 合并与来源约束
  -> Multilingual Cross-Encoder Rerank
  -> 证据分级
       ├─ 充分/需复核 -> 携带引用生成回答
       └─ 不足       -> 受控改写重试或拒答
```

## 评测结果

在自建的中英跨语言问答集上，受控 Top-3 实验得到以下结果：

| 指标 | 基线 | 当前方案 |
|---|---:|---:|
| Recall@3 | 60.00% | **84.62%** |
| MRR | 0.5231 | **0.7564** |
| Recall@5 | - | **89.23%** |
| 越界问题拒答 | - | **20 / 20** |

这些数字衡量的是特定语料和题集上的检索/拒答表现，不等同于通用回答准确率。原始第三方文章和本地生成的逐题报告不随仓库分发；你可以使用自己的合法语料复现实验，评测题与评分脚本保留在 `eval/`。

## 快速开始

要求 Python 3.11+。以下命令以 PowerShell 为例；macOS/Linux 将激活命令替换为 `source .venv/bin/activate`。

```powershell
git clone https://github.com/Wanbinyu/rag-langchain-lab.git
cd rag-langchain-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少配置一个 OpenAI 兼容的 Chat API：

```env
OPENAI_API_KEY=your-key
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_CHAT_MODEL=deepseek-chat
EMBEDDING_PROVIDER=local
```

API Key 只应放在本地 `.env`；该文件已被 Git 忽略。Embedding 默认可在本地运行，首次使用时会下载模型。

把你有权使用的 `.txt` 或 `.md` 文档放入 `data/`，然后运行：

```powershell
# 增量同步索引
.\.venv\Scripts\python.exe examples\01_basic_rag.py index

# 查看索引状态（不调用模型）
.\.venv\Scripts\python.exe examples\01_basic_rag.py status

# 单次提问；--sync 会先同步变化的文档
.\.venv\Scripts\python.exe examples\01_basic_rag.py ask --sync "你的问题"

# 连续问答
.\.venv\Scripts\python.exe examples\01_basic_rag.py chat --sync -k 3
```

也可以启动本地控制台：

```powershell
.\run_dashboard.ps1
```

浏览器访问 `http://127.0.0.1:8765`。控制台可以检查文档、同步索引、只运行检索，或在明确点击后调用 Chat API 生成答案。

## 评测与诊断

评测默认不调用 Chat API：

```powershell
# 检索指标
.\.venv\Scripts\python.exe -m src.evaluation `
  --dataset eval\questions_graham_85.jsonl -k 3

# 失败样本归因（A-E 分桶）
.\.venv\Scripts\python.exe scripts\diagnose_holdout_failures.py `
  --dataset eval\questions_graham_holdout.jsonl -k 3 --wide-k 10

# 同时生成回答（会产生 API 调用费用）
.\.venv\Scripts\python.exe -m src.evaluation `
  --dataset eval\questions_graham_85.jsonl -k 3 --with-answers
```

若使用仓库内的 Graham 题集，需要自行准备有权使用的对应文章，并保持 `source` 文件名与题集一致。评测报告会写入本地 `eval/results/`，该目录默认不提交。

二次检索默认使用规则改写（`RETRIEVAL_REWRITE_MODE=rule`）：只在证据为 `insufficient` 且分数未低至 hopeless 时触发；普通 `review` 不重试，以控制延迟。所有阈值都应针对自己的语料重新校准，不应把 Reranker 分数当作概率。

## 自动测试

测试不需要 API Key，也不会发起模型请求：

```powershell
python -m pip install -r requirements-ci.txt
python -m pytest -q
```

覆盖内容包括文档加载、内容哈希增量索引、检索融合、改写重试、证据状态、拒答与评测逻辑。GitHub Actions 会在推送和 Pull Request 时运行同一套测试。

## 项目结构

```text
src/
  loaders.py       文档扫描、读取与哈希
  indexing.py      切分、稳定 chunk ID、Chroma 增量同步
  retrieval.py     混合召回、重排、证据分级与重试
  chains.py        带证据边界的回答生成
  guardrails.py    问题边界与拒答判断
  evaluation.py    Recall/MRR/拒答/答案结构评测
  cli.py           index/status/ask/chat 命令
  web.py           本地控制台服务
web/               无框架的本地控制台前端
tests/             不调用外部 API 的自动测试
eval/              评测题、评分配置与诊断工具
scripts/           数据导入、审计与失败分析脚本
```

## 技术栈

Python、LangChain、ChromaDB、FastEmbed、Multilingual Cross-Encoder、BM25/RRF、OpenAI-compatible API、原生 HTML/CSS/JavaScript。

## 数据与许可

- 本仓库不分发第三方文章全文、EPUB、本地向量库或生成的逐题回答。
- 请只导入你有权使用的资料，并遵守模型与数据来源的许可。
- 本人原创的源代码以 [MIT License](LICENSE) 开放；第三方依赖和用户导入的数据仍适用各自许可。

## 安全

不要提交 `.env`、API Key、含敏感信息的知识库或评测输出。更多说明见 [注意事项-API密钥.md](./注意事项-API密钥.md)。
