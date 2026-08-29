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

在自建的中英跨语言开发/校准集上，当前代码的受控 Top-3 回归结果如下：

| 指标 | 当前回归 |
|---|---:|
| 域内 Recall@3 | **63 / 65（96.92%）** |
| MRR | **0.9128** |
| 检索层越界拒答 | **18 / 20（90.00%）** |
| 综合证据决策 | **82 / 85（96.47%）** |
| 检索延迟 P50 / P95 | 2533.1 / 5040.3 ms |

这些数字来自 2026-08-29 对当前代码的完整回归，衡量的是特定语料和开发/校准题集上的检索与拒答表现。该题集参与过阈值、术语扩展和重试策略调整，因此不是 blind test，也不等同于通用回答准确率。当前仍有 2 个来源召回失败和 3 个证据决策失败，项目不会通过继续添加逐题规则来掩盖它们。原始第三方文章和本地生成的逐题报告不随仓库分发；你可以使用自己的合法语料复现实验，评测题与评分脚本保留在 `eval/`。

## 第二领域：Resume Fit AI

仓库包含一个可公开复现的第二领域实验：把 Resume Fit AI 的产品工作流、结构化生成、异步任务、安全隐私和部署边界整理成 5 篇第一方技术文档，并使用独立索引运行 22 道迁移评测题。这个知识包不包含用户简历、JD、API Key、生产地址、模型输出或 Resume Fit AI 的私有源码。

```powershell
# 默认运行 Hybrid + Cross-Encoder Reranker
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py

# Dense / Hybrid / Hybrid + Reranker 消融对比
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py --mode all
```

该实验默认关闭为 Graham 语料编写的 Query Expansion 词表，并使用隔离的 Chroma collection，目的是检查通用索引、BM25/RRF 和重排链路能否迁移到另一个工程领域。题目和答案均已公开，因此它被明确标记为 **transfer/acceptance dataset**，不是 blind test，也不用于宣称未见数据泛化。语料、运行说明和可提交的紧凑报告见 [`examples/resume_fit_ai/`](./examples/resume_fit_ai/) 与 [`eval/reports/resume_fit_ai_transfer_v1.json`](./eval/reports/resume_fit_ai_transfer_v1.json)。

本机受控 Top-3 运行结果如下；18 个域内问题和 4 个越界问题均不调用 Chat API：

| 检索模式 | Recall@3 | MRR | 证据决策准确率 | P50 延迟 |
|---|---:|---:|---:|---:|
| Dense | 100% | 0.9259 | 18.18%（保守地全部拒答） | 12.4 ms |
| Dense + BM25/RRF | 100% | 0.9444 | 18.18%（保守地全部拒答） | 13.7 ms |
| Dense + BM25/RRF + Cross-Encoder | 100% | **1.0000** | **100%（22/22）** | 957.0 ms |

“证据决策”要求同时接受 18 个域内问题并拒答 4 个越界问题，避免只看越界拒答而忽略“全部拒答”的退化方案。该 5 文档语料把统一候选预算从通用配置的最多 18 个收紧为 6 个；本次记录中 Cross-Encoder P50 从优化前的 1258.2 ms 降到 957.0 ms，Recall@3、MRR 与证据决策不变。延迟包含首次模型加载和冷启动开销，具体数值依赖 CPU、模型缓存和候选数量；这个小型验收集主要验证跨领域链路与排名行为，不用于证明生产吞吐或统计显著性。

在上述开发与迁移验证完成后，项目另外冻结了一个不参与调参的 30 题前瞻测试集。2026-08-29 的首次且唯一一次 Top-3 运行中，20 个域内题全部命中且 MRR 为 1.0000；10 个越界题正确拒绝 8 个，综合证据决策为 24/30。4 个失败是“来源排第 1 但证据门控仍拒答”，另 2 个是把执行简历改写的请求和实时付费用户问题判为需复核。运行后没有继续调阈值或添加规则。冻结清单、校验和、逐类归因和紧凑报告见 [`eval/frozen/`](./eval/frozen/)、[`eval/audits/`](./eval/audits/) 与 [`eval/reports/resume_fit_ai_frozen_v1.json`](./eval/reports/resume_fit_ai_frozen_v1.json)。

仓库还提供 12 题的小型回答审计（9 个应回答、3 个应拒答），在既有概念覆盖和拒答指标之外，检查非拒答是否列出来源、来源是否确实进入过检索上下文，并单独统计缺失引用和虚构引用。这个确定性检查只证明“引用来源可追溯”，不宣称能自动证明句子级忠实度。运行需要有效的 OpenAI-compatible Chat API；若任一回答调用失败，脚本只保留被 Git 忽略的本地详细报告，不发布不完整成绩。

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

# 开发/校准集失败样本归因（A-E 分桶；文件名保留 legacy holdout）
.\.venv\Scripts\python.exe scripts\diagnose_holdout_failures.py `
  --dataset eval\questions_graham_holdout.jsonl -k 3 --wide-k 10

# 同时生成回答（会产生 API 调用费用）
.\.venv\Scripts\python.exe -m src.evaluation `
  --dataset eval\questions_graham_85.jsonl -k 3 --with-answers

# Resume Fit AI 小型回答/引用/拒答审计（会产生 API 调用费用）
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py `
  --dataset answers --mode rerank --with-answers --publish-summary
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
examples/resume_fit_ai/
                   可公开的第二领域知识包与复现实验说明
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
