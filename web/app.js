const state = { busy: false };
const AUTO_REFRESH_SECONDS = 60;
const EVIDENCE_LABELS = {
  sufficient: '证据充分',
  review: '需复核',
  insufficient: '证据不足',
};
const ANSWER_REVIEW_LABELS = {
  sufficient: '答案结构完整',
  review: '答案需复核',
  insufficient: '答案证据不足',
  unscored: '未配置答案审阅组',
};
const EVIDENCE_REASONS = {
  no_results: '没有召回到可用于回答的片段。',
  low_relevance: '最高关联分数偏低，当前片段不足以支撑可靠回答。',
  borderline_relevance: '片段可能相关，但证据强度处于边界，建议检查原文。',
  ambiguous_top_results: '前几个片段分数接近，检索结果存在歧义，建议检查来源。',
  strong_relevance: '最高关联分数较高，且与下一候选有明显差距。',
  heuristic_only: '排序结果较强，但当前使用的是备用规则，建议核对来源原文。',
};
let documentTree = { directories: Object.create(null), files: [], fileCount: 0 };
let currentDirectory = [];
let secondsUntilAutoRefresh = AUTO_REFRESH_SECONDS;

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value).replace(/[&<>"]/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
}[char]));

function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll('button').forEach((button) => { button.disabled = busy; });
}

function setProgress(message, percent, visible = true) {
  $('#progress-area').hidden = !visible;
  $('#progress-message').textContent = message;
  $('#progress-percent').textContent = `${percent}%`;
  $('#progress-bar').style.width = `${percent}%`;
}

function reportText(report) {
  const integrity = report.index_integrity_status === 'ok' ? '索引完整' : `索引需检查 ${report.index_integrity_issue_count ?? 0} 项`;
  const data = report.data_integrity_status === 'ok' ? '数据完整' : `数据${report.data_integrity_status === 'warning' ? '有警告' : '有错误'}`;
  return `扫描 ${report.scanned_files} 个文件，新增 ${report.added_files}，更新 ${report.updated_files}，删除 ${report.removed_files}，跳过 ${report.skipped_files}；当前共 ${report.total_chunks} 个向量片段；${data}，${integrity}。`;
}

function buildDocumentTree(files, directories) {
  const root = { directories: Object.create(null), files: [], fileCount: 0 };

  const getOrCreateDirectory = (parts) => {
    let node = root;
    parts.forEach((part) => {
      if (!node.directories[part]) {
        node.directories[part] = { directories: Object.create(null), files: [], fileCount: 0 };
      }
      node = node.directories[part];
    });
    return node;
  };

  directories
    .sort((left, right) => left.split('/').length - right.split('/').length || left.localeCompare(right, 'zh-CN'))
    .forEach((directory) => getOrCreateDirectory(directory.split('/')));

  files.forEach((file) => {
    const parts = file.source.split('/');
    let node = root;
    node.fileCount += 1;
    parts.slice(0, -1).forEach((part) => {
      if (!node.directories[part]) {
        node.directories[part] = { directories: Object.create(null), files: [], fileCount: 0 };
      }
      node = node.directories[part];
      node.fileCount += 1;
    });
    node.files.push(file);
  });
  return root;
}

function currentDirectoryNode() {
  let node = documentTree;
  for (const part of currentDirectory) {
    if (!node.directories[part]) {
      currentDirectory = [];
      return documentTree;
    }
    node = node.directories[part];
  }
  return node;
}

function renderDocumentBrowser() {
  const node = currentDirectoryNode();
  const browser = $('#directory-browser');
  const backButton = $('#directory-back-button');
  browser.replaceChildren();
  $('#directory-path').textContent = `data/${currentDirectory.length ? `${currentDirectory.join('/')}/` : ''}`;
  backButton.disabled = currentDirectory.length === 0;

  const directoryNames = Object.keys(node.directories).sort((left, right) => left.localeCompare(right, 'zh-CN'));
  if (directoryNames.length) {
    const grid = document.createElement('div');
    grid.className = 'directory-grid';
    directoryNames.forEach((name) => {
      const entry = document.createElement('button');
      entry.className = 'directory-entry';
      entry.type = 'button';
      const title = document.createElement('strong');
      const metadata = document.createElement('span');
      title.textContent = name;
      metadata.textContent = `${node.directories[name].fileCount} 个文档`;
      entry.append(title, metadata);
      entry.addEventListener('click', () => {
        currentDirectory.push(name);
        renderDocumentBrowser();
      });
      grid.appendChild(entry);
    });
    browser.appendChild(grid);
  }

  if (node.files.length) {
    const files = document.createElement('div');
    files.className = 'directory-files';
    [...node.files]
      .sort((left, right) => left.source.localeCompare(right.source, 'zh-CN'))
      .forEach((file) => {
        const row = document.createElement('article');
        row.className = 'file-entry';
        const icon = document.createElement('div');
        const details = document.createElement('div');
        const name = document.createElement('div');
        const metadata = document.createElement('div');
        const hash = document.createElement('div');
        icon.className = 'file-icon';
        details.className = 'file-name';
        metadata.className = 'file-meta';
        hash.className = 'file-hash';
        icon.textContent = file.source.split('.').pop().toUpperCase();
        name.textContent = file.source.split('/').pop();
        metadata.textContent = `${file.characters.toLocaleString()} 字符 · ${file.modified_at}`;
        hash.textContent = `#${file.hash}`;
        details.append(name, metadata);
        row.append(icon, details, hash);
        files.appendChild(row);
      });
    browser.appendChild(files);
  }

  if (!directoryNames.length && !node.files.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-directory';
    empty.textContent = '此目录尚无可索引文档。';
    browser.appendChild(empty);
  }
}

function updateDocumentTree(files, directories) {
  documentTree = buildDocumentTree(files, directories);
  if ($('#document-dialog').open) renderDocumentBrowser();
}

function openDocumentBrowser() {
  currentDirectory = [];
  renderDocumentBrowser();
  $('#document-dialog').showModal();
}

function closeDocumentBrowser() {
  const dialog = $('#document-dialog');
  if (dialog.open) dialog.close();
}

async function refreshStatus() {
  const response = await fetch('/api/status');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '无法读取状态');
  $('#document-count').textContent = data.documents.toLocaleString();
  $('#chunk-count').textContent = data.chunks.toLocaleString();
  $('#source-count').textContent = data.indexed_sources.toLocaleString();
  $('#embedding-provider').textContent = data.embedding_languages || (data.embedding_provider === 'local' ? '本地' : 'API');
  $('#embedding-provider').title = data.embedding_model || '';
  $('#collection-name').textContent = data.collection;
  $('#file-summary').textContent = `${data.documents} 个可索引文件`;
  const dataIntegrity = data.data_integrity || {};
  const indexIntegrity = data.index_integrity || {};
  const integrityLabel = dataIntegrity.status === 'ok' && indexIntegrity.status === 'ok'
    ? '数据完整，索引链路正常'
    : `数据 ${dataIntegrity.status || '--'}；索引 ${indexIntegrity.status || '--'}；问题 ${Number(dataIntegrity.issue_count || 0) + Number(indexIntegrity.issue_count || 0)}`;
  $('#integrity-status').textContent = `完整性：${integrityLabel}`;
  updateDocumentTree(data.files, data.directories);
  secondsUntilAutoRefresh = AUTO_REFRESH_SECONDS;
  updateAutoRefreshStatus();
}

function updateAutoRefreshStatus() {
  $('#auto-refresh-status').textContent = `自动刷新：每 1 分钟 · ${secondsUntilAutoRefresh} 秒后`;
}

function runAutoRefreshClock() {
  secondsUntilAutoRefresh -= 1;
  if (secondsUntilAutoRefresh > 0) {
    updateAutoRefreshStatus();
    return;
  }
  secondsUntilAutoRefresh = AUTO_REFRESH_SECONDS;
  if (!state.busy) {
    refreshStatus().catch((error) => { $('#index-report').textContent = `自动刷新失败：${error.message}`; });
  }
  updateAutoRefreshStatus();
}

function renderEvidence(evidence, retryCount = 0) {
  const panel = $('#evidence-panel');
  if (!evidence) {
    panel.hidden = true;
    return;
  }
  const status = evidence.status || 'review';
  panel.className = `evidence-panel status-${status}`;
  panel.hidden = false;
  $('#evidence-status').textContent = EVIDENCE_LABELS[status] || status;
  const providerLabels = {
    cross_encoder: 'Cross-Encoder Reranker',
    heuristic_fallback: '关键词 + 向量备用排序',
    disabled: '未启用 Reranker',
  };
  $('#evidence-provider').textContent = providerLabels[evidence.reranker_provider] || evidence.reranker_provider || '';
  let reason = EVIDENCE_REASONS[evidence.reason_code] || '请检查召回片段和来源。';
  if (!evidence.reranker_enabled && evidence.reranker_provider === 'heuristic_fallback') {
    reason += ' 当前未使用训练型 Reranker，结果仅供复核。';
  }
  if (evidence.reranker_error) {
    reason += ` Reranker 加载信息：${String(evidence.reranker_error).slice(0, 150)}`;
  }
  $('#evidence-reason').textContent = reason;
  const metrics = $('#evidence-metrics');
  metrics.replaceChildren();
  const values = [
    `候选 ${evidence.candidate_count ?? 0}`,
    `最高分 ${evidence.top_score == null ? '--' : Number(evidence.top_score).toFixed(3)}`,
    `分差 ${evidence.score_gap == null ? '--' : Number(evidence.score_gap).toFixed(3)}`,
    `向量 ${evidence.vector_ms ?? 0} ms`,
    `重排 ${evidence.rerank_ms ?? 0} ms`,
  ];
  if (retryCount > 0) values.push(`query retry ${retryCount}`);
  values.forEach((value) => {
    const item = document.createElement('span');
    item.textContent = value;
    metrics.appendChild(item);
  });
}

function renderAnswerReview(review) {
  const panel = $('#answer-review-panel');
  if (!review || review.status === 'unscored') {
    panel.hidden = true;
    return;
  }
  const status = review.status || 'review';
  panel.className = `answer-review-panel status-${status}`;
  panel.hidden = false;
  $('#answer-review-status').textContent = ANSWER_REVIEW_LABELS[status] || status;
  const covered = Number(review.covered_count || 0);
  const groups = Number(review.group_count || 0);
  $('#answer-review-coverage').textContent = groups
    ? `概念组 ${covered}/${groups} · ${(Number(review.coverage || 0) * 100).toFixed(1)}%`
    : '';
  $('#answer-review-reason').textContent = review.reason || '请检查答案和来源片段。';
  const missing = (review.missing_groups || [])
    .map((group) => group[0])
    .filter(Boolean);
  $('#answer-review-missing').textContent = missing.length
    ? `待复核概念：${missing.join('；')}`
    : '所有必需概念组均已覆盖。';
}

function renderResults(results, retrievalMs, answerMs, evidence, contextResults = []) {
  const list = $('#retrieval-list');
  list.replaceChildren();
  const summary = $('#result-summary');
  summary.hidden = false;
  const rerankMs = evidence?.rerank_ms == null ? null : ` · 重排 ${evidence.rerank_ms} ms`;
  const expanded = contextResults.filter((result) => (result.context_role || 'primary') !== 'primary');
  summary.textContent = `召回 ${results.length} 个片段 · 上下文扩展 ${expanded.length} 个 · 总计 ${retrievalMs} ms${rerankMs || ''}${answerMs == null ? '' : ` · 生成 ${answerMs} ms`}`;
  const template = $('#result-template');
  const appendResult = (result, rank, expandedResult = false) => {
    const row = template.content.cloneNode(true);
    if (expandedResult) row.querySelector('.result-row').classList.add('context-row');
    row.querySelector('.result-rank').textContent = expandedResult ? '↳' : (rank || result.rank);
    const role = result.context_role || (expandedResult ? 'context' : 'primary');
    row.querySelector('.result-meta').textContent = `${result.source} · chunk ${result.chunk_index} · ${role}`;
    row.querySelector('.result-content').textContent = result.content;
    const vectorLabel = expandedResult
      ? '扩展上下文 --'
      : (result.retrieval_method === 'lexical' ? '关键词 --' : `向量 ${Number(result.vector_score ?? result.score).toFixed(3)}`);
    row.querySelector('.score-vector').textContent = vectorLabel;
    row.querySelector('.score-rerank').textContent = expandedResult ? '不参与排序' : (result.rerank_score == null
      ? '重排 --'
      : `重排 ${Number(result.rerank_score).toFixed(3)}`);
    if (!expandedResult && result.fusion_score != null) {
      row.querySelector('.score-rerank').textContent += ` · RRF ${Number(result.fusion_score).toFixed(4)}`;
    }
    list.appendChild(row);
  };
  results.forEach((result, index) => appendResult(result, index + 1));
  expanded.forEach((result) => appendResult(result, null, true));
}

async function runIndex(force) {
  if (state.busy) return;
  setBusy(true);
  setProgress('正在建立同步连接', 2);
  $('#index-report').textContent = '同步进行中...';
  const stream = new EventSource(`/api/index-stream?force=${force ? '1' : '0'}`);
  stream.addEventListener('progress', (event) => {
    const data = JSON.parse(event.data);
    setProgress(data.message, data.percent);
  });
  stream.addEventListener('complete', async (event) => {
    const data = JSON.parse(event.data);
    setProgress('索引同步完成', 100);
    $('#index-report').textContent = reportText(data.report);
    stream.close();
    setBusy(false);
    await refreshStatus();
  });
  stream.addEventListener('error', (event) => {
    let message = '索引同步连接失败';
    if (event.data) message = JSON.parse(event.data).message;
    $('#index-report').textContent = message;
    stream.close();
    setBusy(false);
  });
}

async function query(mode) {
  if (state.busy) return;
  const question = $('#question').value.trim();
  if (!question) { $('#query-status').textContent = '请输入问题后再执行。'; return; }
  setBusy(true);
  $('#query-status').textContent = mode === 'ask' ? '正在检索并生成回答...' : '正在检索向量库...';
  $('#answer-panel').hidden = true;
  $('#answer-review-panel').hidden = true;
  $('#evidence-panel').hidden = true;
  try {
    const response = await fetch(`/api/${mode === 'ask' ? 'ask' : 'retrieve'}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: $('#top-k').value, sync: $('#ask-sync').checked })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '请求失败');
    renderEvidence(data.evidence, data.retry_count || 0);
    renderResults(data.results, data.retrieval_ms, data.answer_ms, data.evidence, data.context_results || []);
    if (mode === 'ask') {
      $('#answer-panel').hidden = false;
      $('#answer-text').textContent = data.answer;
      renderAnswerReview(data.answer_review);
      const evidenceLabel = EVIDENCE_LABELS[data.evidence?.status] || '结果已返回';
      const boundary = data.query_boundary;
      const boundaryLabel = boundary && boundary.status === 'blocked'
        ? `问题边界已拦截（${boundary.reason_code}）`
        : evidenceLabel;
      $('#query-status').textContent = data.sync_report ? `${reportText(data.sync_report)} · ${boundaryLabel}` : `问答完成 · ${boundaryLabel}`;
    } else {
      const evidenceLabel = EVIDENCE_LABELS[data.evidence?.status] || '结果已返回';
      $('#query-status').textContent = `检索完成 · ${evidenceLabel}；请检查下面的来源片段。`;
    }
    await refreshStatus();
  } catch (error) {
    $('#query-status').textContent = `操作失败：${error.message}`;
  } finally {
    setBusy(false);
  }
}

function metricValue(value, formatter = (input) => input) {
  return value == null ? '未生成' : formatter(value);
}

function renderEvaluation(report) {
  const summary = report.summary;
  const summaryPanel = $('#evaluation-summary');
  const metrics = [
    ['来源命中率', metricValue(summary.hit_rate_at_k, (value) => `${(value * 100).toFixed(1)}%`)],
    ['MRR', metricValue(summary.mrr, (value) => value.toFixed(4))],
    ['Answer point coverage', metricValue(summary.answer_point_coverage, (value) => `${(value * 100).toFixed(1)}%`)],
    ['Answer point hit rate', metricValue(summary.answer_point_hit_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['归一化要点覆盖', metricValue(summary.answer_point_normalized_coverage, (value) => `${(value * 100).toFixed(1)}%`)],
    ['归一化要点命中', metricValue(summary.answer_point_normalized_hit_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['Query retries', `${summary.retrieval_retry_count ?? 0}`],
    ['平均上下文片段', `${summary.avg_context_result_count ?? 0}`],
    ['平均 parent / neighbor', `${summary.avg_parent_context_count ?? 0} / ${summary.avg_neighbor_context_count ?? 0}`],
    ['检索 P95', `${summary.retrieval_p95_ms} ms`],
    ['平均 Top-1 分数', metricValue(summary.avg_top_score, (value) => value.toFixed(4))],
    ['回答数量', `${summary.answers_generated} / ${summary.total_cases}`],
    ['模型调用次数', `${summary.model_answer_calls ?? 0}`],
    ['关键词命中', metricValue(summary.keyword_hit_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['拒答准确率', metricValue(summary.abstention_hit_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['证据充分', metricValue(summary.evidence_sufficient_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['需复核', metricValue(summary.evidence_review_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['证据不足', metricValue(summary.evidence_insufficient_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['平均检索', `${summary.retrieval_avg_ms} ms`],
    ['检索 P50', `${summary.retrieval_p50_ms} ms`],
    ['平均答案', metricValue(summary.answer_avg_ms, (value) => `${value} ms`)],
    ['关键词组覆盖', metricValue(summary.answer_keyword_group_coverage, (value) => `${(value * 100).toFixed(1)}%`)],
    ['关键词组命中', metricValue(summary.answer_keyword_group_hit_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['答案结构充分', metricValue(summary.answer_review_sufficient_rate, (value) => `${(value * 100).toFixed(1)}%`)],
    ['答案结构需复核', metricValue(summary.answer_review_review_rate, (value) => `${(value * 100).toFixed(1)}%`)],
  ];
  summaryPanel.replaceChildren();
  metrics.forEach(([label, value]) => {
    const card = document.createElement('article');
    const caption = document.createElement('span');
    const amount = document.createElement('strong');
    caption.textContent = label;
    amount.textContent = value;
    card.append(caption, amount);
    summaryPanel.appendChild(card);
  });
  summaryPanel.hidden = false;

  const cases = $('#evaluation-cases');
  cases.replaceChildren();
  report.cases.forEach((item) => {
    const detail = document.createElement('details');
    detail.className = 'evaluation-case';
    const title = document.createElement('summary');
    const status = item.source_hit == null
      ? (item.abstention_hit === true ? '拒答正确' : item.abstention_hit === false ? '拒答误答' : '拒答题')
      : item.source_hit ? '来源命中' : '来源未命中';
    const evidenceLabel = EVIDENCE_LABELS[item.evidence_status] || '未判断';
    title.textContent = `[${item.category}] ${status} · ${evidenceLabel} · ${item.question}`;
    const source = document.createElement('p');
    source.textContent = `期望来源：${item.expected_sources.join('、') || '无（应拒答）'}；召回来源：${item.retrieved_sources.join('、')}；首个命中排名：${item.first_expected_rank ?? '-'}；证据：${evidenceLabel}；重排：${item.rerank_score == null ? '--' : Number(item.rerank_score).toFixed(3)}；检索：${item.retrieval_ms} ms；上下文 ${item.context_result_count ?? 0}（parent ${item.parent_context_count ?? 0} / neighbor ${item.neighbor_context_count ?? 0}）`;
    detail.append(title, source);
    if (item.answer_point_coverage != null) {
      const points = document.createElement('p');
      const normalized = item.answer_point_normalized_coverage == null
        ? ''
        : ` · normalized ${(Number(item.answer_point_normalized_coverage) * 100).toFixed(1)}% · covered ${item.answer_point_normalized_covered?.filter(Boolean).length ?? 0}/${item.answer_points?.length ?? 0}`;
      points.textContent = `Answer point coverage: ${(Number(item.answer_point_coverage) * 100).toFixed(1)}% · covered ${item.answer_point_covered?.filter(Boolean).length ?? 0}/${item.answer_points?.length ?? 0}${normalized}`;
      detail.append(points);
    }
    if (item.answer_keyword_group_coverage != null) {
      const keywords = document.createElement('p');
      keywords.textContent = `关键词组覆盖：${(Number(item.answer_keyword_group_coverage) * 100).toFixed(1)}% · 命中 ${item.answer_keyword_group_covered?.filter(Boolean).length ?? 0}/${item.answer_keyword_groups?.length ?? 0}`;
      detail.append(keywords);
    }
    if (item.answer_review_status) {
      const review = document.createElement('p');
      const missing = (item.answer_review_missing_groups || [])
        .map((group) => group[0])
        .filter(Boolean);
      review.textContent = `答案审阅：${ANSWER_REVIEW_LABELS[item.answer_review_status] || item.answer_review_status} · ${item.answer_review_reason || ''}`
        + (missing.length ? ` · 待复核：${missing.join('、')}` : '');
      detail.append(review);
    }
    if (item.retry_count) {
      const retry = document.createElement('p');
      retry.textContent = `Query rewrite retry: ${item.retry_count}`;
      detail.append(retry);
    }
    if (item.reference_answer) {
      const reference = document.createElement('p');
      reference.className = 'evaluation-reference';
      reference.textContent = `参考答案：${item.reference_answer}`;
      detail.appendChild(reference);
    }
    if (item.answer) {
      const answer = document.createElement('pre');
      answer.textContent = item.answer;
      detail.appendChild(answer);
    }
    if (item.answer_error) {
      const error = document.createElement('p');
      error.className = 'metric-warn';
      error.textContent = `回答失败：${item.answer_error}`;
      detail.appendChild(error);
    }
    cases.appendChild(detail);
  });
}

async function runEvaluation() {
  if (state.busy) return;
  const includeAnswers = $('#eval-answers').checked;
  setBusy(true);
  $('#evaluation-status').textContent = includeAnswers ? '正在运行评测并请求 Chat 模型...' : '正在运行检索评测...';
  try {
    const response = await fetch('/api/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: $('#eval-dataset').value,
        top_k: $('#eval-top-k').value,
        include_answers: includeAnswers,
      })
    });
    const report = await response.json();
    if (!response.ok) throw new Error(report.error || '评测请求失败');
    renderEvaluation(report);
    $('#evaluation-status').textContent = `评测完成：${report.summary.total_cases} 题。参数：chunk ${report.parameters.chunk_size}/${report.parameters.chunk_overlap}，Top-K ${report.parameters.top_k}；结果已保存到 ${report.report_path}`;
  } catch (error) {
    $('#evaluation-status').textContent = `评测失败：${error.message}`;
  } finally {
    setBusy(false);
  }
}

function updateEvaluationDatasetHint() {
  const descriptions = {
    'eval/questions_graham_150.jsonl': '当前选择 Graham 150 题：125 道知识库问题、25 道应拒答问题；结果会保存到 eval/results/。',
    'eval/questions_graham_85.jsonl': '当前选择 Graham 85 题：65 道知识库问题、20 道应拒答问题；结果会保存到 eval/results/。',
    'eval/questions.jsonl': '当前选择原始 15 题基线；结果会保存到 eval/results/。',
  };
  $('#evaluation-status').textContent = descriptions[$('#eval-dataset').value] || '当前选择自定义评测集；结果会保存到 eval/results/。';
}

$('#refresh-button').addEventListener('click', () => refreshStatus().catch((error) => { $('#index-report').textContent = error.message; }));
$('#sync-button').addEventListener('click', () => runIndex(false));
$('#force-button').addEventListener('click', () => runIndex(true));
$('#retrieve-button').addEventListener('click', () => query('retrieve'));
$('#ask-button').addEventListener('click', () => query('ask'));
$('#evaluate-button').addEventListener('click', runEvaluation);
$('#eval-dataset').addEventListener('change', updateEvaluationDatasetHint);
$('#browse-documents-button').addEventListener('click', openDocumentBrowser);
$('#close-documents-button').addEventListener('click', closeDocumentBrowser);
$('#directory-back-button').addEventListener('click', () => {
  currentDirectory.pop();
  renderDocumentBrowser();
});
$('#document-dialog').addEventListener('click', (event) => {
  if (event.target === $('#document-dialog')) closeDocumentBrowser();
});

updateAutoRefreshStatus();
updateEvaluationDatasetHint();
setInterval(runAutoRefreshClock, 1000);
refreshStatus().catch((error) => { $('#index-report').textContent = `无法加载控制台：${error.message}`; });
