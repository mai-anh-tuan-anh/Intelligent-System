/* Purpose: Drive the dashboard, local SVG charts, API requests, and presentation interactions. */
const dashboard = window.RNN_DASHBOARD || {};
const charts = {};

/* Purpose: Format large numeric values consistently for cards and tables. */
function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString('en-US');
}

/* Purpose: Convert a probability in [0, 1] into a readable percentage. */
function formatPercent(value, digits = 2) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

/* Purpose: Escape user/model text before inserting it into HTML. */
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

/* Purpose: Show a lightweight status message instead of interrupting the demo with alert(). */
function showToast(message, kind = 'info') {
  let toast = document.getElementById('demoToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'demoToast';
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.dataset.kind = kind;
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 4200);
}

/* Purpose: Switch visible sections when a sidebar button is clicked. */
function setupNavigation() {
  document.querySelectorAll('.nav-btn').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.page-section').forEach(section => section.classList.remove('active'));
      button.classList.add('active');
      document.getElementById(button.dataset.target)?.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

/* Purpose: Draw a dependency-free SVG chart so the demo still works when external JavaScript is unavailable. */
function drawChart(name, canvasId, config) {
  if (charts[name]) charts[name].destroy();
  const canvas = document.getElementById(canvasId);
  if (!canvas || !canvas.parentElement) return;

  const parent = canvas.parentElement;
  const existing = parent.querySelector(`svg[data-chart="${name}"]`);
  existing?.remove();
  canvas.style.display = 'none';

  const width = Math.max(420, parent.clientWidth || 700);
  const height = Math.max(260, canvas.clientHeight || 300);
  const margin = { top: 26, right: 24, bottom: 56, left: 58 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const ns = 'http://www.w3.org/2000/svg';

  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('data-chart', name);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', String(height));
  svg.style.display = 'block';

  const add = (tag, attrs = {}, text = '') => {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    if (text) el.textContent = text;
    svg.appendChild(el);
    return el;
  };

  const colors = ['#7c9cff', '#5ee0c1', '#f5b971', '#ff7183', '#b892ff'];
  const type = config.type || 'bar';
  const labels = config.data?.labels || [];
  const datasets = config.data?.datasets || [];

  if (!datasets.length || !labels.length) {
    add('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle', fill: '#94a2c4', 'font-size': '14' }, 'No chart data available');
    parent.appendChild(svg);
    charts[name] = { destroy: () => { svg.remove(); canvas.style.display = ''; } };
    return;
  }

  const allValues = datasets.flatMap(ds => (ds.data || []).map(Number)).filter(Number.isFinite);
  const max = Math.max(...allValues, 1);
  const min = Math.min(...allValues, 0);
  const range = Math.max(max - Math.min(0, min), 1e-9);
  const y = v => margin.top + innerH - ((v - Math.min(0, min)) / range) * innerH;
  const xStep = labels.length > 1 ? innerW / (labels.length - 1) : innerW;

  // Draw horizontal baseline/grid to make values easier to compare.
  add('line', { x1: margin.left, y1: margin.top + innerH, x2: margin.left + innerW, y2: margin.top + innerH, stroke: '#2a3650' });
  for (let i = 0; i <= 4; i++) {
    const yy = margin.top + (innerH * i / 4);
    const value = max - (range * i / 4);
    add('line', { x1: margin.left, y1: yy, x2: margin.left + innerW, y2: yy, stroke: 'rgba(255,255,255,.06)' });
    add('text', { x: margin.left - 8, y: yy + 4, 'text-anchor': 'end', fill: '#94a2c4', 'font-size': '10' }, Number(value).toFixed(Math.abs(value) < 10 ? 2 : 0));
  }

  if (type === 'line') {
    datasets.forEach((ds, dsIndex) => {
      const pts = (ds.data || []).map((value, i) => [margin.left + xStep * i, y(Number(value))]);
      if (!pts.length) return;
      add('polyline', {
        points: pts.map(p => p.join(',')).join(' '),
        fill: 'none',
        stroke: colors[dsIndex % colors.length],
        'stroke-width': 2.5,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round'
      });
      pts.forEach(([px, py]) => add('circle', { cx: px, cy: py, r: 3, fill: colors[dsIndex % colors.length] }));
    });
  } else if (type === 'doughnut') {
    const vals = datasets[0].data.map(v => Number(v));
    const total = Math.max(vals.reduce((a, b) => a + b, 0), 1e-9);
    const cx = margin.left + innerW * 0.42;
    const cy = margin.top + innerH * 0.48;
    const r = Math.min(innerW, innerH) * 0.30;
    const circumference = 2 * Math.PI * r;
    let offset = 0;
    vals.forEach((v, i) => {
      const length = circumference * (v / total);
      add('circle', {
        cx, cy, r,
        fill: 'none',
        stroke: colors[i % colors.length],
        'stroke-width': Math.max(28, r * 0.34),
        'stroke-dasharray': `${length} ${circumference - length}`,
        'stroke-dashoffset': String(-offset),
        transform: `rotate(-90 ${cx} ${cy})`
      });
      offset += length;
    });
    labels.forEach((label, i) => {
      const yy = margin.top + 18 + i * 20;
      add('rect', { x: margin.left + innerW * 0.70, y: yy - 9, width: 10, height: 10, rx: 2, fill: colors[i % colors.length] });
      add('text', { x: margin.left + innerW * 0.70 + 16, y: yy, fill: '#cfd7ee', 'font-size': '10' }, `${label}: ${formatNumber(vals[i])}`);
    });
  } else {
    const horizontal = config.options?.indexAxis === 'y';
    const vals = datasets[0].data.map(Number);
    if (horizontal) {
      const rowH = innerH / Math.max(labels.length, 1);
      const barGap = Math.max(4, rowH * 0.2);
      labels.forEach((label, i) => {
        const value = Math.max(0, vals[i] || 0);
        const barW = (value / max) * innerW;
        const yy = margin.top + i * rowH + barGap / 2;
        add('rect', { x: margin.left, y: yy, width: Math.max(1, barW), height: Math.max(8, rowH - barGap), rx: 5, fill: colors[i % colors.length], opacity: '.88' });
        add('text', { x: margin.left - 8, y: yy + Math.max(8, rowH - barGap) * 0.68, 'text-anchor': 'end', fill: '#cfd7ee', 'font-size': '10' }, String(label).slice(0, 16));
        add('text', { x: margin.left + barW + 6, y: yy + Math.max(8, rowH - barGap) * 0.68, fill: '#94a2c4', 'font-size': '10' }, formatNumber(value));
      });
    } else {
      const barW = Math.max(7, innerW / Math.max(labels.length, 1) * 0.65);
      labels.forEach((label, i) => {
        const value = Math.max(0, vals[i] || 0);
        const bh = (value / max) * innerH;
        const xx = margin.left + ((i + 0.5) * innerW / labels.length) - barW / 2;
        add('rect', { x: xx, y: margin.top + innerH - bh, width: barW, height: bh, rx: 4, fill: colors[i % colors.length], opacity: '.88' });
        add('text', { x: xx + barW / 2, y: margin.top + innerH + 18, 'text-anchor': 'middle', fill: '#cfd7ee', 'font-size': labels.length > 15 ? '8' : '10' }, String(label).slice(0, 11));
        if (labels.length <= 15) add('text', { x: xx + barW / 2, y: margin.top + innerH - bh - 7, 'text-anchor': 'middle', fill: '#94a2c4', 'font-size': '9' }, Number(value).toFixed(value < 10 ? 2 : 0));
      });
    }
  }

  // Legend for multi-series line charts.
  if (datasets.length > 1) {
    datasets.forEach((ds, i) => {
      const xx = margin.left + i * 125;
      add('line', { x1: xx, y1: height - 20, x2: xx + 18, y2: height - 20, stroke: colors[i % colors.length], 'stroke-width': 3 });
      add('text', { x: xx + 24, y: height - 16, fill: '#cfd7ee', 'font-size': '10' }, ds.label || `Series ${i + 1}`);
    });
  }

  parent.appendChild(svg);
  charts[name] = {
    destroy: () => {
      svg.remove();
      canvas.style.display = '';
    }
  };
}

/* Purpose: Render the main dashboard KPI cards from server-provided metadata. */
function renderOverviewCards() {
  const test = dashboard.optional_metrics?.test || {};
  const cards = [
    ['Vocabulary', formatNumber(dashboard.vocab_size), 'Output classes'],
    ['Embedding dim', formatNumber(dashboard.embedding_dim), 'Vector size per token'],
    ['Hidden size', formatNumber(dashboard.hidden_size), 'Recurrent state dimension'],
    ['Parameters', formatNumber(dashboard.total_parameters), 'Learnable parameters'],
    ['Sequence length', formatNumber(dashboard.sequence_length), 'Context window'],
    ['Best val loss', dashboard.best_validation_loss != null ? Number(dashboard.best_validation_loss).toFixed(4) : '—', 'Checkpoint selection'],
    ['Best val PPL', dashboard.best_validation_perplexity != null ? Number(dashboard.best_validation_perplexity).toFixed(2) : '—', 'exp(validation loss)'],
    ['Test Top-1', test.top1_accuracy !== undefined ? formatPercent(test.top1_accuracy) : '—', 'Exported test metric'],
  ];
  document.getElementById('overviewCards').innerHTML = cards.map(([label, value, sub]) => `<div class="metric-card"><div class="label">${label}</div><div class="value">${escapeHtml(value)}</div><div class="sub">${sub}</div></div>`).join('');
}

/* Purpose: Render training/validation loss, perplexity, and learning-rate curves when exported history exists. */
function renderTrainingCharts() {
  const history = dashboard.optional_metrics?.history;
  if (!history) {
    ['lossChart', 'pplChart', 'lrChart'].forEach(id => {
      const canvas = document.getElementById(id);
      if (canvas) canvas.style.display = 'none';
    });
    ['lossEmpty', 'pplEmpty', 'lrEmpty'].forEach(id => document.getElementById(id)?.classList.remove('hidden'));
    return;
  }
  const epochs = (history.train_loss || []).map((_, i) => i + 1);
  drawChart('loss', 'lossChart', { type: 'line', data: { labels: epochs, datasets: [
    { label: 'Train loss', data: history.train_loss },
    { label: 'Validation loss', data: history.val_loss }
  ] } });
  drawChart('ppl', 'pplChart', { type: 'line', data: { labels: epochs, datasets: [
    { label: 'Train PPL', data: history.train_ppl },
    { label: 'Validation PPL', data: history.val_ppl }
  ] } });
  drawChart('lr', 'lrChart', { type: 'line', data: { labels: epochs, datasets: [
    { label: 'Learning rate', data: history.lr }
  ] } });
}

/* Purpose: Show how total parameters are distributed across embedding, RNN, and classifier. */
function renderParameterCharts() {
  const values = dashboard.parameter_breakdown || {};
  const labels = Object.keys(values);
  const data = labels.map(label => values[label]);
  drawChart('parameter', 'parameterChart', { type: 'doughnut', data: { labels, datasets: [{ data }] } });
  drawChart('parameterDetail', 'parameterDetailChart', { type: 'bar', data: { labels, datasets: [{ label: 'Parameters', data }] }, options: { indexAxis: 'y' } });
  document.getElementById('parameterTable').innerHTML = `<table><thead><tr><th>Component</th><th>Parameters</th><th>Share</th></tr></thead><tbody>${labels.map(label => `<tr><td>${escapeHtml(label)}</td><td>${formatNumber(values[label])}</td><td>${dashboard.total_parameters ? formatPercent(values[label] / dashboard.total_parameters) : '—'}</td></tr>`).join('')}</tbody></table>`;
}

/* Purpose: Render corpus token-frequency and split-size visualizations. */
function renderDataCharts() {
  const topTokens = dashboard.top_tokens || [];
  drawChart('tokens', 'tokenChart', { type: 'bar', data: { labels: topTokens.slice(0, 10).map(x => x.token), datasets: [{ label: 'Count', data: topTokens.slice(0, 10).map(x => x.count) }] }, options: { indexAxis: 'y' } });
  drawChart('topTokens', 'topTokensChart', { type: 'bar', data: { labels: topTokens.map(x => x.token), datasets: [{ label: 'Frequency', data: topTokens.map(x => x.count) }] } });

  const splits = dashboard.dataset || {};
  const splitNames = Object.keys(splits);
  drawChart('splits', 'splitChart', { type: 'doughnut', data: { labels: splitNames.map(s => s[0].toUpperCase() + s.slice(1)), datasets: [{ data: splitNames.map(s => splits[s].tokens) }] } });

  document.getElementById('datasetTable').innerHTML = splitNames.length ? `<table><thead><tr><th>Split</th><th>Tokens</th><th>Unique</th><th>File MB</th><th>UNK rate</th></tr></thead><tbody>${splitNames.map(split => `<tr><td>${escapeHtml(split)}</td><td>${formatNumber(splits[split].tokens)}</td><td>${formatNumber(splits[split].unique_tokens)}</td><td>${splits[split].size_mb}</td><td>${splits[split].unk_rate !== undefined ? formatPercent(splits[split].unk_rate) : '—'}</td></tr>`).join('')}</tbody></table>` : '<div class="empty-state">Chưa load được dataset statistics.</div>';

  document.getElementById('dataInsights').innerHTML = `
    <div class="insight"><h4>Vocabulary coverage</h4><p>Vocabulary đang giới hạn ở ${formatNumber(dashboard.vocab_size)} token. Token ngoài vocabulary được ánh xạ sang <code>&lt;unk&gt;</code>.</p></div>
    <div class="insight"><h4>Context window</h4><p>Model chỉ sử dụng tối đa ${formatNumber(dashboard.sequence_length)} token gần nhất của request trước khi chạy RNN.</p></div>
    <div class="insight"><h4>Output-space cost</h4><p>Mỗi timestep tạo ${formatNumber(dashboard.vocab_size)} logits. Vì vậy vocabulary size tác động mạnh đến classifier size.</p></div>`;
}

/* Purpose: Render the model architecture and its main hyperparameters. */
function renderArchitecture() {
  const nodes = [
    ['Token IDs', `(batch, time) → vocabulary indices`],
    ['Embedding', `${formatNumber(dashboard.vocab_size)} → ${formatNumber(dashboard.embedding_dim)}`],
    ['RNN', `${formatNumber(dashboard.embedding_dim)} → hidden ${formatNumber(dashboard.hidden_size)}, ${formatNumber(dashboard.num_layers)} layer(s)`],
    ['Dropout', 'p=0.20 during training; disabled during inference'],
    ['Linear', `${formatNumber(dashboard.hidden_size)} → ${formatNumber(dashboard.vocab_size)} logits`],
    ['Softmax', 'Inference stage → probability distribution'],
  ];
  document.getElementById('architectureView').innerHTML = nodes.map((node, i) => `<div class="arch-node"><strong>${escapeHtml(node[0])}</strong><small>${escapeHtml(node[1])}</small></div>${i < nodes.length - 1 ? '<div style="text-align:center;color:#65728f">↓</div>' : ''}`).join('');
  document.getElementById('hyperCards').innerHTML = [
    ['Embedding dim', dashboard.embedding_dim, 'Vector size per token'],
    ['Hidden size', dashboard.hidden_size, 'Memory/state dimension'],
    ['RNN layers', dashboard.num_layers, 'Stacked recurrent layers'],
    ['Vocabulary', dashboard.vocab_size, 'Output classes'],
    ['Sequence length', dashboard.sequence_length, 'BPTT context length'],
    ['Device', dashboard.device || 'cpu', 'Inference device'],
  ].map(([l, v, s]) => `<div class="metric-card"><div class="label">${l}</div><div class="value">${escapeHtml(v)}</div><div class="sub">${s}</div></div>`).join('');
}

/* Purpose: Render exported final test metrics and checkpoint information. */
function renderTrainingSection() {
  const test = dashboard.optional_metrics?.test || {};
  const cards = [
    ['Best epoch', dashboard.checkpoint_epoch ?? '—', 'Checkpoint epoch'],
    ['Best validation loss', dashboard.best_validation_loss != null ? Number(dashboard.best_validation_loss).toFixed(4) : '—', 'Selected by validation loss'],
    ['Test perplexity', test.perplexity != null ? Number(test.perplexity).toFixed(2) : '—', 'exp(test loss)'],
    ['Test Top-5', test.top5_accuracy != null ? formatPercent(test.top5_accuracy) : '—', 'Target token in top five'],
  ];
  document.getElementById('trainingCards').innerHTML = cards.map(([l, v, s]) => `<div class="metric-card"><div class="label">${l}</div><div class="value">${escapeHtml(v)}</div><div class="sub">${s}</div></div>`).join('');
  document.getElementById('testMetricTable').innerHTML = `<table><thead><tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr></thead><tbody>${[
    ['Loss', test.loss, 'Average cross-entropy on test tokens'],
    ['Perplexity', test.perplexity, 'exp(loss); lower is better under the same evaluation setup'],
    ['Top-1 accuracy', test.top1_accuracy, 'Correct next token ranked first'],
    ['Top-5 accuracy', test.top5_accuracy, 'Correct next token appears in the five highest-scoring candidates'],
  ].map(([k, v, i]) => `<tr><td>${k}</td><td>${v === undefined ? '—' : String(k).includes('accuracy') ? formatPercent(v) : Number(v).toFixed(4)}</td><td>${i}</td></tr>`).join('')}</tbody></table>`;
}

/* Purpose: Update slider labels live during the presentation. */
function bindSlider(id, outputId, digits = 0) {
  const input = document.getElementById(id);
  const output = document.getElementById(outputId);
  if (!input || !output) return;
  input.addEventListener('input', () => output.textContent = Number(input.value).toFixed(digits));
}

/* Purpose: Send seed text to the backend and render top-k next-word probabilities and uncertainty statistics. */
async function runPrediction() {
  const button = document.getElementById('predictBtn');
  if (!button) return;
  button.disabled = true;
  button.textContent = 'Predicting...';
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: document.getElementById('seedInput').value, top_k: Number(document.getElementById('topK').value), temperature: Number(document.getElementById('temperature').value) }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed');
    const labels = data.candidates.map(x => x.word);
    const values = data.candidates.map(x => x.probability);
    drawChart('prediction', 'predictionChart', { type: 'bar', data: { labels, datasets: [{ label: 'Probability', data: values }] } });
    document.getElementById('predictLatency').textContent = `${data.inference_ms.toFixed(1)} ms`;
    document.getElementById('predictionTable').innerHTML = `<table><thead><tr><th>#</th><th>Candidate</th><th>Probability</th><th>Log P</th><th>Cumulative</th></tr></thead><tbody>${data.candidates.map(x => `<tr><td>${x.rank}</td><td><strong>${escapeHtml(x.word)}</strong></td><td><div class="prob-bar"><span style="width:${Math.min(x.probability * 100 * 3, 100)}%"></span></div>${x.percent.toFixed(2)}%</td><td>${x.log_probability.toFixed(3)}</td><td>${x.cumulative_probability.toFixed(3)}</td></tr>`).join('')}</tbody></table>`;
    document.getElementById('predictionStats').innerHTML = [
      ['Entropy', data.entropy.toFixed(3), 'Uncertainty of vocabulary distribution'],
      ['Top-1 probability', formatPercent(data.top1_probability), 'Probability assigned to the leading candidate'],
      ['Top-5 cumulative', formatPercent(data.top5_cumulative_probability), 'Probability mass in five candidates'],
      ['Top-1 vs Top-2 margin', data.top1_top2_margin.toFixed(4), 'Separation between top candidates'],
      ['Hidden-state norm', data.hidden_state_norm.toFixed(3), 'Magnitude of final recurrent state'],
      ['Context tokens', `${data.context_length} / ${dashboard.sequence_length}`, 'Actual context used'],
      ['Unknown context', data.unknown_context_tokens, 'Tokens mapped to <unk>'],
      ['Temperature', data.temperature.toFixed(2), 'Distribution control at inference'],
    ].map(([l, v, s]) => `<div class="stat-item"><div class="label">${l}</div><div class="value">${escapeHtml(v)}</div><div class="sub">${s}</div></div>`).join('');
    document.getElementById('contextTokens').innerHTML = data.context_tokens.map(t => `<span class="token">${escapeHtml(t)}</span>`).join('');
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Predict next word';
  }
}

/* Purpose: Call the generation endpoint and visualize the probability and hidden-state trajectory across steps. */
async function runGeneration() {
  const button = document.getElementById('generateBtn');
  if (!button) return;
  button.disabled = true;
  button.textContent = 'Generating...';
  try {
    const response = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed_text: document.getElementById('generationSeed').value, num_words: Number(document.getElementById('genWords').value), temperature: Number(document.getElementById('genTemperature').value), top_k: Number(document.getElementById('genTopK').value), mode: document.getElementById('generationMode').value }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Generation failed');
    document.getElementById('generatedText').textContent = data.text;
    document.getElementById('generationSpeed').textContent = `${data.words_per_second.toFixed(1)} words/s`;
    drawChart('generation', 'generationChart', { type: 'line', data: { labels: data.steps.map(x => x.step), datasets: [{ label: 'Next-word probability', data: data.steps.map(x => x.probability) }, { label: 'Hidden-state norm', data: data.steps.map(x => x.hidden_state_norm) }] } });
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Generate continuation';
  }
}

/* Purpose: Safely run one renderer so a missing optional dataset/metric cannot disable all remaining buttons. */
function safeRender(label, fn) {
  try {
    fn();
  } catch (error) {
    console.error(`${label} failed:`, error);
    showToast(`${label}: ${error.message}`, 'error');
  }
}

/* Purpose: Initialize every dashboard renderer and bind the interactive controls. */
function init() {
  setupNavigation();
  safeRender('Overview', renderOverviewCards);
  safeRender('Training charts', renderTrainingCharts);
  safeRender('Parameter charts', renderParameterCharts);
  safeRender('Data charts', renderDataCharts);
  safeRender('Architecture', renderArchitecture);
  safeRender('Training metrics', renderTrainingSection);

  bindSlider('topK', 'topKValue', 0);
  bindSlider('temperature', 'temperatureValue', 2);
  bindSlider('genWords', 'genWordsValue', 0);
  bindSlider('genTemperature', 'genTemperatureValue', 2);
  bindSlider('genTopK', 'genTopKValue', 0);

  document.getElementById('predictBtn')?.addEventListener('click', runPrediction);
  document.getElementById('generateBtn')?.addEventListener('click', runGeneration);

  if (window.RNN_STARTUP_ERROR) {
    showToast(`Model startup error: ${window.RNN_STARTUP_ERROR}`, 'error');
  }
}

document.addEventListener('DOMContentLoaded', init);
