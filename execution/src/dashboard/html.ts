export const DASHBOARD_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polybot Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px; overflow: hidden; height: 100vh; display: flex; flex-direction: column;
  }

  /* ── Header ── */
  .header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; background: #161b22; border-bottom: 1px solid #30363d;
    flex-shrink: 0;
  }
  .header .slug { font-size: 16px; font-weight: bold; color: #58a6ff; }
  .header .timer { font-size: 20px; font-weight: bold; color: #f0883e; }
  .header .mode { background: #1f6feb; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .header .status { display: flex; align-items: center; gap: 8px; }
  .header .dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; }
  .header .dot.dead { background: #f85149; }

  /* ── Kill switch ── */
  .kill-btn {
    background: #da3633; color: #fff; border: 2px solid #f85149; border-radius: 6px;
    padding: 6px 18px; font-family: inherit; font-size: 14px; font-weight: bold;
    cursor: pointer; text-transform: uppercase; letter-spacing: 1px;
    transition: background 0.15s, transform 0.1s;
  }
  .kill-btn:hover { background: #f85149; transform: scale(1.05); }
  .kill-btn:active { transform: scale(0.95); }
  .kill-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .kill-btn.killed { background: #484f58; border-color: #484f58; }
  .kill-banner {
    display: none; background: #da3633; color: #fff; text-align: center;
    padding: 6px; font-weight: bold; font-size: 14px; letter-spacing: 1px;
  }
  .kill-banner.visible { display: block; }

  /* ── Two-column split ── */
  .columns { display: flex; flex: 1; min-height: 0; }
  .col-left { display: flex; flex-direction: column; min-width: 200px; overflow: hidden; }
  .col-right { display: flex; flex-direction: column; min-width: 200px; overflow: hidden; }

  /* ── Drag handles ── */
  .drag-col {
    width: 6px; cursor: col-resize; background: #21262d; flex-shrink: 0;
    transition: background 0.15s;
  }
  .drag-col:hover, .drag-col.active { background: #58a6ff; }
  .drag-row {
    height: 6px; cursor: row-resize; background: #21262d; flex-shrink: 0;
    transition: background 0.15s;
  }
  .drag-row:hover, .drag-row.active { background: #58a6ff; }

  /* ── Panels ── */
  .panel {
    background: #161b22; border: 1px solid #30363d;
    padding: 10px; overflow: auto; min-height: 60px;
  }
  .panel h3 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
    color: #8b949e; margin-bottom: 8px; border-bottom: 1px solid #21262d; padding-bottom: 4px;
  }

  /* ── Chart panels ── */
  .chart-cell {
    position: relative; flex: 1; min-height: 60px; overflow: hidden;
  }
  .chart-cell canvas { display: block; }
  .chart-cell .price-overlay {
    position: absolute; top: 8px; right: 16px;
    pointer-events: none; text-align: right;
    font-size: 22px; font-weight: bold; opacity: 0.8;
  }
  .chart-cell .price-overlay.btc { color: #58a6ff; }
  .chart-cell .price-overlay.up { color: #3fb950; }
  .chart-drag {
    height: 6px; cursor: row-resize; background: #21262d; flex-shrink: 0;
    transition: background 0.15s;
  }
  .chart-drag:hover, .chart-drag.active { background: #58a6ff; }

  /* ── Orderbook ── */
  .book-container { display: flex; gap: 12px; }
  .book-side { flex: 1; }
  .book-side h4 { text-align: center; font-size: 16px; margin-bottom: 6px; color: #8b949e; }
  .book-table { width: 100%; border-collapse: collapse; font-size: 16px; }
  .book-table td { padding: 3px 5px; position: relative; z-index: 1; }
  .book-table .bid-row td:last-child { color: #3fb950; text-align: right; }
  .book-table .ask-row td:last-child { color: #f85149; text-align: right; }
  .book-table .bid-row td:first-child,
  .book-table .ask-row td:first-child { text-align: right; }
  .book-table .price { text-align: center; font-weight: bold; }
  .bar {
    position: absolute; top: 0; bottom: 0; z-index: 0; opacity: 0.15; pointer-events: none;
  }
  .bar.bid { background: #3fb950; right: 0; }
  .bar.ask { background: #f85149; right: 0; }
  .spread-row { text-align: center; color: #8b949e; font-size: 14px; padding: 4px 0; }

  /* ── Tables ── */
  table.data { width: 100%; border-collapse: collapse; font-size: 16px; }
  table.data th {
    text-align: left; color: #8b949e; font-weight: normal;
    border-bottom: 1px solid #21262d; padding: 3px 6px; font-size: 15px;
  }
  table.data td { padding: 3px 6px; border-bottom: 1px solid #21262d10; }
  .buy { color: #3fb950; }
  .sell { color: #f85149; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .our-order { background: #1f6feb20; }

  /* ── Strategy metrics ── */
  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .metric {
    background: #21262d; border-radius: 4px; padding: 8px;
    display: flex; flex-direction: column; align-items: center;
  }
  .metric .label { font-size: 10px; color: #8b949e; text-transform: uppercase; }
  .metric .value { font-size: 18px; font-weight: bold; margin-top: 2px; }
  .metric .value.pos { color: #3fb950; }
  .metric .value.neg { color: #f85149; }

  /* no-select during drag */
  body.dragging { user-select: none; cursor: col-resize; }
  body.dragging-row { user-select: none; cursor: row-resize; }
</style>
</head>
<body>
<div class="header">
  <div>
    <span class="slug" id="h-slug">Waiting for market...</span>
    <span class="mode" id="h-mode">-</span>
  </div>
  <div class="status">
    <div class="dot" id="h-dot"></div>
    <span id="h-btc">-</span>
  </div>
  <button class="kill-btn" id="kill-btn" onclick="activateKill()">KILL</button>
  <div class="timer" id="h-timer">--:--</div>
</div>
<div class="kill-banner" id="kill-banner">KILLED — Strategy stopped, positions flattening</div>

<div class="columns" id="columns">
  <div class="col-left" id="col-left" style="flex: 0 0 60%">
    <div class="panel chart-cell" id="btc-chart-panel">
      <h3>BTC/USDT</h3>
      <div class="price-overlay btc" id="btc-overlay">-</div>
      <canvas id="btc-canvas"></canvas>
    </div>
    <div class="chart-drag" id="chart-drag"></div>
    <div class="panel chart-cell" id="up-chart-panel">
      <h3>Up Token Mid</h3>
      <div class="price-overlay up" id="up-overlay">-</div>
      <canvas id="up-canvas"></canvas>
    </div>
  </div>

  <div class="drag-col" id="drag-col"></div>

  <div class="col-right" id="col-right" style="flex: 1">
    <div class="panel" id="orders-panel" style="flex: 0 0 30%">
      <h3>Open Orders</h3>
      <table class="data" id="orders-table">
        <thead><tr><th>Token</th><th>Side</th><th class="num">Price</th><th class="num">Size</th><th class="num">Age</th></tr></thead>
        <tbody></tbody>
      </table>
      <h3 style="margin-top:10px">Positions</h3>
      <table class="data" id="positions-table">
        <thead><tr><th>Token</th><th>Side</th><th class="num">Size</th><th class="num">Entry</th><th class="num">uPnL</th><th class="num">rPnL</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="drag-row" id="drag-row-1"></div>

    <div class="panel" id="strategy-panel" style="flex: 0 0 30%">
      <h3>Strategy State</h3>
      <div class="metrics" id="strategy-metrics"></div>
    </div>

    <div class="drag-row" id="drag-row-2"></div>

    <div class="panel" id="book-panel" style="flex: 1">
      <h3>Orderbook</h3>
      <div class="book-container">
        <div class="book-side">
          <h4>UP</h4>
          <div id="book-up"></div>
        </div>
        <div class="book-side">
          <h4>DOWN</h4>
          <div id="book-down"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const sse = new EventSource('/api/sse');
let lastData = null;

sse.onmessage = (e) => {
  lastData = JSON.parse(e.data);
  render(lastData);
  // Sync kill state from server
  if (lastData.killed) {
    document.getElementById('kill-banner').classList.add('visible');
    const btn = document.getElementById('kill-btn');
    btn.disabled = true; btn.classList.add('killed'); btn.textContent = 'KILLED';
  }
};
sse.onerror = () => document.getElementById('h-dot').classList.add('dead');
sse.onopen = () => document.getElementById('h-dot').classList.remove('dead');

async function activateKill() {
  if (!confirm('KILL SWITCH: Cancel all orders and market-sell all positions?')) return;
  const btn = document.getElementById('kill-btn');
  btn.disabled = true; btn.textContent = 'KILLING...';
  try {
    const res = await fetch('/api/kill', { method: 'POST' });
    const data = await res.json();
    btn.classList.add('killed'); btn.textContent = 'KILLED';
    document.getElementById('kill-banner').classList.add('visible');
    console.log('Kill result:', data);
  } catch (err) {
    btn.disabled = false; btn.textContent = 'KILL';
    alert('Kill failed: ' + err.message);
  }
}

// ── Drag resize: column splitter ──
(function() {
  const drag = document.getElementById('drag-col');
  const colL = document.getElementById('col-left');
  const cols = document.getElementById('columns');
  let active = false;

  drag.addEventListener('mousedown', (e) => {
    active = true;
    drag.classList.add('active');
    document.body.classList.add('dragging');
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!active) return;
    const rect = cols.getBoundingClientRect();
    const pct = Math.max(20, Math.min(80, (e.clientX - rect.left) / rect.width * 100));
    colL.style.flex = '0 0 ' + pct + '%';
    scheduleChartRedraw();
  });
  document.addEventListener('mouseup', () => {
    if (active) { active = false; drag.classList.remove('active'); document.body.classList.remove('dragging'); scheduleChartRedraw(); }
  });
})();

// ── Drag resize: row splitters ──
function setupRowDrag(handleId, panelAboveId) {
  const handle = document.getElementById(handleId);
  const above = document.getElementById(panelAboveId);
  const colRight = document.getElementById('col-right');
  let active = false, startY = 0, startH = 0;

  handle.addEventListener('mousedown', (e) => {
    active = true; startY = e.clientY; startH = above.getBoundingClientRect().height;
    handle.classList.add('active'); document.body.classList.add('dragging-row'); e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!active) return;
    const colH = colRight.getBoundingClientRect().height;
    const newH = Math.max(40, Math.min(colH - 100, startH + (e.clientY - startY)));
    above.style.flex = '0 0 ' + (newH / colH * 100) + '%';
  });
  document.addEventListener('mouseup', () => {
    if (active) { active = false; handle.classList.remove('active'); document.body.classList.remove('dragging-row'); }
  });
}
setupRowDrag('drag-row-1', 'orders-panel');
setupRowDrag('drag-row-2', 'strategy-panel');

// ── Chart redraw scheduling ──
let chartRedrawTimer = null;
function scheduleChartRedraw() {
  if (chartRedrawTimer) return;
  chartRedrawTimer = requestAnimationFrame(() => {
    chartRedrawTimer = null;
    if (lastData) {
      drawSingleChart('btc-canvas', 'btc-chart-panel', lastData.btcHistory, '#58a6ff', 0);
      drawSingleChart('up-canvas', 'up-chart-panel', lastData.upMidHistory, '#3fb950', 2);
    }
  });
}
window.addEventListener('resize', scheduleChartRedraw);

// ── Drag resize: chart splitter ──
(function() {
  const handle = document.getElementById('chart-drag');
  const above = document.getElementById('btc-chart-panel');
  const col = document.getElementById('col-left');
  let active = false, startY = 0, startH = 0;

  handle.addEventListener('mousedown', (e) => {
    active = true; startY = e.clientY; startH = above.getBoundingClientRect().height;
    handle.classList.add('active'); document.body.classList.add('dragging-row'); e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!active) return;
    const colH = col.getBoundingClientRect().height;
    const newH = Math.max(60, Math.min(colH - 80, startH + (e.clientY - startY)));
    above.style.flex = '0 0 ' + (newH / colH * 100) + '%';
    scheduleChartRedraw();
  });
  document.addEventListener('mouseup', () => {
    if (active) { active = false; handle.classList.remove('active'); document.body.classList.remove('dragging-row'); scheduleChartRedraw(); }
  });
})();

// ── Render ──
function render(d) {
  if (d.market) {
    document.getElementById('h-slug').textContent = d.market.slug;
    const remaining = d.market.timeRemainingMs;
    const h = Math.floor(remaining / 3600000);
    const m = Math.floor((remaining % 3600000) / 60000);
    const s = Math.floor((remaining % 60000) / 1000);
    document.getElementById('h-timer').textContent =
      h > 0 ? h + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0')
             : m + ':' + String(s).padStart(2,'0');
  } else {
    document.getElementById('h-slug').textContent = 'No active market';
    document.getElementById('h-timer').textContent = '--:--';
  }
  if (d.strategy) document.getElementById('h-mode').textContent = d.strategy.mode || '-';

  if (d.btcPrice) {
    const fmt = d.btcPrice.price.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
    document.getElementById('h-btc').textContent = 'BTC $' + fmt;
    document.getElementById('btc-overlay').textContent = '$' + fmt;
  }
  if (d.bookUp && d.bookUp.mid > 0) {
    document.getElementById('up-overlay').textContent = d.bookUp.mid.toFixed(3);
  }

  drawSingleChart('btc-canvas', 'btc-chart-panel', d.btcHistory, '#58a6ff', 0);
  drawSingleChart('up-canvas', 'up-chart-panel', d.upMidHistory, '#3fb950', 2);
  renderBook('book-up', d.bookUp, d.openOrders, d.market?.upTokenId);
  renderBook('book-down', d.bookDown, d.openOrders, d.market?.downTokenId);
  renderOrders(d.openOrders, d.market);
  renderPositions(d.positions, d.bookUp, d.bookDown, d.market);
  renderStrategy(d.strategy);
}

// ── Single-series chart ──
function drawSingleChart(canvasId, panelId, history, color, decimals) {
  const canvas = document.getElementById(canvasId);
  const panel = document.getElementById(panelId);
  const rect = panel.getBoundingClientRect();

  const padL = 58, padR = 12, padT = 10, padB = 22;
  const totalW = Math.floor(rect.width - 22);
  const totalH = Math.floor(rect.height - 36);

  if (totalW < 50 || totalH < 30) return;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = totalW * dpr;
  canvas.height = totalH * dpr;
  canvas.style.width = totalW + 'px';
  canvas.style.height = totalH + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, totalW, totalH);

  const plotW = totalW - padL - padR;
  const plotH = totalH - padT - padB;
  if (plotW < 20 || plotH < 20) return;

  if (!history || history.length < 2) return;

  const prices = history.map(p => p.p);
  const pMin = Math.min(...prices);
  const pMax = Math.max(...prices);
  const pRange = pMax - pMin || (decimals === 0 ? 1 : 0.01);
  const tMin = history[0].t;
  const tMax = history[history.length - 1].t;
  const tRange = tMax - tMin || 1;

  // Grid lines
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
  }

  // Y axis labels
  ctx.fillStyle = color;
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * i / 4;
    ctx.fillText((pMax - pRange * i / 4).toFixed(decimals), padL - 6, y + 4);
  }

  // Price line
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  for (let i = 0; i < history.length; i++) {
    const x = padL + plotW * ((history[i].t - tMin) / tRange);
    const y = padT + plotH * (1 - (history[i].p - pMin) / pRange);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Gradient fill
  const rgb = color === '#58a6ff' ? '88,166,255' : '63,185,80';
  const gradient = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  gradient.addColorStop(0, 'rgba(' + rgb + ',0.10)');
  gradient.addColorStop(1, 'rgba(' + rgb + ',0)');
  ctx.lineTo(padL + plotW * ((history[history.length-1].t - tMin) / tRange), padT + plotH);
  ctx.lineTo(padL + plotW * ((history[0].t - tMin) / tRange), padT + plotH);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Time labels
  ctx.fillStyle = '#484f58';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const t = new Date(tMin + tRange * i / 4);
    const x = padL + plotW * i / 4;
    ctx.fillText(t.toLocaleTimeString(), x, padT + plotH + 16);
  }
}

// ── Orderbook ──
function renderBook(containerId, book, orders, tokenId) {
  const el = document.getElementById(containerId);
  if (!book || (!book.asks.length && !book.bids.length)) {
    el.innerHTML = '<div style="color:#484f58;text-align:center;padding:20px">No data</div>';
    return;
  }

  const ourPrices = new Set();
  if (orders && tokenId) {
    orders.filter(o => o.assetId === tokenId).forEach(o => ourPrices.add(o.price));
  }

  const maxSize = Math.max(...book.bids.map(l => l.size), ...book.asks.map(l => l.size), 1);
  let html = '<table class="book-table">';

  const asks = book.asks.slice(0, 8).reverse();
  for (const lvl of asks) {
    const pct = (lvl.size / maxSize * 100).toFixed(0);
    const ours = ourPrices.has(lvl.price) ? ' our-order' : '';
    html += '<tr class="ask-row' + ours + '"><td>' + lvl.size.toFixed(0) + '</td>';
    html += '<td class="price">' + lvl.price.toFixed(2) + '</td>';
    html += '<td style="position:relative"><div class="bar ask" style="width:' + pct + '%"></div>' + lvl.size.toFixed(0) + '</td></tr>';
  }

  html += '<tr><td colspan="3" class="spread-row">spread: ' + book.spread.toFixed(3) + ' | mid: ' + book.mid.toFixed(3) + '</td></tr>';

  const bids = book.bids.slice(0, 8);
  for (const lvl of bids) {
    const pct = (lvl.size / maxSize * 100).toFixed(0);
    const ours = ourPrices.has(lvl.price) ? ' our-order' : '';
    html += '<tr class="bid-row' + ours + '"><td>' + lvl.size.toFixed(0) + '</td>';
    html += '<td class="price">' + lvl.price.toFixed(2) + '</td>';
    html += '<td style="position:relative"><div class="bar bid" style="width:' + pct + '%"></div>' + lvl.size.toFixed(0) + '</td></tr>';
  }

  html += '</table>';
  el.innerHTML = html;
}

// ── Orders ──
function renderOrders(orders, market) {
  const tbody = document.querySelector('#orders-table tbody');
  if (!orders || orders.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="color:#484f58">No open orders</td></tr>';
    return;
  }
  const now = Date.now();
  tbody.innerHTML = orders.map(o => {
    const token = market ? (o.assetId === market.upTokenId ? 'UP' : 'DOWN') : o.assetId.slice(0,8);
    const age = ((now - o.placedAt) / 1000).toFixed(0) + 's';
    return '<tr><td>' + token + '</td><td class="' + o.side.toLowerCase() + '">' + o.side + '</td>' +
      '<td class="num">' + o.price.toFixed(2) + '</td>' +
      '<td class="num">' + o.remainingSize.toFixed(0) + '/' + o.originalSize.toFixed(0) + '</td>' +
      '<td class="num">' + age + '</td></tr>';
  }).join('');
}

// ── Positions ──
function renderPositions(positions, bookUp, bookDown, market) {
  const tbody = document.querySelector('#positions-table tbody');
  if (!positions || positions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:#484f58">No positions</td></tr>';
    return;
  }
  tbody.innerHTML = positions.map(p => {
    let token = p.assetId.slice(0, 8);
    let mid = 0;
    if (market) {
      if (p.assetId === market.upTokenId) { token = 'UP'; mid = bookUp?.mid || 0; }
      else { token = 'DOWN'; mid = bookDown?.mid || 0; }
    }
    const unrealized = p.side === 'BUY' ? (mid - p.avgEntryPrice) * p.size : (p.avgEntryPrice - mid) * p.size;
    const uClass = unrealized >= 0 ? 'pos' : 'neg';
    const rClass = p.realizedPnl >= 0 ? 'pos' : 'neg';
    return '<tr><td>' + token + '</td><td class="' + p.side.toLowerCase() + '">' + p.side + '</td>' +
      '<td class="num">' + p.size.toFixed(1) + '</td>' +
      '<td class="num">' + p.avgEntryPrice.toFixed(3) + '</td>' +
      '<td class="num ' + uClass + '">' + (unrealized >= 0 ? '+' : '') + unrealized.toFixed(2) + '</td>' +
      '<td class="num ' + rClass + '">' + (p.realizedPnl >= 0 ? '+' : '') + p.realizedPnl.toFixed(2) + '</td></tr>';
  }).join('');
}

// ── Strategy ──
function renderStrategy(s) {
  const el = document.getElementById('strategy-metrics');
  if (!s) { el.innerHTML = '<div style="color:#484f58">No strategy data</div>'; return; }
  const pnlClass = s.realizedPnl >= 0 ? 'pos' : 'neg';
  const pnlSign = s.realizedPnl >= 0 ? '+' : '';
  el.innerHTML =
    metric('Realized PnL', pnlSign + '$' + s.realizedPnl.toFixed(2), pnlClass) +
    metric('Pairs Minted', s.pairsMinted) +
    metric('Inv UP', s.inventoryUp) +
    metric('Inv DOWN', s.inventoryDown) +
    metric('Sell Fills UP', s.sellFillsUp + ' ($' + s.sellRevenueUp.toFixed(2) + ')') +
    metric('Sell Fills DN', s.sellFillsDown + ' ($' + s.sellRevenueDown.toFixed(2) + ')') +
    metric('Redeemable', s.redeemablePairs + ' pairs') +
    metric('Pending Mint', s.pendingMint ? 'YES' : 'no');
}

function metric(label, value, cls) {
  return '<div class="metric"><span class="label">' + label + '</span><span class="value' + (cls ? ' ' + cls : '') + '">' + value + '</span></div>';
}
</script>
</body>
</html>`;
