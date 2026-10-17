/* Shared machinery of the dashboards (consultation and forecast pages): the
   selection state kept in the URL with back / forward, the Qlik-style list boxes
   and selection bar, the tooltip of the HTML marks, and chart helpers.

   createBoard({single, multi, oneAtATime, extras, labels, boxes, codeNames,
                dataUrl, downloadUrl, onData}) loads the data of the current
   selection from dataUrl and hands it to onData; the page draws its own views. */
(function () {
  const TRAFFIC = {verde: 'affidabile', giallo: 'indicativa', grigio: 'non affidabile'};
  const SEQ = [0, 5, 10, 25, 50, 75];  // class lower bounds of the heat colours, as the ECDC maps
  const ICONS = {
    search: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>',
    clear: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>',
  };

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const fmt = (v, d = 0) => v === null || v === undefined ? '–'
    : Number(v).toLocaleString('it-IT', {minimumFractionDigits: d, maximumFractionDigits: d});

  window.createBoard = function (config) {
    const {single, multi, labels, boxes, dataUrl, downloadUrl, onData} = config;
    const oneAtATime = config.oneAtATime || [];
    const extras = config.extras || {};
    const codeNames = config.codeNames || [];
    const root = document.querySelector('.viz-root');
    const css = getComputedStyle(root);
    const token = name => css.getPropertyValue(name).trim();
    const board = document.getElementById('board');
    const tooltip = document.getElementById('vizTooltip');

    // --- state: selections in the URL, with a history like Qlik's back / forward
    function fromUrl() {
      const q = new URLSearchParams(location.search);
      const s = {};
      Object.entries(extras).forEach(([k, fallback]) => { s[k] = q.get(k) || fallback; });
      single.forEach(k => { s[k] = q.get(k) || ''; });
      multi.forEach(k => { s[k] = q.getAll(k); });
      return s;
    }
    function toQuery(s) {
      const q = new URLSearchParams();
      single.forEach(k => { if (s[k]) q.append(k, s[k]); });
      multi.forEach(k => s[k].forEach(v => q.append(k, v)));
      Object.entries(extras).forEach(([k, fallback]) => { if (s[k] && s[k] !== fallback) q.append(k, s[k]); });
      return q;
    }
    const copy = s => JSON.parse(JSON.stringify(s));
    const api = {state: fromUrl(), answer: null};
    let past = [], future = [], request = 0;
    const openSearch = new Set();

    function set(changes) {
      past.push(copy(api.state)); future = [];
      Object.assign(api.state, changes);
      load();
    }
    const clearAll = () => Object.fromEntries(multi.map(k => [k, []]));
    document.getElementById('backButton').addEventListener('click', () => { future.push(copy(api.state)); api.state = past.pop(); load(); });
    document.getElementById('forwardButton').addEventListener('click', () => { past.push(copy(api.state)); api.state = future.pop(); load(); });
    document.getElementById('clearButton').addEventListener('click', () => set(clearAll()));
    document.querySelectorAll('[data-key][data-value]').forEach(b =>
      b.addEventListener('click', () => set({[b.dataset.key]: b.dataset.value})));

    // --- tooltip shared by the HTML marks
    function showTip(event, title, lines) {
      tooltip.replaceChildren(el('strong', '', title), ...lines.map(l => el('div', '', l)));
      tooltip.hidden = false;
      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX || rect.right, y = event.clientY || rect.top;
      tooltip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - tooltip.offsetWidth - 8)) + 'px';
      tooltip.style.top = Math.max(8, y - tooltip.offsetHeight - 10) + 'px';
    }
    const hideTip = () => { tooltip.hidden = true; };
    function hover(node, title, lines) {
      node.addEventListener('pointermove', e => showTip(e, title, lines));
      node.addEventListener('pointerleave', hideTip);
      node.addEventListener('focus', e => showTip(e, title, lines));
      node.addEventListener('blur', hideTip);
    }

    // --- list boxes
    function pick(dim, item) {
      const s = api.state;
      if (single.includes(dim)) {
        if (s[dim] === item.valore) return;
        const changes = {[dim]: item.valore};
        if (item.stato === 'escluso') Object.assign(changes, clearAll());  // as in Qlik: clears what excluded it
        set(changes);
      } else if (oneAtATime.includes(dim)) {
        set({[dim]: s[dim].includes(item.valore) ? [] : [item.valore]});
      } else if (item.stato === 'escluso') {
        set({...clearAll(), [dim]: [item.valore]});
      } else {
        set({[dim]: s[dim].includes(item.valore) ? s[dim].filter(v => v !== item.valore) : [...s[dim], item.valore]});
      }
    }
    function iconButton(icon, label) {
      const b = el('button'); b.type = 'button'; b.setAttribute('aria-label', label); b.title = label;
      b.innerHTML = ICONS[icon];
      return b;
    }
    function renderBoxes() {
      const host = document.getElementById('listBoxes');
      const scroll = Object.fromEntries([...host.querySelectorAll('.lb-list')].map(l => [l.dataset.dim, l.scrollTop]));
      const typed = Object.fromEntries([...host.querySelectorAll('.lb-search')].map(i => [i.dataset.dim, i.value]));
      host.replaceChildren(...boxes.filter(dim => api.answer.liste[dim]).map(dim => {
        const items = api.answer.liste[dim];
        const box = el('section', 'listbox');
        const head = el('div', 'lb-head');
        const title = el('span', '', labels[dim]);
        if (single.includes(dim)) title.append(el('span', 'lb-kind', 'uno solo'));
        if (oneAtATime.includes(dim)) title.append(el('span', 'lb-kind', 'uno alla volta'));
        const actions = el('span', 'd-flex');
        const search = iconButton('search', `Cerca in ${labels[dim]}`);
        actions.append(search);
        if (multi.includes(dim) && api.state[dim].length) {
          const clear = iconButton('clear', `Togli la selezione di ${labels[dim]}`);
          clear.addEventListener('click', () => set({[dim]: []}));
          actions.append(clear);
        }
        head.append(title, actions);
        const input = el('input', 'lb-search'); input.type = 'search'; input.placeholder = 'Cerca…'; input.dataset.dim = dim;
        input.value = typed[dim] || ''; input.hidden = !openSearch.has(dim);
        const list = el('ul', 'lb-list'); list.dataset.dim = dim; list.setAttribute('role', 'listbox');
        list.setAttribute('aria-label', labels[dim]);
        if (multi.includes(dim) && !oneAtATime.includes(dim)) list.setAttribute('aria-multiselectable', 'true');
        items.forEach(item => {
          const li = el('li', `lb-item ${item.stato}`); li.tabIndex = 0; li.setAttribute('role', 'option');
          li.setAttribute('aria-selected', item.stato === 'selezionato');
          li.dataset.search = `${item.valore} ${item.nome}`.toLowerCase();
          const name = codeNames.includes(dim) ? `${item.valore} · ${item.nome}` : item.nome;
          const label = el('span', 'lb-name', name); label.title = item.nota ? `${name} · ${item.nota}` : name;
          li.append(label);
          if (item.conteggio !== null && item.conteggio !== undefined) li.append(el('span', 'lb-count', fmt(item.conteggio)));
          li.addEventListener('click', () => pick(dim, item));
          li.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(dim, item); } });
          list.append(li);
        });
        const filterList = () => {
          const text = input.value.trim().toLowerCase();
          list.querySelectorAll('.lb-item').forEach(li => { li.hidden = text && !li.dataset.search.includes(text); });
        };
        input.addEventListener('input', filterList);
        input.addEventListener('keydown', e => {  // Enter selects the values found, as in Qlik
          if (e.key !== 'Enter') return;
          e.preventDefault();
          const found = items.filter((item, i) => !list.children[i].hidden && item.stato !== 'escluso');
          if (!found.length) return;
          if (single.includes(dim) || oneAtATime.includes(dim)) pick(dim, found[0]);
          else set({[dim]: [...new Set([...api.state[dim], ...found.map(f => f.valore)])]});
        });
        search.addEventListener('click', () => {
          if (openSearch.has(dim)) { openSearch.delete(dim); input.value = ''; input.hidden = true; filterList(); }
          else { openSearch.add(dim); input.hidden = false; input.focus(); }
        });
        box.append(head, input, list);
        filterList();
        requestAnimationFrame(() => { list.scrollTop = scroll[dim] || 0; });
        return box;
      }));
    }
    function renderSelections() {
      const s = api.state, a = api.answer;
      const chips = [...single, ...multi].filter(k => a.liste[k] && (single.includes(k) ? a.selezione[k] : s[k].length)).map(k => {
        const values = single.includes(k) ? [a.selezione[k]] : s[k];
        const names = values.map(v => (a.liste[k].find(i => i.valore === v) || {nome: v}).nome);
        const chip = el('span', 'chip');
        const body = el('span', 'chip-body');
        body.append(el('span', 'chip-field', labels[k]),
                    el('span', 'chip-values', names.length > 3 ? `${names.length} di ${a.liste[k].length}` : names.join(', ')));
        body.title = names.join(', ');
        chip.append(body);
        if (multi.includes(k)) {
          const x = el('button', '', '×'); x.type = 'button'; x.setAttribute('aria-label', `Togli ${labels[k]}`);
          x.addEventListener('click', () => set({[k]: []}));
          chip.append(x);
        }
        return chip;
      });
      document.getElementById('selections').replaceChildren(...chips);
      const count = multi.filter(k => s[k].length).length;
      const paneCount = document.getElementById('paneCount');
      if (paneCount) paneCount.textContent = count ? `(${count} attive)` : '';
      document.getElementById('backButton').disabled = !past.length;
      document.getElementById('forwardButton').disabled = !future.length;
      document.getElementById('clearButton').disabled = !count;
      document.querySelectorAll('[data-key][data-value]').forEach(b => b.classList.toggle('active', s[b.dataset.key] === b.dataset.value));
    }

    // --- data: the previous render stays, dimmed, until the new one arrives
    async function load() {
      const id = ++request;
      board.classList.add('loading');
      const query = toQuery(api.state);
      history.replaceState(null, '', query.toString() ? `?${query}` : location.pathname);
      const link = document.getElementById('downloadLink');
      if (link && downloadUrl) link.href = `${downloadUrl}?${query}`;
      try {
        const response = await (await fetch(`${dataUrl}?${query}`)).json();
        if (id !== request) return;
        if (response.vuoto) { location.reload(); return; }
        api.answer = response;
        single.forEach(k => { api.state[k] = response.selezione[k]; });
        Object.keys(extras).forEach(k => { if (response.selezione[k]) api.state[k] = response.selezione[k]; });
        const settled = toQuery(api.state);  // the values the server chose (defaults, forced horizon)
        history.replaceState(null, '', settled.toString() ? `?${settled}` : location.pathname);
        renderSelections(); renderBoxes(); onData(response, api);
      } finally {
        if (id === request) board.classList.remove('loading');
      }
    }

    // --- shared drawing helpers
    function trafficBadge(semaforo) {
      const b = el('span', `traffic ${semaforo}`); b.append(el('i'), document.createTextNode(TRAFFIC[semaforo])); return b;
    }
    const forecastLines = f => f.semaforo === 'grigio'
      ? ['Previsione non affidabile: nelle prove sul passato l\'intervallo non conteneva abbastanza spesso il valore reale']
      : [`${fmt(f.pct, 1)}% resistenti previsti (intervallo 80% ${fmt(f.basso, 1)}–${fmt(f.alto, 1)}%)`,
         `Previsione ${TRAFFIC[f.semaforo]}${f.errore != null ? `, errore tipico ${fmt(f.errore, 1)} punti` : ''}`];
    function cellColor(pct) {
      let i = SEQ.length - 1; while (i > 0 && pct < SEQ[i]) i--;
      return {bg: token(`--seq-${i + 1}`), fg: i >= 3 ? '#ffffff' : token('--ink-1')};
    }
    function heatLegend(node, lastLabel) {
      node.replaceChildren(...SEQ.map((low, i) => {
        const span = el('span'), sw = el('i'); sw.style.background = token(`--seq-${i + 1}`);
        span.append(sw, document.createTextNode(i === SEQ.length - 1 ? `≥${low}%` : `${low}–${SEQ[i + 1]}%`));
        return span;
      }));
      const thin = el('span'), sw = el('i');
      sw.style.background = getComputedStyle(document.documentElement).getPropertyValue('--bg-app');
      thin.append(sw, document.createTextNode(lastLabel));
      node.append(thin);
    }
    const crosshair = {
      id: 'crosshair',
      afterDatasetsDraw(chart) {
        const active = chart.tooltip && chart.tooltip.getActiveElements();
        if (!active || !active.length) return;
        const x = active[0].element.x, {top, bottom} = chart.chartArea, ctx = chart.ctx;
        ctx.save(); ctx.strokeStyle = token('--axis'); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); ctx.restore();
      },
    };
    const axisStyle = () => ({grid: {color: token('--grid'), drawTicks: false}, border: {color: token('--axis')},
                              ticks: {color: token('--ink-muted'), padding: 6, font: {size: 11}}});
    const tooltipStyle = () => ({mode: 'index', intersect: false, backgroundColor: '#fff', borderColor: token('--axis'), borderWidth: 1,
                                 titleColor: token('--ink-1'), bodyColor: token('--ink-2'), padding: 8, displayColors: false});
    const niceMax = v => [10, 20, 40, 60, 80, 100].find(m => m >= v) || 100;  // halves stay whole numbers

    // A row of the horizontal bar charts: bar with its interval, optional marker and reference line
    function barRow({code, name, pct, low, high, thin, label, lines, ref, marker, max, selected, onClick}) {
      const at = v => `${Math.min(100, 100 * v / max)}%`;
      const row = el('button', 'lab-row' + (selected ? ' selected' : '')); row.type = 'button';
      const track = el('span', 'lab-track');
      if (pct !== null && pct !== undefined) {
        const bar = el('span', 'lab-bar' + (thin ? ' thin' : '')); bar.style.width = at(pct);
        const ci = el('span', 'lab-ci'); ci.style.left = at(low); ci.style.width = `calc(${at(high)} - ${at(low)})`;
        track.append(bar, ci);
      }
      if (marker !== null && marker !== undefined) { const m = el('span', 'lab-marker'); m.style.left = at(marker); track.append(m); }
      if (ref !== null && ref !== undefined) { const r = el('span', 'lab-ref'); r.style.left = at(ref); track.append(r); }
      row.append(el('span', 'lab-name', `${code} · ${name}`), track, el('span', 'lab-value' + (thin ? ' thin' : ''), label));
      hover(row, `${code} · ${name}`, lines);
      row.addEventListener('click', onClick);
      return row;
    }
    function barAxis(node, max) {
      const ticks = el('span', 'ticks');
      [0, max / 2, max].forEach(v => { const t = el('span', '', `${fmt(v)}%`); t.style.left = `${100 * v / max}%`; ticks.append(t); });
      node.replaceChildren(el('span'), ticks, el('span'));
    }

    // Trend chart: observed %R per quarter with its 95% band, then the forecasts dashed with
    // their 80% band; rows {label, pct, basso, alto, riportabile, testati}, ahead {label, pct, basso, alto, semaforo}
    const charts = {};
    function drawTrend(canvasId, {rows, ahead, observedLines}) {
      const labels = [...rows.map(r => r.label), ...ahead.map(f => f.label).filter(l => !rows.some(r => r.label === l))];
      const at = Object.fromEntries(rows.map(r => [r.label, r]));
      const fAt = Object.fromEntries(ahead.map(f => [f.label, f]));
      const blue = token('--series-1'), muted = token('--muted-mark'), surface = token('--surface-1');
      const shown = f => f && f.semaforo !== 'grigio';
      const obs = l => at[l] && at[l].pct !== null && at[l].pct !== undefined ? at[l] : null;
      // the forecast line starts from the last observed quarter, so the two read as one
      const last = [...rows].reverse().find(r => obs(r.label));
      const fLine = labels.map(l => shown(fAt[l]) ? fAt[l].pct : (ahead.length && last && l === last.label ? last.pct : null));
      const tip = {...tooltipStyle(),
        filter: item => item.datasetIndex === 2 || (item.datasetIndex === 5 && fAt[item.label] && !obs(item.label)),
        callbacks: {title: items => items[0].label,
                    label: item => item.datasetIndex === 2 ? observedLines(at[item.label]) : forecastLines(fAt[item.label])}};
      const data = {labels, datasets: [
        {data: labels.map(l => obs(l) ? obs(l).alto : null), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false},
        {data: labels.map(l => obs(l) ? obs(l).basso : null), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: '-1', backgroundColor: token('--series-1-wash')},
        {data: labels.map(l => obs(l) ? obs(l).pct : null), borderColor: blue, borderWidth: 2, pointRadius: 4, pointHoverRadius: 5, pointBorderWidth: 2,
         pointBackgroundColor: labels.map(l => at[l] && at[l].riportabile ? blue : surface),
         pointBorderColor: labels.map(l => at[l] && at[l].riportabile ? blue : muted), spanGaps: false, tension: 0},
        {data: labels.map(l => shown(fAt[l]) ? fAt[l].alto : null), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false},
        {data: labels.map(l => shown(fAt[l]) ? fAt[l].basso : null), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: '-1', backgroundColor: token('--forecast-wash')},
        {data: fLine, borderColor: blue, borderWidth: 2, borderDash: [6, 4], pointRadius: labels.map(l => shown(fAt[l]) ? 4 : 0),
         pointHoverRadius: labels.map(l => fAt[l] ? 5 : 0), pointBackgroundColor: surface, pointBorderColor: blue, pointBorderWidth: 2, spanGaps: false, tension: 0},
      ]};
      const chart = charts[canvasId];
      if (chart) { chart.data = data; chart.options.plugins.tooltip = tip; chart.update(); return labels; }
      charts[canvasId] = new Chart(document.getElementById(canvasId), {
        type: 'line', data, plugins: [crosshair],
        options: {maintainAspectRatio: false, animation: false, interaction: {mode: 'index', intersect: false},
                  plugins: {legend: {display: false}, tooltip: tip},
                  scales: {x: axisStyle(), y: {...axisStyle(), beginAtZero: true, suggestedMax: 10,
                           ticks: {...axisStyle().ticks, callback: v => `${v}%`}}}},
      });
      return labels;
    }
    // Isolates tested per quarter, aligned under the trend (same labels)
    function drawTested(canvasId, labels, rows) {
      const at = Object.fromEntries(rows.map(r => [r.label, r]));
      const blue = token('--series-1'), muted = token('--muted-mark');
      const data = {labels, datasets: [{data: labels.map(l => at[l] ? at[l].testati : null), maxBarThickness: 24,
                    borderRadius: {topLeft: 4, topRight: 4}, borderSkipped: 'bottom',
                    backgroundColor: labels.map(l => at[l] && at[l].riportabile ? blue : muted)}]};
      const tip = {...tooltipStyle(), filter: item => item.raw !== null, callbacks: {title: items => items[0].label,
                   label: item => `${fmt(item.raw)} testati${at[item.label].riportabile ? '' : ' (non riportabile)'}`}};
      const chart = charts[canvasId];
      if (chart) { chart.data = data; chart.options.plugins.tooltip = tip; chart.update(); return; }
      charts[canvasId] = new Chart(document.getElementById(canvasId), {
        type: 'bar', data,
        options: {maintainAspectRatio: false, animation: false, interaction: {mode: 'index', intersect: false},
                  plugins: {legend: {display: false}, tooltip: tip},
                  scales: {x: {...axisStyle(), ticks: {display: false}, grid: {display: false}},
                           y: {...axisStyle(), beginAtZero: true, ticks: {...axisStyle().ticks, maxTicksLimit: 3,
                               callback: v => v >= 1000 ? `${fmt(v / 1000)}K` : v}}}},
      });
    }
    // Antibiogram heat table: rows antibiotics, columns pathogens; cell {pct, ...} or none
    function drawHeat(table, {columns, rows, cell, good, text, lines, selected, onPick}) {
      const head = el('thead'), hr = el('tr');
      if (columns.length) hr.append(el('th'));
      columns.forEach(c => { const th = el('th', '', c.nome); th.scope = 'col'; hr.append(th); });
      head.append(hr);
      const body = el('tbody');
      rows.forEach(a => {
        const tr = el('tr'), th = el('th', '', `${a.valore} · ${a.nome}`); th.scope = 'row'; tr.append(th);
        columns.forEach(col => {
          const td = el('td'), c = cell(a.valore, col.valore);
          if (!c) { td.className = 'empty'; tr.append(td); return; }
          td.textContent = text(c); td.tabIndex = 0;
          if (good(c)) { const color = cellColor(c.pct); td.style.background = color.bg; td.style.color = color.fg; }
          if (selected(a.valore, col.valore)) td.classList.add('selected');
          hover(td, `${col.nome} e ${a.nome}`, lines(c));
          const choose = () => onPick(a.valore, col.valore);
          td.addEventListener('click', choose);
          td.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
          tr.append(td);
        });
        body.append(tr);
      });
      table.replaceChildren(head, body);
    }
    function tile(label, value, note, hero, extra) {
      const t = el('div', 'kpi' + (hero ? ' kpi-hero' : ''));
      t.append(el('div', 'kpi-label', label), el('div', 'kpi-value', value));
      const n = el('div', 'kpi-note', note);
      if (extra) { n.append(' '); n.append(extra); }
      t.append(n);
      return t;
    }
    const forecastText = c => c.semaforo === 'grigio' ? 'n.a.' : `${c.semaforo === 'giallo' ? '≈' : ''}${fmt(c.pct, 0)}%`;

    if (window.matchMedia('(max-width: 991.98px)').matches) {
      const pane = document.getElementById('paneToggle');
      if (pane) pane.open = false;
    }
    Object.assign(api, {set, load, el, fmt, token, hover, trafficBadge, forecastLines, forecastText, cellColor, heatLegend,
                        niceMax, barRow, barAxis, drawTrend, drawTested, drawHeat, tile, TRAFFIC});
    load();
    return api;
  };
})();
