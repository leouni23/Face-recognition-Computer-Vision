/* Validation review + metrics. Loaded after the page's inline script, which defines the
   shared globals FPS, GAP, VERDICT_META, esc, avg, mode, vtime. The player requests GLOBAL
   frame indices; the server maps them to the right video segment, so playback is seamless
   across segments without any change here. */

let sessionId = null, gallery = [], threshold = null;
let detections = [], cameras = [], curCam = null, segments = [];
let curIndex = 0, maxIndex = 0, t0ms = null;
let playTimer = null, detChart = null, cmcChart = null;
let noVideo = false;   // session recorded no footage → hide the frame player, review via timeline only

// ── Sessions ───────────────────────────────────────────────────────────────
function refreshSessions() {
  fetch('/api/validation/sessions').then(r => r.json()).then(list => {
    const sel = document.getElementById('session-sel');
    const keep = sel.value;
    sel.innerHTML = '<option value="">— scegli una sessione —</option>' +
      list.map(s => `<option value="${esc(s.session_id)}">${esc(s.session_id)} · ${esc(s.session_type || '')} · soglia ${s.threshold}</option>`).join('');
    if (keep) { sel.value = keep; }
    else if (list.length) { sel.value = list[0].session_id; loadSession(); }
  });
}

function loadSession() {
  stopPlay();
  sessionId = document.getElementById('session-sel').value || null;
  detections = []; segments = []; curIndex = 0; maxIndex = 0; t0ms = null; gallery = []; threshold = null;
  if (detChart) { detChart.destroy(); detChart = null; }
  if (cmcChart) { cmcChart.destroy(); cmcChart = null; }
  document.getElementById('metrics-out').classList.add('hidden');
  document.getElementById('metrics-note').textContent = '';
  document.getElementById('metrics-btn').disabled = !sessionId;
  document.getElementById('frame-img').removeAttribute('src');
  document.getElementById('sess-meta').textContent = '';
  if (!sessionId) return;
  Promise.all([
    fetch(`/api/validation/${encodeURIComponent(sessionId)}/session`).then(r => r.json()).catch(() => ({})),
    fetch(`/api/validation/${encodeURIComponent(sessionId)}/detections`).then(r => r.json()),
  ]).then(([manifest, dets]) => {
    gallery = manifest.enrolled || [];
    threshold = manifest.threshold;
    noVideo = manifest.record_video === false;
    document.getElementById('player-block').classList.toggle('hidden', noVideo);
    document.getElementById('no-video-note').classList.toggle('hidden', !noVideo);
    document.getElementById('sess-meta').textContent =
      `${manifest.session_type || ''} · ${noVideo ? 'senza video' : 'video'} · ${gallery.length} iscritti · soglia ${threshold ?? '—'} · ${manifest.platform || ''}`;
    detections = dets;
    t0ms = dets.reduce((m, d) => Math.min(m, d.timestamp_ms ?? Infinity), Infinity);
    if (!isFinite(t0ms)) t0ms = null;
    cameras = [...new Set(dets.map(d => String(d.camera_id)))].sort();
    document.getElementById('cam-sel').innerHTML = cameras.map(c => `<option value="${esc(c)}">camera ${esc(c)}</option>`).join('');
    switchCamera();
  });
}

function switchCamera() {
  stopPlay();
  curCam = document.getElementById('cam-sel').value || cameras[0];
  const items = detections.map((d, i) => ({ d, i })).filter(({ d }) => String(d.camera_id) === String(curCam));
  maxIndex = items.reduce((m, { d }) => Math.max(m, d.frame_index), 0);
  document.getElementById('frame-slider').max = maxIndex;
  segments = buildSegments(items);
  setFrame(segments.length ? segments[0].first : 0);
  renderTimeline();
}

// ── Events (one continuous appearance per face slot) ───────────────────────────
function buildSegments(items) {
  const byFace = {};
  items.forEach(it => { (byFace[it.d.face_id] = byFace[it.d.face_id] || []).push(it); });
  const segs = [];
  Object.values(byFace).forEach(list => {
    list.sort((a, b) => a.d.frame_index - b.d.frame_index);
    let run = [list[0]];
    for (let k = 1; k < list.length; k++) {
      if (list[k].d.frame_index - run[run.length - 1].d.frame_index <= GAP) run.push(list[k]);
      else { segs.push(mkSeg(run)); run = [list[k]]; }
    }
    segs.push(mkSeg(run));
  });
  return segs.sort((a, b) => a.first - b.first || a.faceId - b.faceId);
}
function mkSeg(run) {
  const ds = run.map(r => r.d);
  return {
    idxs: run.map(r => r.i), faceId: ds[0].face_id,
    first: ds[0].frame_index, last: ds[ds.length - 1].frame_index, nFrames: ds.length,
    meanDist: avg(ds.map(d => d.raw_cosine_distance).filter(x => x != null)),
    pred: mode(ds.map(d => d.predicted_identity)),
    preset: mode(ds.map(d => d.preset_id).filter(Boolean)),
  };
}
function segTruth(seg) { for (const i of seg.idxs) if (detections[i].truth) return detections[i].truth; return null; }
function derivedVerdict(seg) {
  const t = segTruth(seg);
  if (!t) return null;
  const preds = seg.idxs.map(i => detections[i].predicted_person_id);
  if (t.non_mate) { const acc = preds.filter(p => p != null).length; return acc > preds.length - acc ? 'false_positive' : 'non_mate_correct'; }
  let correct = 0, swap = 0, miss = 0;
  preds.forEach(p => { if (p == null) miss++; else if (p === t.true_person_id) correct++; else swap++; });
  if (correct >= swap && correct >= miss) return 'mate_correct';
  return swap >= miss ? 'swap' : 'mate_miss';
}
function currentSegment() {
  return segments.find(s => curIndex >= s.first && curIndex <= s.last)
      || segments.filter(s => s.first <= curIndex).slice(-1)[0] || segments[0] || null;
}

// ── Frame player ─────────────────────────────────────────────────────────────
function setFrame(i) {
  if (!sessionId || curCam == null) return;
  curIndex = Math.max(0, Math.min(maxIndex, i));
  if (!noVideo) {
    document.getElementById('frame-img').src =
      `/api/validation/${encodeURIComponent(sessionId)}/frame/${encodeURIComponent(curCam)}/${curIndex}`;
  }
  document.getElementById('frame-slider').value = curIndex;
  document.getElementById('frame-info').textContent = `frame ${curIndex} / ${maxIndex} · ${vtime(curIndex)}`;
  renderEventBar();
  const seg = currentSegment();
  document.querySelectorAll('#rows tr').forEach(tr =>
    tr.classList.toggle('cur', seg && Number(tr.dataset.first) === seg.first && Number(tr.dataset.face) === seg.faceId));
}
function playPause() { playTimer ? stopPlay() : startPlay(); }
function startPlay() {
  if (playTimer) return;
  document.getElementById('play-btn').textContent = '⏸ Pausa';
  playTimer = setInterval(() => { if (curIndex >= maxIndex) { stopPlay(); return; } setFrame(curIndex + 1); }, 1000 / FPS);
}
function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; } document.getElementById('play-btn').textContent = '▶ Play'; }
function gotoEvent(dir) {
  stopPlay(); if (!segments.length) return;
  const cur = currentSegment(); let next;
  if (dir > 0) next = segments.find(s => s.first > (cur ? cur.first : curIndex));
  else { const prev = segments.filter(s => s.first < (cur ? cur.first : curIndex)); next = prev.slice(-1)[0]; }
  if (next) setFrame(next.first);
}

// ── current-event true-identity control (manual correction / common session) ───
function renderEventBar() {
  const bar = document.getElementById('event-label-bar');
  const seg = currentSegment();
  if (!seg) { bar.innerHTML = '<p class="text-xs text-gray-600">Nessun evento per questa camera.</p>'; return; }
  const v = derivedVerdict(seg), vm = v ? VERDICT_META[v] : null, t = segTruth(seg);
  const subjBtns = gallery.map((g, k) =>
    `<button class="truth-btn ${t && t.true_person_id === g.id ? 'ring-2 ring-emerald-400' : ''}"
       onclick="assignTruth('${g.id}')">${k < 9 ? `<span class="text-gray-500 mr-1">${k + 1}</span>` : ''}${esc(g.name)}</button>`).join('');
  bar.innerHTML = `
    <div class="text-[11px] text-gray-500">Evento · volto ${seg.faceId} · frame ${seg.first}–${seg.last} (${vtime(seg.first)}–${vtime(seg.last)}) · ${seg.nFrames} frame${seg.preset ? ' · cond. ' + esc(seg.preset) : ''}</div>
    <div class="text-xs text-gray-300">predetto <b>${esc(seg.pred)}</b> · dist media ${seg.meanDist == null ? '—' : seg.meanDist.toFixed(3)}
      ${vm ? `· <span style="color:${vm.color}">● ${vm.label}</span>` : '· <span class="text-gray-500">senza verità</span>'}</div>
    <div class="text-[11px] text-gray-500 pt-1">Correggi l'identità vera dell'evento:</div>
    <div class="flex flex-wrap gap-1.5">
      ${subjBtns || '<span class="text-[11px] text-amber-500">galleria non disponibile</span>'}
      <button class="truth-btn ${t && t.non_mate ? 'ring-2 ring-sky-400' : ''}" onclick="assignTruth(null)"><span class="text-gray-500 mr-1">0</span>Non-mate</button>
    </div>`;
}
function assignTruth(personId) {
  const seg = currentSegment(); if (!seg || !sessionId) return;
  const truth = personId == null ? { non_mate: true } : { true_person_id: Number(personId) };
  seg.idxs.forEach(i => detections[i].truth = truth);
  const labels = seg.idxs.map(i => { const d = detections[i]; return { camera_id: d.camera_id, frame_index: d.frame_index, face_id: d.face_id, ...truth }; });
  fetch(`/api/validation/${encodeURIComponent(sessionId)}/labels`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ labels }) }).catch(() => alert('Errore salvataggio.'));
  renderEventBar(); renderTimeline(); gotoEvent(1);
}

// ── Timeline ───────────────────────────────────────────────────────────────────
function renderTimeline() {
  const fst = document.getElementById('filter-status').value;
  const tbody = document.getElementById('rows');
  const shown = segments.filter(s => { const labelled = !!segTruth(s); return fst === 'all' || (fst === 'labelled' ? labelled : !labelled); });
  if (!segments.length) tbody.innerHTML = '<tr><td colspan="6" class="py-3 px-2 text-gray-600">Nessun evento.</td></tr>';
  else if (!shown.length) tbody.innerHTML = '<tr><td colspan="6" class="py-3 px-2 text-gray-600">Nessun evento per il filtro.</td></tr>';
  else tbody.innerHTML = shown.map(s => {
    const t = segTruth(s), v = derivedVerdict(s), vm = v ? VERDICT_META[v] : null, cur = currentSegment();
    const truthLabel = !t ? '<span class="text-gray-600">—</span>'
      : t.non_mate ? '<span class="text-sky-400">Non-mate</span>'
      : esc((gallery.find(g => g.id === t.true_person_id) || {}).name || ('#' + t.true_person_id));
    return `<tr data-first="${s.first}" data-face="${s.faceId}" onclick="setFrame(${s.first})"
        class="border-t border-gray-800/50 cursor-pointer ${cur && cur.first === s.first && cur.faceId === s.faceId ? 'cur' : ''}">
      <td class="py-1.5 px-2">${vtime(s.first)}<span class="text-gray-600">–${vtime(s.last)}</span><br><span class="text-[10px] text-gray-600">f ${s.first}–${s.last}</span></td>
      <td class="py-1.5 px-2">${esc(s.pred)}</td>
      <td class="py-1.5 px-2 text-right">${s.nFrames}</td>
      <td class="py-1.5 px-2 text-[11px] text-gray-400">${esc(s.preset || '—')}</td>
      <td class="py-1.5 px-2">${truthLabel}</td>
      <td class="py-1.5 px-2">${vm ? `<span class="font-medium" style="color:${vm.color}">● ${vm.label}</span>` : '<span class="text-gray-600">—</span>'}</td>
    </tr>`;
  }).join('');
  const labelled = segments.filter(s => segTruth(s)).length;
  document.getElementById('progress').textContent = `${labelled}/${segments.length} eventi con verità`;
}

document.addEventListener('keydown', e => {
  if (/input|select|textarea/i.test(e.target.tagName)) return;
  if (e.key === 'ArrowRight') { e.preventDefault(); stopPlay(); setFrame(curIndex + 1); return; }
  if (e.key === 'ArrowLeft')  { e.preventDefault(); stopPlay(); setFrame(curIndex - 1); return; }
  if (e.key === '0') { e.preventDefault(); assignTruth(null); return; }
  if (/[1-9]/.test(e.key)) { const g = gallery[Number(e.key) - 1]; if (g) { e.preventDefault(); assignTruth(g.id); } }
});

// ── Metrics ─────────────────────────────────────────────────────────────────
function computeMetrics() {
  if (!sessionId) return;
  fetch(`/api/validation/${encodeURIComponent(sessionId)}/metrics`, { method: 'POST' }).then(r => r.json()).then(renderMetrics).catch(() => alert('Errore calcolo metriche.'));
}
const pct = x => x == null ? '—' : (x * 100).toFixed(1) + '%';
const ciStr = b => (b.ci95 && b.ci95[0] != null) ? `IC95 ${pct(b.ci95[0])}–${pct(b.ci95[1])}` : '';
function rateCard(title, b, color) {
  const warn = b.low_error_warning ? ' <span title="meno di 30 errori: stima imprecisa">⚠</span>' : '';
  const r3 = b.rule_of_3_upper != null ? `<br><span class="text-[10px] text-gray-500">0 errori → ≤ ${pct(b.rule_of_3_upper)} (regola del 3)</span>` : '';
  return `<div class="bg-gray-950 rounded-xl px-3 py-2"><div class="text-[10px] text-gray-500 uppercase tracking-wide">${title}${warn}</div>
    <div class="font-semibold" style="color:${color}">${pct(b.value)}</div>
    <div class="text-[10px] text-gray-500">${b.errors}/${b.n} · ${ciStr(b)}${r3}</div></div>`;
}
function renderMetrics(m) {
  document.getElementById('metrics-out').classList.remove('hidden');
  const c = m.counts, faf = m.fnir_at_fpir || {};
  const plain = [
    ['Eventi mate', c.mate_events], ['Eventi non-mate', c.non_mate_events],
    ['EER', m.eer && m.eer.value != null ? `${pct(m.eer.value)} @ ${m.eer.threshold}` : '—'],
    ['Rank-1', pct(m.rank1)],
    ['FNIR @ FPIR 1%', faf['0.01'] ? pct(faf['0.01'].fnir) : '—'],
    ['FNIR @ FPIR 0.3%', faf['0.003'] ? pct(faf['0.003'].fnir) : '—'],
  ];
  document.getElementById('metrics-cards').innerHTML =
    rateCard('FPIR (falsi positivi)', m.fpir, '#ef4444') + rateCard('FNIR (mate mancati)', m.fnir, '#f59e0b') +
    plain.map(([k, v]) => `<div class="bg-gray-950 rounded-xl px-3 py-2"><div class="text-[10px] text-gray-500 uppercase tracking-wide">${k}</div><div class="text-gray-100 font-semibold">${v}</div></div>`).join('');
  document.getElementById('caveats').innerHTML = (m.caveats || []).map(t =>
    `<div class="text-[11px] text-amber-400/90 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-1.5">⚠ ${esc(t)}</div>`).join('');
  document.getElementById('metrics-note').textContent =
    `${c.labelled_detections}/${c.total_detections} detection con verità · ${c.total_events} eventi · FAR/FRR ${pct(m.aliases.far)}/${pct(m.aliases.frr)} · esportato metrics.json + det_curve.csv + cmc.csv.`;

  const thrBox = document.getElementById('threshold-apply');
  if (m.eer && m.eer.threshold != null) {
    thrBox.style.display = ''; document.getElementById('eer-thr').textContent = m.eer.threshold;
    document.getElementById('thr-input').value = m.eer.threshold; document.getElementById('thr-msg').textContent = '';
  } else { thrBox.style.display = 'none'; }

  // per-condition breakdown
  const bd = document.getElementById('breakdown');
  const cond = m.per_condition || {};
  if (Object.keys(cond).length) {
    bd.classList.remove('hidden');
    document.getElementById('breakdown-rows').innerHTML = Object.entries(cond).map(([cid, cm]) =>
      `<tr class="border-t border-gray-800/50"><td class="py-1 px-2">${esc(cid)}</td>
        <td class="py-1 px-2 text-right">${cm.counts.mate_events}</td>
        <td class="py-1 px-2 text-right">${cm.counts.non_mate_events}</td>
        <td class="py-1 px-2 text-right">${pct(cm.fpir.value)}</td>
        <td class="py-1 px-2 text-right">${pct(cm.fnir.value)}</td>
        <td class="py-1 px-2 text-right">${pct(cm.rank1)}</td></tr>`).join('');
  } else { bd.classList.add('hidden'); }

  drawDet(m); drawCmc(m);
}
function drawDet(m) {
  const curve = (m.det_curve || []).filter(p => p.fpir != null && p.fnir != null);
  if (detChart) detChart.destroy();
  detChart = new Chart(document.getElementById('det-chart'), {
    type: 'line', data: { labels: curve.map(p => p.threshold.toFixed(2)),
      datasets: [{ label: 'FPIR', data: curve.map(p => p.fpir), borderColor: '#ef4444', pointRadius: 0, borderWidth: 2 },
                 { label: 'FNIR', data: curve.map(p => p.fnir), borderColor: '#f59e0b', pointRadius: 0, borderWidth: 2 }] },
    options: { animation: false,
      scales: { x: { title: { display: true, text: 'soglia', color: '#6b7280' }, ticks: { color: '#6b7280', maxTicksLimit: 8, font: { size: 9 } }, grid: { color: 'rgba(75,85,99,.2)' } },
                y: { min: 0, max: 1, ticks: { color: '#6b7280', font: { size: 9 } }, grid: { color: 'rgba(75,85,99,.2)' } } },
      plugins: { legend: { labels: { color: '#9ca3af', font: { size: 10 } } },
        title: { display: !!(m.eer && m.eer.value != null), color: '#e5e7eb', font: { size: 11 },
                 text: (m.eer && m.eer.value != null) ? `EER ≈ ${(m.eer.value*100).toFixed(1)}% @ ${m.eer.threshold}` : '' } } },
  });
}
function drawCmc(m) {
  const cmc = m.cmc || [];
  if (cmcChart) cmcChart.destroy();
  if (!cmc.length) { cmcChart = null; return; }
  cmcChart = new Chart(document.getElementById('cmc-chart'), {
    type: 'line', data: { labels: cmc.map(p => 'rank ' + p.rank),
      datasets: [{ label: 'CMC', data: cmc.map(p => p.rate), borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.15)', pointRadius: 3, borderWidth: 2, fill: true, stepped: true }] },
    options: { animation: false, scales: { y: { min: 0, max: 1, ticks: { color: '#6b7280', font: { size: 9 } }, grid: { color: 'rgba(75,85,99,.2)' } }, x: { ticks: { color: '#6b7280', font: { size: 9 } }, grid: { color: 'rgba(75,85,99,.2)' } } }, plugins: { legend: { display: false } } },
  });
}
function applyThreshold() {
  const value = parseFloat(document.getElementById('thr-input').value);
  if (isNaN(value)) { document.getElementById('thr-msg').textContent = 'valore non valido'; return; }
  fetch('/api/settings/match_threshold', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ value }) })
    .then(r => r.json()).then(d => { document.getElementById('thr-msg').textContent = d.error ? '⚠ ' + d.error : `✓ soglia ${d.value} applicata${d.persisted ? ' e salvata nel .env' : ''}.`; })
    .catch(() => { document.getElementById('thr-msg').textContent = '⚠ errore di rete'; });
}

refreshSessions();
