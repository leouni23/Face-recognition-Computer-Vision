/* Offline review: data source (root + archive mode), campaign picker, session list with
   include/exclude, and the final export. Everything here is read-only w.r.t. the copied
   experiment: exclusions live in review_state.json, a new file written next to campaign.json.

   Shared globals from validation.html: esc(). */

let reviewCampaigns = [];
let reviewSessions = [];
let reviewCampaign = null;      // folder attualmente in revisione
let archiveMode = null;

// ── Sorgente dati ────────────────────────────────────────────────────────────
function loadReviewRoot() {
  return fetch('/api/review/root').then(r => r.json()).then(d => {
    const el = document.getElementById('review-root');
    if (el && !el.value) el.value = d.path || '';
    showRootMsg(d);
    return d;
  }).catch(() => {});
}

function showRootMsg(d) {
  const box = document.getElementById('review-root-msg');
  if (!box) return;
  if (d && d.ok) {
    box.className = 'text-xs text-emerald-400';
    box.textContent = `✓ cartella valida: ${d.n_campaigns} campagne · ${d.n_sessions} sessioni`;
  } else {
    box.className = 'text-xs text-red-400';
    box.textContent = d && d.errors && d.errors.length ? '✕ ' + d.errors.join('; ') : '';
  }
}

function applyReviewRoot() {
  const path = document.getElementById('review-root').value.trim();
  fetch('/api/review/root', {method:'POST', headers:{'Content-Type':'application/json'},
                             body: JSON.stringify({path})})
    .then(r => r.json().then(d => ({s:r.status, d}))).then(({s, d}) => {
      showRootMsg(d);
      if (s >= 400) return;
      loadArchiveMode(); loadReviewCampaigns();
    }).catch(() => alert('Errore di rete.'));
}

// ── Modalità archivio (sola lettura) ─────────────────────────────────────────
function loadArchiveMode() {
  return fetch('/api/review/mode').then(r => r.json()).then(m => {
    archiveMode = m;
    const badge = document.getElementById('arch-badge');
    const reason = document.getElementById('arch-reason');
    if (badge) badge.classList.toggle('hidden', !m.archive);
    if (reason) reason.textContent = m.archive ? ('— ' + m.reason) : m.reason;
    const sel = document.getElementById('arch-mode');
    if (sel) sel.value = m.override === null || m.override === undefined ? 'auto' : (m.override ? 'on' : 'off');
    // In archivio non si registra: nascondo wizard e barra campagna attiva (restano revisione,
    // metriche, confronto ed export).
    ['wizard'].forEach(id => { const el = document.getElementById(id); if (el) el.classList.toggle('hidden', !!m.archive); });
    document.querySelectorAll('[data-recording-only]').forEach(el => el.classList.toggle('hidden', !!m.archive));
    return m;
  }).catch(() => {});
}

function setArchiveMode(v) {
  const body = {archive: v === 'auto' ? null : (v === 'on')};
  fetch('/api/review/mode', {method:'POST', headers:{'Content-Type':'application/json'},
                             body: JSON.stringify(body)})
    .then(r => r.json()).then(() => loadArchiveMode()).catch(() => {});
}

// ── Campagne ─────────────────────────────────────────────────────────────────
function loadReviewCampaigns() {
  return fetch('/api/review/campaigns').then(r => r.json()).then(list => {
    reviewCampaigns = list || [];
    const sel = document.getElementById('review-campaign');
    if (!sel) return;
    sel.innerHTML = reviewCampaigns.map(c =>
      `<option value="${esc(c.folder)}">${esc(c.name)} — ${c.n_sessions} sess.` +
      `${c.n_excluded ? ' · ' + c.n_excluded + ' escluse' : ''}` +
      `${c.excluded ? ' [CAMPAGNA ESCLUSA]' : ''}${c.is_test ? ' [prova]' : ''}` +
      ` · ${(c.profiles || []).join(', ')}</option>`).join('')
      || '<option value="">— nessuna campagna —</option>';
    // la più recente è quella valida: preselezionata
    if (reviewCampaigns.length) { sel.value = reviewCampaigns[0].folder; loadReviewSessions(); }
  }).catch(() => {});
}

function currentCampaign() {
  return reviewCampaigns.find(c => c.folder === reviewCampaign) || null;
}

function toggleCampaignExcluded(checked) {
  if (!reviewCampaign) return;
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/flag`,
        {method:'POST', headers:{'Content-Type':'application/json'},
         body: JSON.stringify({excluded: !!checked, reason: checked ? 'esclusa dalla revisione' : ''})})
    .then(r => r.json()).then(() => loadReviewCampaigns()).catch(() => {});
}

// ── Sessioni + esclusione ────────────────────────────────────────────────────
function loadReviewSessions() {
  const sel = document.getElementById('review-campaign');
  reviewCampaign = sel ? sel.value : null;
  const camp = currentCampaign();
  const cb = document.getElementById('camp-excluded');
  if (cb) cb.checked = !!(camp && camp.excluded);
  if (!reviewCampaign) { reviewSessions = []; renderReviewSessions(); return; }
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/sessions`).then(r => r.json()).then(list => {
    reviewSessions = list || [];
    const pf = document.getElementById('f-profile');
    if (pf) {
      const profs = [...new Set(reviewSessions.map(s => s.profile).filter(Boolean))].sort();
      pf.innerHTML = '<option value="">tutti i profili</option>' +
        profs.map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join('');
    }
    renderReviewSessions();
  }).catch(() => {});
}

function filteredSessions() {
  const onlyInc = document.getElementById('f-included');
  const onlyRev = document.getElementById('f-toreview');
  const prof = document.getElementById('f-profile');
  return reviewSessions.filter(s =>
    (!onlyInc || !onlyInc.checked || !s.excluded) &&
    (!onlyRev || !onlyRev.checked || !s.reviewed) &&
    (!prof || !prof.value || s.profile === prof.value));
}

function sessionStatusCell(s) {
  if (s.excluded) return `<span class="text-amber-400">🚫 esclusa</span>${s.reason ? ' <span class="text-gray-600">' + esc(s.reason) + '</span>' : ''}`;
  if (s.reviewed) return '<span class="text-emerald-400">✓ rivista</span>';
  return '<span class="text-gray-500">⚠ da rivedere</span>';
}

function renderReviewSessions() {
  const box = document.getElementById('review-session-list');
  if (!box) return;
  const rows = filteredSessions();
  const counts = document.getElementById('sess-counts');
  if (counts) {
    const inc = reviewSessions.filter(s => !s.excluded).length;
    counts.textContent = `${inc} incluse / ${reviewSessions.length - inc} escluse`;
  }
  if (!rows.length) { box.innerHTML = '<p class="text-gray-600 py-2">Nessuna sessione con questi filtri.</p>'; return; }
  box.innerHTML = `<table class="w-full text-left">
    <thead class="text-gray-500"><tr>
      <th class="py-1 pr-2"><input type="checkbox" onchange="selectAllSessions(this.checked)" class="accent-emerald-500"></th>
      <th class="py-1 pr-2">sessione (verità a terra dichiarata)</th>
      <th class="py-1 px-2">profilo</th><th class="py-1 px-2">tipo</th>
      <th class="py-1 px-2 text-right">det</th><th class="py-1 px-2 text-right">label</th>
      <th class="py-1 px-2">stato</th>
    </tr></thead><tbody>` +
    rows.map(s => `<tr class="border-t border-gray-800/60 hover:bg-gray-800/40 ${s.excluded ? 'opacity-60' : ''}">
      <td class="py-1 pr-2"><input type="checkbox" class="sess-cb accent-emerald-500" value="${esc(s.session_id)}"></td>
      <td class="py-1 pr-2"><button onclick="openReviewSession('${esc(s.session_id)}')" class="text-left hover:text-emerald-300 font-mono">${esc(s.session_id)}</button></td>
      <td class="py-1 px-2">${esc(s.profile || '')}</td>
      <td class="py-1 px-2">${s.session_type === 'common' ? 'gruppo' : 'singola'}</td>
      <td class="py-1 px-2 text-right tabular-nums">${s.n_detections}</td>
      <td class="py-1 px-2 text-right tabular-nums">${s.n_labels}</td>
      <td class="py-1 px-2">${sessionStatusCell(s)}</td>
    </tr>`).join('') + '</tbody></table>';
}

function selectAllSessions(checked) {
  document.querySelectorAll('.sess-cb').forEach(cb => { cb.checked = checked; });
}

function selectedSessionIds() {
  return [...document.querySelectorAll('.sess-cb')].filter(cb => cb.checked).map(cb => cb.value);
}

function bulkExclude(excluded) {
  const ids = selectedSessionIds();
  if (!ids.length) { alert('Seleziona almeno una sessione.'); return; }
  const reason = document.getElementById('excl-reason').value.trim();
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/sessions`,
        {method:'POST', headers:{'Content-Type':'application/json'},
         body: JSON.stringify({session_ids: ids, excluded: !!excluded, reason: excluded ? reason : ''})})
    .then(r => r.json().then(d => ({s:r.status, d}))).then(({s, d}) => {
      if (s >= 400) { alert(d.error || 'Errore'); return; }
      loadReviewSessions(); loadReviewCampaigns();
    }).catch(() => alert('Errore di rete.'));
}

// Esclusione rapida della sessione aperta nel player (scorciatoia E)
function excludeCurrentSession(sessionId, reason) {
  if (!reviewCampaign || !sessionId) return;
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/sessions`,
        {method:'POST', headers:{'Content-Type':'application/json'},
         body: JSON.stringify({session_ids: [sessionId], excluded: true,
                               reason: reason || 'esclusa durante la revisione'})})
    .then(() => { loadReviewSessions(); loadReviewCampaigns(); }).catch(() => {});
}

// Verdetto della sessione aperta (scorciatoie C / X)
function setSessionVerdict(sessionId, verdict, notes) {
  if (!reviewCampaign || !sessionId) return Promise.resolve();
  return fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/sessions`,
        {method:'POST', headers:{'Content-Type':'application/json'},
         body: JSON.stringify({session_ids: [sessionId], reviewed: true, verdict, notes})})
    .then(() => { loadReviewSessions(); }).catch(() => {});
}

// Apre la sessione nel player esistente (validation_review.js)
function openReviewSession(sid) {
  const sel = document.getElementById('session-sel');
  if (sel) {
    if (![...sel.options].some(o => o.value === sid)) {
      const o = document.createElement('option'); o.value = sid; o.textContent = sid; sel.appendChild(o);
    }
    sel.value = sid;
    if (typeof loadSession === 'function') loadSession();
    const host = document.getElementById('gt-review-section');
    if (host) {
      const det = host.closest('details');
      if (det) det.open = true;                     // la revisione è in un toggle chiuso di default
      host.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  }
}

// Sessione precedente/successiva (scorciatoie ↑/↓)
function stepReviewSession(dir) {
  const list = filteredSessions();
  if (!list.length) return;
  const cur = document.getElementById('session-sel');
  const idx = list.findIndex(s => s.session_id === (cur ? cur.value : ''));
  const next = list[Math.max(0, Math.min(list.length - 1, (idx < 0 ? 0 : idx + dir)))];
  if (next) openReviewSession(next.session_id);
}

// ── Pre-transcodifica campagna ───────────────────────────────────────────────
function pretranscode() {
  if (!reviewCampaign) return;
  const msg = document.getElementById('pretrans-msg');
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/transcode`, {method:'POST'})
    .then(r => r.json().then(d => ({s:r.status, d}))).then(({s, d}) => {
      if (s >= 400) { msg.className = 'text-xs text-red-400'; msg.textContent = d.error || 'Errore'; return; }
      msg.className = 'text-xs text-gray-400';
      pollTranscode();
    }).catch(() => {});
}

function pollTranscode() {
  const msg = document.getElementById('pretrans-msg');
  fetch('/api/review/transcode/status').then(r => r.json()).then(p => {
    if (!p || !p.total) { msg.textContent = ''; return; }
    msg.textContent = `${p.done}/${p.total} video convertiti` + (p.errors && p.errors.length ? ` · ${p.errors.length} errori` : '');
    if (p.running) setTimeout(pollTranscode, 1500);
    else msg.className = 'text-xs text-emerald-400';
  }).catch(() => {});
}

// ── Export finale ────────────────────────────────────────────────────────────
function concludeValidation() {
  if (!reviewCampaign) { alert('Nessuna campagna selezionata.'); return; }
  const inc = reviewSessions.filter(s => !s.excluded).length;
  const exc = reviewSessions.length - inc;
  const toReview = reviewSessions.filter(s => !s.excluded && !s.reviewed).length;
  const warn = toReview ? `\n\nATTENZIONE: ${toReview} sessioni incluse non sono ancora state riviste.` : '';
  if (!confirm(`Concludere la validazione «${reviewCampaign}»?\n\n${inc} sessioni incluse, ${exc} escluse.${warn}\n\nVerranno calcolate le metriche finali e generato l'export per il paper.`)) return;
  const msg = document.getElementById('export-msg');
  msg.className = 'text-xs text-gray-400'; msg.textContent = 'Export in corso…';
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/export`, {method:'POST'})
    .then(r => r.json().then(d => ({s:r.status, d}))).then(({s, d}) => {
      if (s >= 400) { msg.className = 'text-xs text-red-400'; msg.textContent = '✕ ' + (d.error || 'Errore'); return; }
      msg.className = 'text-xs text-emerald-400';
      msg.innerHTML = `✓ Export creato: <span class="font-mono">${esc(d.dir)}</span> — ${d.n_included} sessioni incluse, ${d.n_excluded} escluse (profili: ${(d.profiles || []).join(', ')}). Apri <b>REPORT.md</b> per le tabelle del paper.`;
    }).catch(() => { msg.className = 'text-xs text-red-400'; msg.textContent = 'Errore di rete.'; });
}

// ── init ─────────────────────────────────────────────────────────────────────
loadReviewRoot().then(loadArchiveMode).then(loadReviewCampaigns);
