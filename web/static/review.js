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
  if (!reviewCampaign) { reviewSessions = []; renderReviewSessions(); return Promise.resolve(); }
  return fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/sessions`).then(r => r.json()).then(list => {
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
  if (s.reviewed) return '<span class="text-emerald-400">✓ confermata</span>';
  return '<span class="text-gray-500">⚠ da rivedere</span>';
}

// L'esperimento è stato fatto prima con un modello e poi con l'altro: la revisione va per blocchi.
// STANDARD prima, OPTIMIZED poi; ogni sezione ha conteggi, "seleziona tutto" e azioni PROPRIE.
const PROFILE_ORDER = ['standard', 'optimized-tx2'];
const PROFILE_LABEL = {'standard': 'STANDARD', 'optimized-tx2': 'OPTIMIZED-TX2'};
let collapsedProfiles = {};

function profileOf(s) {
  if (s.profile) return s.profile;                      // dal manifest session.json
  return /_optTX2/i.test(s.session_id) ? 'optimized-tx2' : 'standard';   // fallback dal nome
}

function profileGroups() {
  const rows = filteredSessions();
  const groups = {};
  rows.forEach(s => { (groups[profileOf(s)] = groups[profileOf(s)] || []).push(s); });
  const keys = PROFILE_ORDER.filter(p => groups[p]);
  Object.keys(groups).sort().forEach(p => { if (keys.indexOf(p) < 0) keys.push(p); });
  return keys.map(p => ({profile: p, rows: groups[p]}));
}

function countsFor(list) {
  const excluded = list.filter(s => s.excluded).length;
  const confirmed = list.filter(s => !s.excluded && s.reviewed).length;
  return {total: list.length, todo: list.length - excluded - confirmed, confirmed, excluded};
}

function renderReviewSessions() {
  const box = document.getElementById('review-session-list');
  if (!box) return;
  const all = countsFor(reviewSessions);
  const counts = document.getElementById('sess-counts');
  if (counts) counts.textContent =
    `${all.todo} da rivedere / ${all.confirmed} confermate / ${all.excluded} escluse`;
  const groups = profileGroups();
  if (!groups.length) { box.innerHTML = '<p class="text-gray-600 py-2">Nessuna sessione con questi filtri.</p>'; return; }
  box.innerHTML = groups.map(g => {
    const c = countsFor(g.rows);
    const open = !collapsedProfiles[g.profile];
    const pid = g.profile.replace(/[^a-z0-9]/gi, '');
    return `
    <section class="mb-4 border border-gray-800 rounded-xl overflow-hidden" data-profile="${esc(g.profile)}">
      <div class="flex flex-wrap items-center gap-2 px-3 py-2 bg-gray-950 cursor-pointer" onclick="toggleProfile('${esc(g.profile)}')">
        <span class="text-gray-400">${open ? '▼' : '▶'}</span>
        <span class="font-semibold text-sm">${esc(PROFILE_LABEL[g.profile] || g.profile)}</span>
        <span class="text-[11px] text-gray-500">${c.total} sessioni ·
          <b class="${c.todo ? 'text-amber-400' : 'text-gray-600'}">${c.todo} da rivedere</b> /
          <b class="text-emerald-400">${c.confirmed} confermate</b> /
          <b class="text-gray-400">${c.excluded} escluse</b></span>
      </div>
      <div class="${open ? '' : 'hidden'}">
        <div class="overflow-x-auto"><table class="w-full text-left">
          <thead class="text-gray-500"><tr>
            <th class="py-1 px-2"><input type="checkbox" class="accent-emerald-500" onchange="selectAllInProfile('${esc(g.profile)}', this.checked)" title="seleziona tutto (solo ${esc(g.profile)})"></th>
            <th class="py-1 pr-2">sessione (verità a terra dichiarata)</th>
            <th class="py-1 px-2">tipo</th>
            <th class="py-1 px-2 text-right">det</th><th class="py-1 px-2 text-right">label</th>
            <th class="py-1 px-2">stato</th>
          </tr></thead><tbody>` +
          g.rows.map(s => `<tr class="border-t border-gray-800/60 hover:bg-gray-800/40 ${s.excluded ? 'opacity-60' : ''}">
            <td class="py-1 px-2"><input type="checkbox" class="sess-cb accent-emerald-500" data-profile="${esc(g.profile)}" value="${esc(s.session_id)}"></td>
            <td class="py-1 pr-2"><button onclick="openReviewSession('${esc(s.session_id)}')" class="text-left hover:text-emerald-300 font-mono">${esc(s.session_id)}</button></td>
            <td class="py-1 px-2">${s.session_type === 'common' ? 'gruppo' : 'singola'}</td>
            <td class="py-1 px-2 text-right tabular-nums">${s.n_detections}</td>
            <td class="py-1 px-2 text-right tabular-nums">${s.n_labels}</td>
            <td class="py-1 px-2">${sessionStatusCell(s)}</td>
          </tr>`).join('') + `</tbody></table></div>
        <div class="flex flex-wrap items-center gap-2 px-3 py-2 border-t border-gray-800/60 bg-gray-900/50">
          <input id="reason-${pid}" class="fld flex-1 min-w-[9rem] text-xs" placeholder="motivo (es. ripetizione errata)" />
          <button onclick="bulkAction('${esc(g.profile)}','exclude')" class="px-2.5 py-1 rounded-lg bg-amber-700 hover:bg-amber-600 text-xs font-semibold">🚫 Escludi selezionate</button>
          <button onclick="bulkAction('${esc(g.profile)}','confirm')" class="px-2.5 py-1 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-xs font-semibold">✓ Conferma selezionate come valide</button>
          <button onclick="bulkAction('${esc(g.profile)}','reset')" class="px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs">↺ Annulla revisione</button>
          <span id="msg-${pid}" class="text-xs text-gray-500"></span>
        </div>
      </div>
    </section>`;
  }).join('');
}

function toggleProfile(profile) {
  collapsedProfiles[profile] = !collapsedProfiles[profile];
  renderReviewSessions();
}

// "Seleziona tutto" agisce SOLO dentro la sua sezione, mai sull'altro profilo.
function selectAllInProfile(profile, checked) {
  document.querySelectorAll(`.sess-cb[data-profile="${profile}"]`).forEach(cb => { cb.checked = checked; });
}

function selectedSessionIds(profile) {
  const sel = profile ? `.sess-cb[data-profile="${profile}"]` : '.sess-cb';
  return [...document.querySelectorAll(sel)].filter(cb => cb.checked).map(cb => cb.value);
}

// Tre azioni dal significato inequivocabile. "Conferma" scrive reviewed:true — senza quello
// l'operazione sarebbe un no-op (era esattamente il bug che ha fatto perdere due ore di lavoro).
const BULK = {
  exclude: {body: r => ({excluded: true, reviewed: true, reason: r || 'esclusa in revisione'}),
            done: n => `🚫 ${n} sessioni escluse`},
  confirm: {body: () => ({excluded: false, reviewed: true, reason: ''}),
            done: n => `✓ ${n} sessioni confermate valide`},
  reset:   {body: () => ({excluded: false, reviewed: false, reason: ''}),
            done: n => `↺ revisione annullata per ${n} sessioni`},
};

function bulkAction(profile, action) {
  const pid = profile.replace(/[^a-z0-9]/gi, '');
  const msg = document.getElementById('msg-' + pid);
  const ids = selectedSessionIds(profile);
  const setMsg = (t, cls) => { if (msg) { msg.className = 'text-xs ' + (cls || 'text-gray-500'); msg.textContent = t; } };
  if (!ids.length) { setMsg('Seleziona almeno una sessione in questa sezione.', 'text-amber-400'); return; }
  const spec = BULK[action];
  const reasonEl = document.getElementById('reason-' + pid);
  const body = Object.assign({session_ids: ids}, spec.body(reasonEl ? reasonEl.value.trim() : ''));
  setMsg('salvataggio…');
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/sessions`,
        {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json().then(d => ({s:r.status, d}))).then(({s, d}) => {
      if (s >= 400) { setMsg('✕ ' + (d.error || 'Errore'), 'text-red-400'); return; }
      const before = countsFor(reviewSessions.filter(x => profileOf(x) === profile)).todo;
      loadReviewSessions().then(() => {
        const after = countsFor(reviewSessions.filter(x => profileOf(x) === profile)).todo;
        const el = document.getElementById('msg-' + pid);
        if (el) {
          el.className = 'text-xs text-emerald-400';
          el.textContent = `${spec.done(ids.length)} · "da rivedere" ${before} → ${after}`;
        }
      });
      loadReviewCampaigns();
    }).catch(() => setMsg('Errore di rete.', 'text-red-400'));
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

// ── Modale di revisione ──────────────────────────────────────────────────────
// Il pannello si apre SOPRA la lista: niente scrollIntoView, così la posizione nella lista e le
// checkbox restano dove sono. Aprire un'altra sessione a modale aperta ne cambia solo il contenuto.
let modalOpen = false;

function openReviewSession(sid) {
  const sel = document.getElementById('session-sel');
  if (!sel) return;
  if (![...sel.options].some(o => o.value === sid)) {
    const o = document.createElement('option'); o.value = sid; o.textContent = sid; sel.appendChild(o);
  }
  sel.value = sid;
  if (typeof loadSession === 'function') loadSession();
  showReviewModal(sid);
}

function showReviewModal(sid) {
  const ov = document.getElementById('review-modal');
  const host = document.getElementById('gt-review-section');
  const slot = document.getElementById('review-modal-body');
  if (!ov || !host || !slot) return;
  if (!modalOpen) {
    const det = host.closest('details');
    if (det) det.open = true;          // la sezione vive in un toggle chiuso di default
    slot.appendChild(host);            // sposto il pannello dentro la modale (stesso nodo: stato intatto)
    ov.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    modalOpen = true;
  }
  const t = document.getElementById('review-modal-title');
  if (t) t.textContent = sid || '';
}

function closeReviewModal() {
  const ov = document.getElementById('review-modal');
  const host = document.getElementById('gt-review-section');
  // La casa del pannello è il toggle di Configurazione (dove l'init lo sposta); se non c'è,
  // il contenitore originale della pagina.
  const home = document.getElementById('gt-review-host') || document.getElementById('gt-review-home');
  if (!ov || !modalOpen) return;
  if (host && home) home.appendChild(host);   // rimetto il pannello al suo posto
  ov.classList.add('hidden');
  document.body.style.overflow = '';
  modalOpen = false;
  if (typeof stopPlay === 'function') stopPlay();
}

// Sessione precedente/successiva SOLO dentro il gruppo di profilo di quella aperta (↑/↓)
function stepReviewSession(dir) {
  const cur = document.getElementById('session-sel');
  const curId = cur ? cur.value : '';
  const curSession = reviewSessions.find(s => s.session_id === curId);
  const prof = curSession ? profileOf(curSession) : null;
  const list = filteredSessions().filter(s => !prof || profileOf(s) === prof);
  if (!list.length) return;
  const idx = list.findIndex(s => s.session_id === curId);
  const next = list[Math.max(0, Math.min(list.length - 1, (idx < 0 ? 0 : idx + dir)))];
  if (next && next.session_id !== curId) openReviewSession(next.session_id);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && modalOpen) { e.preventDefault(); closeReviewModal(); }
});

// ── Recupero della selezione precedente ──────────────────────────────────────
// La vecchia "Includi selezionate" non scriveva `reviewed`, quindi era un no-op: ma ogni sessione
// toccata ha comunque lasciato un record nello stato. Questo lo riconverte in inclusione/esclusione
// reale — anteprima obbligatoria, backup automatico, e due guardie contro stati ambigui.
function recoverSelection() {
  if (!reviewCampaign) { alert('Nessuna campagna selezionata.'); return; }
  fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/recovery`)
    .then(r => r.json().then(d => ({s: r.status, d}))).then(({s, d}) => {
      if (s >= 400) { alert(d.error || 'Errore'); return; }
      if (!d.can_apply) {
        alert('Recupero non applicabile.\n\n' + d.blocked_reason +
              `\n\nSessioni: ${d.n_sessions} · toccate: ${d.n_touched} · mai toccate: ${d.n_untouched}`);
        return;
      }
      const perProf = Object.keys(d.per_profile).sort().map(p =>
        `  • ${p}: ${d.per_profile[p].n_touched} confermate, ${d.per_profile[p].n_untouched} da escludere`).join('\n');
      const stamps = (d.touch_timestamps || []).map(t => `  • ${t.at.replace('T', ' ')} → ${t.n} sessioni`).join('\n');
      const sample = Object.keys(d.per_profile).sort()
        .flatMap(p => d.per_profile[p].untouched).slice(0, 12).map(x => '  – ' + x).join('\n');
      const ok = confirm(
        `RECUPERO SELEZIONE — campagna ${reviewCampaign}\n\n` +
        `Sessioni totali: ${d.n_sessions}\n` +
        `Toccate in revisione (→ confermate valide): ${d.n_touched}\n` +
        `Mai toccate (→ ESCLUSE): ${d.n_untouched}\n\n` +
        `Per profilo:\n${perProf}\n\n` +
        `Quando le hai selezionate:\n${stamps}\n\n` +
        `Sessioni che verranno ESCLUSE:\n${sample}${d.n_untouched > 12 ? '\n  … e altre' : ''}\n\n` +
        `Viene creato un backup dello stato prima di scrivere. Procedo?`);
      if (!ok) return;
      fetch(`/api/review/${encodeURIComponent(reviewCampaign)}/recovery`,
            {method: 'POST', headers: {'Content-Type': 'application/json'},
             body: JSON.stringify({confirm: true})})
        .then(r => r.json().then(d2 => ({s: r.status, d2}))).then(({s, d2}) => {
          if (s >= 400) { alert(d2.error || 'Errore'); return; }
          alert(`Recupero applicato.\n\n${d2.n_confirmed} sessioni confermate valide\n` +
                `${d2.n_excluded} sessioni escluse\n\nBackup: ${d2.backup}`);
          loadReviewSessions(); loadReviewCampaigns();
        }).catch(() => alert('Errore di rete.'));
    }).catch(() => alert('Errore di rete.'));
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
