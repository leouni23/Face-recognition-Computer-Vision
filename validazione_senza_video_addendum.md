# Addendum — Validazione senza video, ground truth dal vivo

> Companion di `DOCUMENTAZIONE.md` §11 (Modalità di Validazione). Descrive la modalità di
> validazione **senza registrazione video**, con verità a terra stabilita **dal vivo** per
> presentazione controllata. È la modalità di **default**; la registrazione video resta
> disponibile come fallback opt-in.

## 1. Motivazione

Il sistema in produzione **non salva immagini** (design GDPR data-minimization). La modalità
di validazione storica era l'unica deroga: registrava video annotato per camera per poter
fornire la ground truth *a posteriori* rivedendo il filmato. Questo:

1. **reintroduce immagini sensibili su disco** (volti identificabili), e
2. impone i vincoli dello storage del video (limite 4 GB di FAT32 → segmentazione, spazio,
   filesystem rimovibili).

L'osservazione chiave: per misurare l'accuratezza open-set 1:N **non serve il filmato**, serve
la **verità a terra per ogni detection**. Se la otteniamo *dal vivo* tramite **presentazione
controllata** dei soggetti, il video diventa superfluo e si ripristina la postura
«no images on disk».

## 2. Cosa cambia (e cosa NO)

- **Metriche invariate.** FPIR/FNIR/DET/EER/CMC e la matematica in `core/validation_metrics.py`
  **non cambiano**. Entrambe le modalità producono gli stessi due file — `detections.jsonl`
  (un record per volto per frame, con `raw_cosine_distance` + ranking candidati) e
  `labels.jsonl` (override manuali) — e il calcolo è identico. La GT automatica è letta
  direttamente dal record di detection (`_declared_truth`), con la label manuale che vince.
- **Default = nessuna immagine.** `VALIDATION_RECORD_VIDEO=false`. Nessun
  `cv2.VideoWriter`, nessuna cartella `video/`, nessun byte di immagine su disco: solo log
  testuali append-only.
- **Video come fallback.** Toggle per-sessione nel wizard `/validation` o default globale in
  `.env`. Con video attivo il comportamento è quello storico (player a frame nella review).

## 3. Le tre fonti di ground truth (presentazione controllata)

Conforme a **ISO/IEC 19795** (valutazione di scenario): l'identità vera è nota perché la
presentazione dei soggetti è **controllata e dichiarata**, non desunta da immagini archiviate.

### 3.1 Attraversamenti — soggetto dichiarato (invariato)
L'operatore dichiara chi sta attraversando (`set_run_context(subject, preset)`); ogni detection
è taggata e riceve la GT automatica: `S…` → `true_person_id` (mate), `U…` → `non_mate`.
Questa modalità non aveva bisogno del video già prima.

### 3.2 Sessione comune — mappa dei posti (nuovo)
Per N soggetti statici, l'operatore dichiara una **mappa dei posti**: una lista ordinata di
posizioni occupate (**sinistra→destra, fila per fila**) → etichetta soggetto (`S…`/`U…`),
risolte contro il registro (`subject_truth`).

A ogni frame, in `record()`:
1. le detection sono **ordinate per posizione del bounding box** — `bbox = [top, right,
   bottom, left]`, centro orizzontale `cx=(left+right)/2`, verticale `cy=(top+bottom)/2`;
2. **raggruppamento multi-fila**: ordinamento per `cy`, nuova fila quando il salto rispetto
   all'ancora di fila supera **~0.6 × l'altezza mediana del volto** (tolleranza al jitter
   verticale); dentro ogni fila ordinamento `cx` crescente; file concatenate dall'alto in basso;
3. la *k*-esima detection ordinata riceve la GT del *k*-esimo posto, scritta nel record di
   detection con le stesse chiavi degli attraversamenti.

**Deterministico, senza video, senza click.** Robustezza: se `#volti ≠ #posti`, si etichettano
solo le posizioni allineate (`min`), lasciando le eccedenze **senza verità** invece di
assegnarle male.

> Assunzione operativa: i soggetti restano nelle posizioni dichiarate e l'ordine della mappa
> rispecchia la disposizione fisica (sinistra→destra, file dall'alto). Cambi di disposizione →
> ri-applicare la mappa (vale dai frame successivi).

### 3.3 Correzione live click-to-assign (fallback manuale)
`GET /api/validation/live/<cam>` espone i box dell'ultimo frame elaborato (solo metadati:
`frame_index` + bbox + identità predetta, **nessuna immagine persistita**). Nella UI i box sono
sovrapposti allo stream MJPEG: click su un volto → scelta soggetto → `POST /<id>/labels` con
`{camera_id, frame_index, face_id, true_person_id|non_mate}`. La label manuale **vince** sulla
GT automatica (a livello di frame). Per correzioni a livello di **evento** si usa il pannello di
revisione (assegnazione bulk per evento), che funziona anche senza video.

## 4. Privacy e conformità

- Ripristina **«no images on disk»**: di default solo log testuali (distanze, ranking, bbox,
  etichette) lasciano la pipeline; nessun volto archiviato.
- Il **DB biometrico** resta su storage interno; in modalità senza video sulla destinazione
  vanno **solo** gli artefatti append-only di testo (niente più vincoli FAT32 sul video).
- Ground truth per **presentazione controllata** (ISO/IEC 19795), non per analisi di immagini
  conservate → minimizzazione dei dati coerente col GDPR.

## 5. Compromesso (esplicito)

Senza video **non esiste un audit trail a posteriori**: non è possibile rivedere il filmato per
contestare un'etichetta, ispezionare un falso positivo o rifare il labeling da capo. La verità a
terra è quella dichiarata dal vivo. Nei contesti che richiedono quella verificabilità (es.
contenzioso, audit esterno), **attivare la registrazione video** come fallback opt-in,
accettando il ritorno delle immagini su disco.

## 6. Riferimenti nel codice

- `core/validation.py` — `start(record_video=…)`, contatore `_frames` (frame_index unica
  fonte), `record()` (video condizionale + GT seating), `set_seating_map()`,
  `live_detections()`, `_order_faces_by_seat()`, `build_protocol_md()` (condizionale).
- `web/app.py` — `POST /api/validation/start` (`record_video`), `POST /api/validation/seating`,
  `GET /api/validation/live/<cam>`, riuso di `POST /api/validation/<id>/labels`.
- `web/static/validation.html` + `validation_review.js` — toggle video, editor mappa posti,
  pannello live click-to-assign, review timeline-only senza video.
- `config/settings.py` / `.env.example` — `VALIDATION_RECORD_VIDEO` (default `false`).
- `core/validation_metrics.py` — **immutato**.
