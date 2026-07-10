# Face ID — Documentazione Tecnica Completa

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Architettura del Sistema](#2-architettura-del-sistema)
3. [Componenti Principali](#3-componenti-principali)
4. [Tracking Posizioni e Mappa](#4-tracking-posizioni-e-mappa)
5. [Flusso Dati End-to-End](#5-flusso-dati-end-to-end)
6. [Sicurezza della Web UI](#6-sicurezza-della-web-ui)
7. [Stack Tecnologico](#7-stack-tecnologico)
8. [Conformità GDPR](#8-conformità-gdpr)
9. [Configurazione e Deploy](#9-configurazione-e-deploy)
10. [Strumentazione delle Prestazioni](#10-strumentazione-delle-prestazioni)
11. [Modalità di Validazione (esperimenti)](#11-modalità-di-validazione-esperimenti)
12. [Bot Telegram](#12-bot-telegram)
13. [Containerizzazione](#13-containerizzazione)
14. [Argomenti da Studiare per la Presentazione](#14-argomenti-da-studiare-per-la-presentazione)

---

## 1. Panoramica del Progetto

Sistema di riconoscimento facciale in tempo reale con interfaccia web. Acquisisce il video da una o più camere, rileva i volti, li confronta con un database di persone iscritte, mostra i risultati live nel browser e **traccia la posizione dei soggetti identificati nel tempo**, proiettandola su una planimetria della stanza.

**Funzionalità principali:**

- Rilevamento e riconoscimento volti in tempo reale (~25 FPS)
- Accelerazione GPU tramite CUDA (NVIDIA), fallback automatico su CPU
- Web UI per monitoraggio, iscrizione e gestione persone
- Tracking della posizione nel tempo (storico + mappa live) con due modalità di calibrazione camera
- Embedding biometrici cifrati a riposo (AES-128-CBC + HMAC-SHA256)
- Autenticazione opzionale della Web UI, guard CSRF e security headers
- Conformità GDPR: consenso, data retention, diritto all'oblio

---

## 2. Architettura del Sistema

```text
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│  Orchestrator: avvia camera threads, pipeline, Flask server      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
  │ CameraStream │  │FaceIdPipeline│  │   Flask (web/)   │
  │  (thread)    │  │  (thread)    │  │  (daemon thread) │
  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
         │                 │                    │
         │  frame BGR      │  results +         │  MJPEG / SSE / REST
         └────────────────►│  posizioni mappa   │  mappa / calibrazione
                           │◄───────────────────┤ broadcaster
                           │                    │  (thread-safe bridge)
                    ┌──────▼──────┐             │
                    │  Database   │             │
                    │  SQLite /   │             │
                    │  PostgreSQL │             │
                    └─────────────┘             ▼
                                        ┌──────────────┐
                                        │   Browser    │
                                        │ dashboard ·  │
                                        │ mappa live · │
                                        │ calibrazione │
                                        └──────────────┘
```

### Struttura cartelle

```text
Face-recognition-Computer-Vision/
├── config/
│   └── settings.py          # Configurazione centralizzata (Pydantic)
├── core/
│   ├── camera.py            # Acquisizione video threaded (webcam + RTSP)
│   ├── detector.py          # Rilevamento volti (SCRFD) + embedding ArcFace
│   ├── recognizer.py        # Matching embedding (cosine distance, matrice precalcolata)
│   ├── geometry.py          # Omografia + calibrazione polare (distanza dal viso)
│   └── pipeline.py          # Pipeline: detect → recognize → log → proiezione mappa
├── database/
│   ├── models.py            # Person, FaceTemplate, RecognitionEvent,
│   │                        #   PositionLog, CameraCalibration
│   ├── session.py           # Engine SQLAlchemy + context manager
│   └── repository.py        # CRUD, traiettorie, calibrazioni
├── privacy/
│   ├── crypto.py            # Cifratura Fernet + guard chiave placeholder
│   └── retention.py         # Pulizia automatica dati scaduti (GDPR Art. 5)
├── ui/
│   └── display.py           # Annotazione frame OpenCV + FPS counter
├── web/
│   ├── app.py               # Flask: stream, SSE, enrollment, mappa, calibrazione
│   ├── broadcaster.py       # Bridge thread-safe con conteggio viewer
│   └── static/
│       ├── index.html       # Dashboard (Tailwind CSS + Vanilla JS)
│       ├── calibrate.html   # Pagina calibrazione con live feed
│       └── maps/            # Planimetria caricata
├── scripts/
│   ├── init_db.py           # Creazione tabelle (idempotente)
│   ├── enroll.py            # Iscrizione persona da CLI
│   └── delete_person.py     # Cancellazione dati biometrici (GDPR Art. 17)
└── main.py                  # Entry point
```

---

## 3. Componenti Principali

### 3.1 Acquisizione Video — `core/camera.py`

`CameraStream` apre una webcam locale (indice intero) o telecamera IP (URL RTSP) in un thread separato per non bloccare la pipeline.

**Meccanismo:**

- Thread daemon legge frame in loop continuo
- Coda interna `maxsize=2`: scarta i frame più vecchi se la pipeline è lenta (evita accumulo)
- Per RTSP: riconnessione automatica fino a 30 tentativi, con ritardo fisso di 2 s tra un tentativo e l'altro

**Perché un thread separato?**
`cv2.VideoCapture.read()` è bloccante: senza thread, ogni chiamata aspetterebbe il frame successivo dalla camera, introducendo latenza variabile e bloccando tutto il resto del programma.

---

### 3.2 Rilevamento e Embedding — `core/detector.py`

Usa **InsightFace** con il pacchetto di modelli **buffalo_l** (detection **SCRFD** + recognition **ArcFace**).

**Due fasi:**

1. **Face Detection** (SCRFD — *Sample and Computation Redistribution for Face Detection*): localizza i volti nel frame → bbox `(x1, y1, x2, y2)` + 5 landmark facciali
2. **Face Recognition** (ArcFace): estrae l'**embedding**, un vettore di 512 numeri float32 che rappresenta univocamente le caratteristiche del volto

**Embedding ArcFace:**

- Output di una ResNet-100 addestrata con *ArcFace loss* (additive angular margin)
- Già normalizzato L2: `||e|| = 1` → vettore sulla sfera unitaria in 512 dimensioni
- Due volti della stessa persona → embedding vicini; persone diverse → embedding lontani

**GPU vs CPU:**

- Con `USE_GPU=true`: usa `CUDAExecutionProvider` (ONNX Runtime → NVIDIA CUDA)
- Su Windows: le DLL CUDA vengono caricate dai pacchetti `nvidia-*-cu12` installati via pip (il PATH viene aggiornato all'avvio)
- L'inizializzazione del modello è protetta da un lock (più camera-thread potrebbero richiederla in parallelo)
- Speedup tipico: 5-10× rispetto a CPU

---

### 3.3 Riconoscimento — `core/recognizer.py`

`FaceRecognizer` confronta l'embedding del volto rilevato con tutti gli embedding iscritti nel database.

**Algoritmo:**

```python
# Template: matrice N×512 PRECALCOLATA in load() (non ricostruita a ogni frame)
# Query:    vettore 1×512 (volto da identificare)

cosine_distances = 1 - (template_matrix @ query_embedding)
# dato che tutti i vettori sono normalizzati L2:
# dot product = cosine similarity → distance = 1 - similarity

best_idx = argmin(cosine_distances)
best_distance = cosine_distances[best_idx]
confidence = clamp(1 - best_distance, 0.0, 1.0)   # clamp contro errori float32

if best_distance < MATCH_THRESHOLD:   # default: 0.5
    return (person_id, name, confidence)
else:
    return (None, "Sconosciuto", confidence)
```

**Perché cosine distance?**
Gli embedding ArcFace sono ottimizzati per la similarità coseno. Due embedding della stessa persona hanno cosine similarity tipicamente > 0.6 (distanza < 0.4). `MATCH_THRESHOLD=0.5` è il valore consigliato da InsightFace.

**Performance:** la matrice dei template viene costruita una sola volta al caricamento e scambiata atomicamente (i thread non vedono mai stati misti); il confronto per frame è una singola moltiplicazione matrice-vettore NumPy → O(N×512), veloce anche con centinaia di persone.

---

### 3.4 Pipeline — `core/pipeline.py`

`FaceIdPipeline` orchestra l'intero flusso per ogni frame:

1. **Reload periodico** (ogni 60 s): rilegge dal database gli embedding e le calibrazioni camera. Permette di iscrivere persone e calibrare camere a caldo, senza riavvii
2. **Detect & Encode**: chiama `detect_and_encode(frame)`
3. **Identify**: per ogni volto rilevato, chiama `recognizer.identify(embedding)`
4. **Log eventi**: se la persona è riconosciuta e non è stata loggata negli ultimi 10 s, registra l'evento e aggiorna `last_seen`
5. **Log posizioni**: per ogni soggetto identificato, salva la posizione (bounding box) a cadenza `POSITION_LOG_INTERVAL` e — se la camera è calibrata — la proietta sulla planimetria con smoothing EMA (vedi §4)

**Throttling dei log** (`_LOG_COOLDOWN=10s` per gli eventi, `POSITION_LOG_INTERVAL=1s` per le posizioni): evita di scrivere migliaia di righe al secondo nel database quando una persona è ferma davanti alla camera.

---

### 3.5 Database — `database/`

**Cinque tabelle:**

| Tabella | Scopo |
| --- | --- |
| `Person` | Dati anagrafici: nome, consenso, date |
| `FaceTemplate` | Embedding cifrati (Fernet) |
| `RecognitionEvent` | Log: chi, quando, con quale confidenza, su quale camera |
| `PositionLog` | Posizione nel tempo: centro/dimensione bbox in pixel + coordinate mappa (`world_x/world_y`) |
| `CameraCalibration` | Calibrazione per camera: punti di riferimento + parametri del proiettore (omografia o polare) |

**ORM:** SQLAlchemy v2 con pattern Repository — tutta la logica di accesso ai dati è in `repository.py`, mai SQL grezzo nei moduli di business logic.

**SQLite vs PostgreSQL:** il progetto supporta entrambi tramite `DATABASE_URL`. SQLite non richiede installazioni aggiuntive ed è sufficiente per uso personale/locale; per deployment multi-utente si passa a PostgreSQL cambiando solo la variabile d'ambiente.

**Cancellazione a cascata:** eliminando una `Person` vengono eliminati automaticamente template e traiettorie (`ON DELETE CASCADE`) — requisito per il diritto all'oblio.

---

### 3.6 Cifratura — `privacy/crypto.py`

Gli embedding biometrici non vengono mai salvati in chiaro nel database.

**Schema di cifratura:**

```text
secret_key (stringa)
    → SHA-256 → 32 bytes
    → Base64url encode
    → chiave Fernet valida

Fernet = AES-128-CBC + HMAC-SHA256 + timestamp
```

**Fernet** (dalla libreria `cryptography`) garantisce:

- **Confidenzialità**: AES-128-CBC cifra il payload
- **Integrità**: HMAC-SHA256 previene manomissioni
- **Autenticità**: solo chi conosce la chiave può decifrare

Se `BIOMETRIC_SECRET_KEY` viene ruotata, tutti i template esistenti diventano automaticamente inutilizzabili (le persone devono essere re-iscritte). Se la chiave è **vuota o è ancora il placeholder**, il sistema solleva un errore esplicito e si rifiuta di cifrare: impossibile avviare il trattamento con una chiave non configurata.

---

### 3.7 Web UI — `web/`

**Flask** espone quattro categorie di endpoint:

**MJPEG Streaming** (`/stream/<camera_id>`):
Protocollo multipart/x-mixed-replace — il server invia frame JPEG in sequenza nella stessa connessione HTTP. Il browser li interpreta come video live senza plugin. L'encoding JPEG avviene **solo se c'è almeno un viewer collegato** (conteggio nel broadcaster) e vengono inviati solo i frame nuovi.

**Server-Sent Events** (`/api/events`):
Connessione HTTP long-lived: il server invia eventi JSON ogni volta che viene riconosciuta una persona. Il browser aggiorna la lista riconoscimenti in tempo reale senza polling.

**Enrollment API** (processo stateful in 3 step):

1. `/api/enroll/start` — inizializza la sessione con nome (validato) e numero campioni
2. `/api/enroll/capture` — cattura e analizza un frame dal broadcaster (senza aprire una seconda connessione alla camera)
3. `/api/enroll/save` — media degli embedding, **rinormalizzazione L2** e salvataggio cifrato

**Mappa e calibrazione:**

- `/api/map` (GET/POST) — lettura/upload della planimetria (validata come immagine, max 10 MB)
- `/api/calibration/<camera>` (POST) — salva la calibrazione (polare od omografia)
- `/api/calibration/<camera>/sample` (POST) — cattura un campione per la calibrazione polare
- `/api/positions/map` — ultima posizione sulla mappa per ogni persona
- `/api/persons/<id>/trajectory` — storico posizioni di un soggetto
- `/calibrate` — pagina di calibrazione guidata con **video live** della camera

**Broadcaster** (`web/broadcaster.py`):
Bridge thread-safe tra i worker thread (che elaborano i frame) e Flask (che li serve al browser). Usa `threading.Lock` per i frame condivisi, `queue.Queue` per gli eventi SSE e tiene il conteggio dei viewer MJPEG per evitare encoding inutile.

---

## 4. Tracking Posizioni e Mappa

### 4.1 Cosa viene salvato

Per ogni soggetto **identificato** (mai per gli sconosciuti), a cadenza configurabile (default 1 punto/s per persona per camera), la pipeline salva in `PositionLog`:

- centro e dimensione del bounding box del volto (pixel) + dimensioni del frame
- coordinate sulla planimetria (`world_x/world_y`) se la camera è calibrata
- camera, timestamp e confidenza

Il **re-identification tra camere è implicito**: la stessa persona ha lo stesso `person_id` su ogni camera grazie all'embedding ArcFace — non serve alcun algoritmo aggiuntivo di "ricucitura".

### 4.2 Calibrazione: due modalità

La proiezione da pixel del frame a coordinate della planimetria dipende da cosa inquadra la camera.

#### Modalità polare — *la camera NON vede il pavimento* (es. webcam su scrivania)

Senza pavimento inquadrato l'omografia non è applicabile. Si usa il metodo documentato in letteratura per la **stima monoculare della distanza dalla dimensione apparente del volto**: il viso umano adulto ha dimensioni fisiche quasi costanti, quindi per il modello pinhole:

```text
distanza   d = k / h_px                  (h_px = altezza del volto in pixel)
angolo     β = heading + s·(x_norm − ½)  (x_norm = posizione orizzontale nel frame)
posizione  P = C + d·(cos β, sin β)      (C = posizione della camera sulla mappa)
```

Le costanti (`heading` = orientamento camera, `s` ≈ FOV orizzontale, `k` = costante di distanza) vengono risolte **ai minimi quadrati** da ≥2 campioni di riferimento: l'utente si mette in un punto noto della stanza, clicca quel punto sulla mappa e preme "Cattura campione". I campioni devono trovarsi in zone orizzontali diverse dell'inquadratura (controllo di degenerazione).

#### Modalità omografia — *la camera vede il pavimento* (es. telecamera a soffitto)

Una matrice 3×3 stimata con `cv2.findHomography` da ≥4 coppie di punti **a terra** (pixel ↔ planimetria) mappa il piano del pavimento sulla mappa. Viene proiettato il centro-basso del bounding box.

### 4.3 Smoothing e precisione

- La dimensione del volto in pixel è rumorosa frame a frame → le posizioni sulla mappa sono filtrate con una **media mobile esponenziale** (α = 0.4, reset dopo 5 s senza avvistamenti)
- La precisione è **a livello di zona** ("vicino al letto / alla porta"), non centimetrica: il volto sta ~1.5 m sopra il piano modellato (omografia) e la dimensione del viso varia tra persone di ±10% (polare)

### 4.4 Visualizzazione

- **Vista Mappa** (dashboard): pallini live dei soggetti sulla planimetria, aggiornati ogni 2 s
- **Storico** (per persona): traiettoria su canvas per ogni camera, gradiente blu→verde dal punto più vecchio al più recente, intervallo da 15 minuti a 24 ore

---

## 5. Flusso Dati End-to-End

### 5.1 Riconoscimento in tempo reale

```text
[Webcam]
    │ frame BGR
    ▼
[CameraStream thread]
    │ frame BGR
    ▼
[FaceIdPipeline.process_frame()]
    │
    ├─[detect_and_encode(frame)]
    │     InsightFace: SCRFD detection → ArcFace embedding
    │     Output: [(top,right,bottom,left), embedding_512d]
    │
    ├─[recognizer.identify(embedding)]
    │     cosine_distance = 1 - (templates @ embedding)
    │     Output: (person_id, name, confidence)
    │
    ├─[repository.log_event()]          → DB (throttle 10 s)
    ├─[repository.log_position()]       → DB (1/s, con proiezione mappa + EMA)
    │
    └─ Output: [(location, person_id, name, confidence), ...]
         │
         ├─► broadcaster.push_frame(camera_id, annotated_frame)
         │       → JPEG encode SOLO se c'è un viewer collegato
         │
         └─► broadcaster.push_event(camera_id, name, confidence)
                 → queue dei subscriber SSE

[Browser — MJPEG]   GET /stream/0      → frame JPEG live
[Browser — SSE]     GET /api/events    → {"camera":"0","name":"Leo","confidence":0.91}
[Browser — Mappa]   GET /api/positions/map → [{"name":"Leo","world_x":...,"world_y":...}]
```

### 5.2 Iscrizione nuova persona (Web UI)

```text
[Browser] POST /api/enroll/start {"name": "Leo", "samples": 5}
    └─► _enroll_session = {name: "Leo", embeddings: [], required: 5}

[Browser] POST /api/enroll/capture (×5)
    └─► broadcaster.get_raw_frame("0") → frame ndarray
        detect_and_encode(frame) → embedding
        _enroll_session["embeddings"].append(embedding)

[Browser] POST /api/enroll/save
    └─► mean = np.mean(embeddings, axis=0)
        mean /= np.linalg.norm(mean)        ← rinormalizzazione L2 (essenziale!)
        repository.add_template(person, mean)
            → encrypt_embedding() → FaceTemplate → DB
        pipeline.force_reload() → riconoscimento attivo subito
```

### 5.3 Calibrazione polare di una camera

```text
[Browser /calibrate]
    1. clic sulla planimetria → posizione della camera (C)
    2. l'utente si mette in un punto noto della stanza
       clic di quel punto sulla mappa
       POST /api/calibration/0/sample → {x_norm, h_px} del volto live
    3. ripetuto per ≥2 punti in zone diverse dell'inquadratura
    4. POST /api/calibration/0 {mode:"polar", camera:C, samples:[...]}
       └─► solve_polar_calibration() → {heading, scale, k}  (minimi quadrati)
           pipeline.force_reload() → proiezione attiva subito
```

---

## 6. Sicurezza della Web UI

| Misura | Implementazione |
| --- | --- |
| **Basic Auth opzionale** | `WEB_PASSWORD` nel `.env`; confronto in tempo costante con `hmac.compare_digest` (previene timing attack) |
| **Default sicuro** | Il server ascolta solo su `127.0.0.1`; se esposto (`--host 0.0.0.0`) senza password viene loggato un warning esplicito |
| **Guard CSRF** | Le richieste mutanti (POST/PUT/PATCH/DELETE) con header `Sec-Fetch-Site: cross-site` vengono rifiutate (403) — l'header è impostato dal browser e non falsificabile da una pagina malevola |
| **Security headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` |
| **Validazione input** | Nome persona: regex `[\w\s'.\-]`, max 100 caratteri; upload planimetria: max 10 MB, verificato come immagine decodificabile prima del salvataggio |
| **Chiave biometrica** | Errore esplicito all'avvio se `BIOMETRIC_SECRET_KEY` è vuota o è il placeholder |

---

## 7. Stack Tecnologico

| Componente | Tecnologia | Versione |
| --- | --- | --- |
| Face Detection | SCRFD (InsightFace) | buffalo_l |
| Face Recognition | ArcFace ResNet-100 | buffalo_l |
| Inference Runtime | ONNX Runtime | ≥1.18.0 |
| GPU Acceleration | CUDA | 12.x |
| Acquisizione Video | OpenCV | ≥4.8.0 |
| Geometria (omografia) | OpenCV `findHomography` | — |
| Web Framework | Flask | ≥3.0.0 |
| ORM | SQLAlchemy | ≥2.0.0 |
| Database | SQLite (locale) / PostgreSQL | — |
| Crittografia | cryptography (Fernet) | ≥41.0.0 |
| Configurazione | Pydantic Settings | ≥2.0.0 |
| Logging | Loguru | ≥0.7.0 |
| Runtime Python | Python | 3.10+ |

---

## 8. Conformità GDPR

I dati biometrici (embedding facciali) rientrano nella **categoria speciale** ai sensi dell'Art. 9 GDPR.

| Articolo GDPR | Requisito | Implementazione nel progetto |
| --- | --- | --- |
| Art. 5 — Minimizzazione | Solo dati strettamente necessari | Nessuna immagine salvata, solo vettori numerici cifrati |
| Art. 5 — Limitazione conservazione | Scadenza dati | `DATA_RETENTION_DAYS` + `run_retention()` automatico all'avvio |
| Art. 9 — Consenso esplicito | Per dati biometrici | `consent_given=True` prima di salvare qualsiasi template |
| Art. 17 — Diritto all'oblio | Cancellazione su richiesta | `DELETE /api/persons/{name}` + `delete_person.py` — elimina anche traiettorie (`CASCADE`) |
| Art. 25 — Privacy by design | Protezione by default | Embedding cifrati a riposo, server solo su localhost di default |
| Art. 32 — Sicurezza del trattamento | Misure tecniche | Fernet (AES-128-CBC + HMAC-SHA256), key derivation SHA-256, Basic Auth |

> **Nota sul tracking:** la registrazione della posizione nel tempo amplia il perimetro del trattamento rispetto al solo riconoscimento. Il consenso raccolto in fase di iscrizione deve coprire anche questa finalità; per deployment reali valutare una DPIA (Art. 35).

---

## 9. Configurazione e Deploy

### Variabili d'ambiente (`.env`)

| Variabile | Default | Descrizione |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL locale | Stringa connessione — per SQLite: `sqlite:///./face_id.db` |
| `BIOMETRIC_SECRET_KEY` | — | **Obbligatoria.** Chiave per la cifratura degli embedding |
| `WEB_PASSWORD` | *(vuota)* | Basic Auth Web UI — obbligatoria se esposta oltre localhost |
| `CAMERA_SOURCES` | `0` | Indici/URL camere separati da virgola (es. `0,1` o `rtsp://...`) |
| `MATCH_THRESHOLD` | `0.5` | Soglia cosine distance (più basso = più severo) |
| `DET_THRESHOLD` | `0.5` | Confidenza minima del rilevatore SCRFD (più alto = meno falsi rilevamenti) |
| `MIN_FACE_PX` | `80` | Scarta volti più bassi di N px (riflessi specchio, volti lontani) |
| `USE_GPU` | `true` | Usa CUDA se disponibile |
| `POSITION_LOG_INTERVAL` | `1.0` | Secondi tra i punti di posizione salvati (per persona/camera) |
| `DATA_RETENTION_DAYS` | `365` | Giorni di conservazione dati biometrici |
| `DATA_DIR` | `data` | Base per artefatti persistenti (DB, validation, benchmark, log) — `/data` in Docker |
| `METRICS_ENABLED` | `true` | Strumentazione timing della pipeline (vedi §11) |
| `TELEGRAM_ENABLED` | `false` | Abilita gli alert Telegram (vedi §13) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | — | Credenziali bot (da @BotFather / @userinfobot) |
| `UNKNOWN_ALERT_COOLDOWN` | `10.0` | Secondi minimi tra un alert sconosciuto e il successivo |
| `UNKNOWN_ALERT_MIN_DURATION` | `1.0` | Secondi di "sconosciuto" continuo prima di allertare |
| `UNKNOWN_ALERT_WARMUP` | `15.0` | Nessun alert nei primi N secondi (warm-up camera all'avvio) |
| `LOG_LEVEL` | `INFO` | Verbosità log (DEBUG/INFO/WARNING/ERROR) |

### Avvio

```bash
# Solo web UI (consigliato) — http://localhost:8000
python main.py --web

# Solo finestre OpenCV locali
python main.py

# Entrambe
python main.py --web --local

# Esposizione sulla LAN (richiede WEB_PASSWORD nel .env)
python main.py --web --host 0.0.0.0

# Iscrizione nuova persona da CLI
python scripts/enroll.py --name "Nome Cognome" --samples 5

# Cancellazione dati (GDPR Art. 17)
python scripts/delete_person.py --name "Nome Cognome"
```

---

## 10. Strumentazione delle Prestazioni

Strumentazione integrata per misurare FPS, latenza e uso risorse, visibile live nella dashboard. Disattivabile con `METRICS_ENABLED=false` (overhead trascurabile).

**`core/metrics.py`:**

- **`PerfTracker`** — finestra mobile (ultimi 60 frame) per camera: calcola FPS smussato e latenza media per stage. Thread-safe.
- **Provider risorse cross-platform** — sceglie il backend a runtime: **NVML** (`pynvml`) su x86+NVIDIA, **`tegrastats`** su Jetson (L4T), **`psutil`** come fallback. Restituisce GPU util/memoria/potenza/temperatura, CPU%, RAM; ogni campo degrada a `None` se non disponibile.

**Timing (`detector.py` + `pipeline.py`):** `detect_and_encode(timing=...)` separa il tempo di **detection** da quello di **embedding** (replicando `FaceAnalysis.get()`, con fallback difensivo). La pipeline misura detection/embedding/matching/end-to-end e il numero di volti per frame.

**Matcher ottimizzato (`recognizer.py`):** la matrice `(N×512)` dei template è precalcolata in `load()` e scambiata atomicamente — niente `np.stack` per frame. `benchmark_matching()` misura la latenza di `identify()` al variare della dimensione del DB (10…5000 template): resta sotto il millisecondo anche con migliaia di iscritti.

**Dashboard e API:**

- `GET /api/metrics` → JSON con FPS (per camera + aggregata), latenza per-stage e risorse.
- Pannello **Prestazioni** in `index.html`: card risorse, tabella per-camera, due grafici time-series (Chart.js) per FPS e latenza, polling 1s.
- **`scripts/benchmark.py`** (o `python main.py --benchmark`): carico controllato → report machine-readable in `data/benchmarks/<run_id>/` (`report.json` + `matching.csv`), etichettato con piattaforma e backend così gli stessi scenari sono confrontabili tra i target hardware (x86 CUDA, Jetson TX2, Orin).

---

## 11. Modalità di Validazione (esperimenti)

Strumento per **misurare l'accuratezza** del riconoscimento fornendo la ground truth: il sistema da solo non può sapere se un'identificazione è corretta. La verità a terra si stabilisce **dal vivo** (presentazione controllata: soggetto dichiarato per gli attraversamenti, mappa dei posti per la sessione comune) — modalità di **default, senza video** — oppure *a posteriori* rivedendo il filmato, se la registrazione video è esplicitamente attivata (fallback).

Il compito valutato è **identificazione open-set 1:N** (scenario watchlist / controllo accessi: una *galleria* di iscritti e dei *probe* che includono anche persone **non iscritte** che vanno rifiutate). Quindi le metriche primarie sono **FPIR** e **FNIR** (come in NIST FRVT/FRTE 1:N, valutazione di scenario ISO/IEC 19795), con curva DET ed EER — **non** metriche di verifica 1:1. FAR/FRR sono forniti solo come alias colloquiali. **Le metriche sono identiche** nelle due modalità: entrambe alimentano `detections.jsonl` + `labels.jsonl`.

> ⚠️ **Privacy:** di **default non viene salvata alcuna immagine** (`VALIDATION_RECORD_VIDEO=false`) → ripristino della postura «no images on disk». La registrazione del video annotato resta disponibile come **fallback opt-in** (toggle per-sessione o `.env`): in quel caso è l'unica funzione che salva immagini, deroga consapevole al design senza-immagini. Gli artefatti stanno sotto `data/validation/<id>/`, **fuori dal DB biometrico** e **mai versionati su git** (`data/` è in `.gitignore`). `scripts/clear_validation.py` li elimina dopo l'analisi. Dettagli e razionale: vedi **`validazione_senza_video_addendum.md`**.

**Registrazione (`core/validation.py` → `ValidationManager`):** una sessione nominata produce, per camera, il **video annotato** (`video/cam_<id>.mp4`) + **`detections.jsonl`** (un record per volto per frame: `timestamp_ms`, `raw_cosine_distance`, `best_match_person_id` e il **ranking completo dei candidati** `candidates`, così FPIR/FNIR/CMC sono ricalcolabili a qualsiasi soglia offline) + `session.json` (manifest con piattaforma auto-rilevata, galleria, soglia, camere). Allo start scrive anche **`PROTOCOL.md`**, un protocollo auto-documentante (tipo di test, schema dati, metriche, tassonomia verdetti). La pipeline registra ogni frame mentre la sessione è attiva (anche senza volti). Avvio/stop da UI o `python main.py --validation [NOME]`.

**Review UI (`/validation`):** layout split.

- **Sinistra — player a frame**: i frame JPEG sono estratti dall'mp4 on-demand (`/api/validation/<id>/frame/<cam>/<index>`) — evita il problema di codec (gli mp4 `mp4v` non sono riproducibili in `<video>`) e dà sync esatto col log. Navigazione con frecce/slider; ⏮⏭ saltano da un **evento** all'altro.
- **Destra — timeline eventi**: le detection sono raggruppate per **soggetto e tempo** (un evento = una comparsa continua), non JSON grezzo. Ogni riga mostra intervallo temporale, identità predetta, n frame, distanza media, identità vera e verdetto derivato.
- **Labeling per-evento**: l'operatore assegna la **identità vera** dell'evento (un iscritto della galleria, oppure *Non-mate*) — soglia-indipendente, così le metriche si ricalcolano a ogni soglia. Un click etichetta l'intero evento (bulk); tasti `1`–`9` = iscritti, `0` = Non-mate. Salvato in `labels.jsonl`. Il verdetto a soglia operativa è derivato per la visualizzazione.

**Tassonomia verdetti** (derivati da identità vera + distanze): `mate_correct` (Mate corretto, TP), `mate_miss` (Mate mancato → FNIR), `swap` (Scambio → FNIR, a parte), `false_positive` (Falso positivo → FPIR, critico), `non_mate_correct` (Non-mate corretto, TN).

**Metriche (`core/validation_metrics.py`):** unità di analisi = **evento/track** (i frame consecutivi dello stesso soggetto si aggregano per voto di maggioranza; il per-frame è riportato come secondario, con caveat di correlazione). Calcola:

- **FPIR** = eventi non-mate accettati / eventi non-mate; **FNIR** = eventi mate non identificati correttamente (miss + scambio) / eventi mate.
- **Curva DET** spazzando la soglia su tutte le `raw_cosine_distance` → `det_curve.csv`; **EER**; **FNIR a FPIR fisso** (1% e 0.3%, punti operativi di sicurezza).
- **Rank-1 e CMC** dai ranking dei candidati → `cmc.csv`.
- **Intervalli di confidenza Wilson** su FPIR/FNIR, **regola del 3** (limite ~3/N a zero errori) e **flag rule-of-30** di Doddington (<30 errori → stima imprecisa).
- **FAR/FRR** come alias etichettati. Breakdown **per-camera** con più camere. Export `metrics.json` + `det_curve.csv` + `cmc.csv`. Tutto riproducibile dai JSONL senza rieseguire le camere.

**Come migliora davvero la precisione:** la rete ArcFace è **pre-addestrata e congelata** (non si ri-addestra). Ma il risultato della validazione si applica al sistema: il bottone **"Applica soglia EER"** (`POST /api/settings/match_threshold`) imposta la `MATCH_THRESHOLD` ottimale **a caldo** sul recognizer e la **persiste nel `.env`**. La validazione serve anche a tarare `DET_THRESHOLD`/`MIN_FACE_PX` e a individuare iscrizioni deboli (da ri-fare).

**Layout dei file** prodotti per sessione:

```text
data/validation/<session_id>/
  PROTOCOL.md   session.json   detections.jsonl   labels.jsonl   runs.jsonl
  metrics.json  det_curve.csv  cmc.csv            video/cam_<id>_NNN.mp4   video/segments.json
```

I preset di condizione e il registro soggetti sono globali e riusabili: `data/validation_presets.json`, `data/validation_subjects.json`.

### 11.1 Setup esperimento (preset, soggetti, sessioni, run)

Per rendere le metriche *condition-aware* e la raccolta probe frictionless, la pagina `/validation` aggiunge un setup configurabile una volta e riusato:

- **Preset di condizione** (`core/validation_presets.py`): condizioni ambientali controllate di un attraversamento — posizione camera vs soggetto (frontale/laterale/angolata + angolo) e illuminazione (numero luci, disposizione, angolo) + note. CRUD da UI; i **parametri interi** vengono copiati nel `session.json` (record self-contained anche se il preset viene poi modificato).
- **Registro soggetti**: `S1…Sn` iscritti mappati a un `person_id` della galleria (mate); `U1…Un` sconosciuti **anonimi**, solo numero (non-mate — privacy). `subject_truth(label)` fornisce la verità a terra soglia-indipendente.
- **Tipi di sessione**: *attraversamento singolo-soggetto* (1 camera, 1 preset) e *sessione comune* (10 soggetti statici).
- **Run context + ground-truth automatica**: con `set_run_context(subject, preset)` (POST `/api/validation/run`) l'operatore dichiara chi sta attraversando e sotto quale condizione; ogni detection è taggata con `subject_label`/`preset_id` e — negli attraversamenti — riceve la **GT automatica** (S→mate `true_person_id`, U→`non_mate`): nessun labeling manuale. I cambi di run sono loggati in `runs.jsonl`. La **sessione comune** usa il labeling manuale esistente.
- **Wizard di avvio** (UI): destinazione → tipo+nome → soggetto (saltato per comune) → preset → conferma. Durante la registrazione un **summary live** mostra soggetto, condizione, destinazione, frame/segmenti e tempo, con cambio run a 2 tap.
- **Metriche per-condizione/per-soggetto**: `metrics.json` aggiunge `per_condition[preset_id]` e `per_subject[label]` (FPIR/FNIR/rank-1) per vedere come l'accuratezza varia tra le condizioni.

### 11.2 Disco esterno e filesystem

Gli artefatti (video, JSONL, CSV) possono andare su un **HDD/SSD esterno**: `VALIDATION_DIR` nel `.env` o il picker nel wizard (`core/storage.py`: rilevamento dischi via `psutil`, validazione scrivibilità/spazio/filesystem). **Vincoli gestiti:**

- Il **database SQLite resta sempre su storage interno** (ext4): FAT32/exFAT/NTFS hanno problemi di permessi/lock POSIX. Solo gli artefatti append-only vanno sull'esterno; l'app non monta dischi né installa driver (lavora con percorsi **già montati**).
- Il **video è registrato a segmenti** (`cam_<id>_NNN.mp4`, rotazione a ~3.5 GB o ~10 min) con indice `segments.json`: così il **limite 4 GB di FAT32** non viene mai raggiunto e la review scorre tra i segmenti (mapping frame-globale→segmento lato server).

**Montaggio host per piattaforma** (l'app usa il path già montato):

| FS | Linux/Jetson | Note |
| --- | --- | --- |
| **FAT32** (`vfat`) | nativo (`mount /dev/sdX1 /mnt/ext`) | limite 4 GB/file → gestito con i segmenti |
| **NTFS** | `ntfs-3g` (`apt install ntfs-3g`) | scritture sequenziali ok |
| **exFAT** | `exfat-fuse` + `exfatprogs` (`apt install exfat-fuse exfatprogs`) | su **Jetson TX2** (kernel vecchio) può servire il driver kernel exFAT abilitato / un rebuild; su JetPack recenti di norma ok |

**Docker:** il disco si monta **sull'host** e si fa **bind-mount** nel container (`-v /mnt/ext/faceid:/data/ext`), poi `VALIDATION_DIR=/data/ext`. L'app dentro il container scrive sul bind-mount; non gestisce mount/driver.

### 11.3 Modalità senza video (default) — ground truth dal vivo

La **modalità** è la **prima scelta del wizard** (step 1): *Mappatura (senza video)* o *Registrazione video (come prima)* — scelta esplicita obbligatoria che determina tutto il resto; il wizard è sequenziale (ogni step va completato prima del successivo). Di default `VALIDATION_RECORD_VIDEO=false`. In modalità mappatura una sessione **non registra video**: nessun byte di immagine tocca il disco, si scrivono solo i log testuali append-only. La verità a terra è stabilita **dal vivo**, per *presentazione controllata* (coerente con ISO/IEC 19795), in tre modi complementari:

- **Attraversamenti** — invariati: il soggetto dichiarato (`set_run_context`) dà la GT automatica per detection. Non serviva il video già prima.
- **Sessione comune — mappa dei posti** (`set_seating_map`, POST `/api/validation/seating`): l'operatore dichiara i posti occupati nell'ordine **sinistra→destra, fila per fila** → soggetto. A ogni frame le detection vengono **ordinate per posizione del bbox** (righe raggruppate per banda verticale con tolleranza ~0.6× l'altezza mediana del volto, poi sinistra→destra) e ciascuna riceve la GT del posto corrispondente, scritta in `detections.jsonl` come per gli attraversamenti. Deterministico, senza video, senza click. Se il numero di volti ≠ numero di posti, vengono etichettate solo le posizioni allineate (le eccedenze restano senza verità anziché essere assegnate male).
- **Correzione live click-to-assign** (`GET /api/validation/live/<cam>` + POST `/<id>/labels`): nel pannello live l'operatore clicca un volto sullo stream e assegna il soggetto; la label manuale finisce in `labels.jsonl` e **ha priorità** sulla GT automatica (a livello di frame). Il fotogramma è mostrato ma **mai salvato**. Per correzioni a livello di evento si usa il pannello di revisione (assegnazione bulk per evento), che funziona anche senza video.

`frame_index` è gestito da un **contatore per-camera del manager** (unica fonte di verità, con o senza video). Nel layout dei file, quando il video è spento la cartella `video/` non viene creata. La modalità è scelta da `record_video` in `POST /api/validation/start`; `session.json` registra `record_video` e `ground_truth_source`, e `PROTOCOL.md` si adatta di conseguenza.

> **Compromesso:** senza video **non c'è audit trail a posteriori** — non si può rivedere il filmato per contestare un'etichetta o ispezionare un errore. In contesti che richiedono quella verificabilità, attivare il video come fallback. Razionale completo: **`validazione_senza_video_addendum.md`**.

### 11.4 Profilo prestazioni (Standard vs Optimized-TX2) e confronto fasi

Una **singola immagine** espone due profili commutabili a runtime (`core/profile.py`, env `PERFORMANCE_PROFILE`, barra in alto nella dashboard + `POST /api/settings/profile`):

- **Standard** (Fase 1): comportamento **identico** all'attuale — `buffalo_l`, FP32/CUDA, risoluzione piena, nessun skip/tracker/batch.
- **Optimized-TX2** (Fase 2): FP16/**TensorRT** (engine costruito on-device, cache su `/data/engines`; INT8 opzionale), pack leggero `buffalo_s`, downsampling con remap dei bbox, frame-skip, **tracker IoU** (salta il re-embedding dei volti già identificati) e **batch embedding**. Ogni parametro è un `OPT_*` in `.env`.

Cambiare profilo ricostruisce l'analyzer InsightFace (nuovo pack/provider) **in background, senza riavvio e senza freeze** (`core/detector._rebuild_worker`): il vecchio analyzer continua a servire i frame finché il nuovo è pronto, poi swap atomico. Al cambio, la memoria nativa onnxruntime/TensorRT del vecchio analyzer viene **liberata** (`_dispose_analyzer` + `gc.collect()`) per non saturare gli 8 GB del TX2. Se su device la RAM non torna a baseline, `PROFILE_SWITCH_RESTART=true` fa ripartire il processo in modo pulito sul nuovo profilo (zero leak, ~secondi di gap feed). Il profilo attivo e i suoi parametri sono scritti in `session.json` (`profile`, `performance{…}`) e in `PROTOCOL.md`, e il **nome cartella** porta il suffisso `_standard` / `_optTX2`: sessioni delle due fasi restano separate e auto-descrittive su disco. Ogni run è una **cartella nuova** (`mkdir exist_ok=False`, suffisso `_2/_3…` su collisione) — niente è mai sovrascritto; se la destinazione esterna non è montata/scrivibile la sessione **non parte** (nessun fallback su eMMC).

> ⚠️ **Modello di riconoscimento per profilo (registrazione).** Standard usa `buffalo_l` (recogniser `w600k_r50`), Optimized di default `buffalo_s` (`w600k_mbf`): **spazi di embedding incompatibili**. I template volto sono perciò **taggati per modello** (`FaceTemplate.model_pack`) e il matching usa solo quelli del pack attivo → ogni profilo ha la **sua** registrazione (registra una volta per profilo; tornando a un profilo già registrato il riconoscimento funziona sempre). Per evitare la doppia registrazione: `OPT_MODEL_PACK=buffalo_l` nel `.env` → Optimized condivide il recogniser di Standard (embedding compatibili, una registrazione sola); resta comunque diverso per TensorRT fp16 + det downscale + frame-skip + tracker + batch (ma perde il confronto Fase-1/Fase-2 sul recogniser). **Migrazione obbligatoria** dopo l'aggiornamento su un DB già usato (template misti non taggati): `python scripts/wipe_templates.py` o `POST /api/persons/templates/wipe`, poi ri-registra.
>
> **Nota feed/risoluzione:** il video mostrato è sempre il **frame grezzo** della camera; `det_size` (Optimized 1280×720) imposta solo la risoluzione passata al **rilevatore**, non cambia il feed. Con tracker on l'identità è riportata tra frame (`embed_ms≈0`) senza ri-embedding. `pipeline._downsample` non fa mai upscaling (frame più piccolo di `det_size` → passa invariato).

**Confronto offline Fase 1 vs Fase 2** (nessuna camera): `core/compare.py` raggruppa le sessioni per profilo (opz. per preset), **ricalcola** FPIR/FNIR/EER/Rank-1 per gruppo dai JSONL salvati riusando `compute_session_metrics` (matematica invariata) ed emette il confronto + i delta in JSON/CSV. Da CLI `python scripts/compare_sessions.py [--by-preset] [--json out.json --csv out.csv]`, da web `GET /api/validation/compare` o il bottone **Confronta** in `/validation`.

> Deploy completo sul TX2 (immagine singola, disco esterno, profili): vedi il README §«Jetson TX2».

### 11.5 Wizard sessione, semaforo profilo e telemetria per-inferenza

**Semaforo profilo** (dashboard, config bar): pallino accanto ai bottoni Standard/Optimized-TX2 —
🔴 caricamento/switch in corso, 🟢 **verificato** (il pack/precisione REALMENTE caricati dal
build dell'analyzer coincidono col profilo richiesto; `GET /api/settings/profile/status`),
🟠 **mismatch** con la discrepanza mostrata (es. `.env` che punta entrambi i profili a buffalo_l —
incidente di laboratorio ora impossibile da non vedere). All'avvio `main.py` logga i valori
effettivi di `PERFORMANCE_PROFILE` e di tutti gli `OPT_*`.

**Wizard `/validation` (5 passi, sequenziale, back sempre attivo):** 1) preset a sinistra +
**destinazione disco obbligatoria** a destra (ultima scelta ricordata); 2) tipo **Singolo**
(attraversamenti, anche più utenti) / **Gruppo** + modalità mappatura/video; 3) **partecipanti dal
registro con nomi completi** ("S1 — Mario Rossi"), multi-selezione con spunte stabili +
"Sconosciuto"; 4) preset della sessione (default = ultimo usato); 5) riepilogo con **nome
auto-generato** `YYYYMMDD_HHMMSS_<Nomi>_<preset>_<profilo>_provaN` (N = contatore **globale** per
combinazione soggetti+preset+profilo, `trial_key`/`trial_n` in session.json) — l'operatore non
digita mai il nome → nome↔ground-truth non possono più divergere. Le sessioni finiscono
**automaticamente** nella cartella del giorno `validation/<YYYYMMDD>/` (niente più apri/chiudi
validazione); l'**Archivio** raggruppa per giorno (compat: layout flat e gruppi `val_*` legacy).

**Telemetria Tier A** (sempre attiva, overhead ~trascurabile, misurato dal benchmark): ogni record
di `detections.jsonl` porta `t_ms{pre,det,emb,match,total}`, `cycles`/`instructions` per
identificazione (perf_event via ctypes, `core/perfcounters.py`; se il kernel li nega —
`perf_event_paranoid`>2 nel container — servono `--cap-add PERFMON` o sysctl host, e il manifest
segna `perf_counters: unavailable`), `n_faces`, `det_input_wh`. A fine sessione `session.json`
aggiunge delta I/O disco (`/proc/self/io`), `mem_vmrss_kb` e latenza media/p95 delle append JSONL.

**Telemetria Tier B** (solo benchmark dedicati, MAI durante i test di accuratezza):
`scripts/benchmark_profili.py` (dentro il container) con `--preflight` (verifica dipendenze ARM
senza crash a metà run), stats mean/mediana/p95 per stadio, cicli/istruzioni mediani,
`--ort-profile` (profiler ONNX Runtime per-operatore su sessioni raw det+rec) e
`telemetry_overhead_pct`; `scripts/run_benchmark.sh` (host) cattura tegrastats a 5 Hz con **EMC%**
(memory controller — su TX2 CPU+GPU condividono la banda LPDDR4: EMC saturo a GPU scarica = collo
di bottiglia di memoria); `scripts/generate_report.py` produce `performance_report.md` con la
sezione intrusività. Opzionale host-side: `trtexec --loadEngine=... --dumpProfile` per i per-layer
TensorRT.

---

## 12. Bot Telegram

Notifiche per soggetti **sconosciuti** + iscrizione da messaggio. Layer di messaggistica **intercambiabile** (`Notifier` astratto → un backend WhatsApp si aggiungerebbe senza toccare la detection). Disattivo di default.

**`core/notifier.py`:** `TelegramNotifier` (pyTelegramBotAPI). Alert **solo testuali** (camera + ora + riferimento breve), **nessuna immagine** — coerente col design. Pulsanti inline **Autorizza/Nega**; su Autorizza il bot chiede il **nome**, poi iscrive (template cifrato + consenso) e chiama `force_reload()` → riconosciuto subito. Risponde solo al `chat_id` configurato.

**Qualità degli alert (tarata sul campo):**

- **Warm-up** (`UNKNOWN_ALERT_WARMUP=15s`): nessun alert nei primi secondi dall'avvio (la webcam regola l'esposizione → "sconosciuti" spuri).
- **Debounce con grace period** (`UNKNOWN_ALERT_MIN_DURATION=1s`): serve ≥1s di "sconosciuto" *continuo* prima di allertare, ma un singolo frame perso (flicker del rilevatore) non azzera il timer → un mezzo secondo di testa girata non fa scattare nulla.
- **Un alert alla volta**: finché non risolvi (Autorizza/Nega/nome) non ne arrivano altri (TTL 180s), più un `UNKNOWN_ALERT_COOLDOWN=10s`.
- **Embedding fresco**: al momento del nome il bot cattura l'embedding dal frame corrente (volto reale frontale) invece del template "vecchio" da quando eri sconosciuto → la camera passa da Sconosciuto al nome quasi subito.

La pipeline accumula gli embedding sconosciuti in un buffer per camera e invia l'alert in un **thread separato** (non blocca il loop di elaborazione).

---

## 13. Containerizzazione

Due Dockerfile, perché x86 e Jetson hanno esigenze opposte sul fronte CUDA/ONNX:

| Target | File | Base | CUDA/cuDNN + onnxruntime | Esecuzione |
| --- | --- | --- | --- | --- |
| x86-64 + CUDA 12 (RTX) | `Dockerfile` | `nvidia/cuda:12.4.1-cudnn-runtime` | dal base (CUDA+cuDNN) + `onnxruntime-gpu` pip; **multi-stage** | `--gpus all` / `docker compose` |
| Jetson Orin (L4T r36) | `Dockerfile.jetson` | `dustynv/onnxruntime:r36.x` | già nel base jetson-containers | `--runtime nvidia` |
| Jetson TX2 (L4T r32.7) | `Dockerfile.jetson` | `dustynv/onnxruntime:r32.7.1` | già nel base jetson-containers | `--runtime nvidia` |

- **x86 (`Dockerfile`, multi-stage):** stage `builder` con build-tools compila le dipendenze in un venv; lo stage finale copia solo il venv → niente build-tools nell'immagine. Il base **`cudnn-runtime`** è scelto perché il provider CUDA di onnxruntime si linka a *tutte* le librerie CUDA + cuDNN (un base senza cuDNN fa ripiegare silenziosamente su CPU). Immagine validata con inferenza GPU reale; pubblicata come **`t018/faceid:x86-cuda`** su Docker Hub. Compose con riserva GPU NVIDIA e volumi `faceid-data` (`/data`) e `faceid-models` (`~/.insightface`).
- **Jetson (`Dockerfile.jetson`):** parte da una base **jetson-containers** (`dustynv/onnxruntime:<L4T>`) che fornisce CUDA+cuDNN+onnxruntime+OpenCV compilati per la L4T del device (i wheel pip x86 non valgono su ARM/L4T); aggiunge solo app + dipendenze pure-Python. Va **costruito ed eseguito sul device** (`--runtime nvidia`). Per TX2 (Python 3.6) lo stack scientifico può richiedere una base più completa (`dustynv/l4t-ml`).
- **Multi-arch index:** `scripts/publish_manifest.sh` combina i tag per-arch già pushati in un image index (`docker buildx imagetools create`) → `docker pull t018/faceid` sceglie amd64 (PC) o arm64 (Orin) in automatico. Il TX2, anch'esso `linux/arm64`, non condivide lo slot con l'Orin: resta tag esplicito (`t018/faceid:jetson-tx2`).
- **Windows:** resta **installazione nativa** (massime prestazioni CUDA), non containerizzata.

I comandi build/run per ogni target sono nel [README](README.md#-installazione) e negli header di `Dockerfile` / `Dockerfile.jetson`.

---

## 14. Argomenti da Studiare per la Presentazione

### LIVELLO 1 — Fondamentali (obbligatori)

#### 1.1 Computer Vision di base

- **Cos'è un'immagine digitale**: matrice di pixel, canali BGR/RGB, risoluzione
- **OpenCV**: libreria per elaborazione immagini, `cv2.VideoCapture`, `cv2.imencode`
- **Frame rate (FPS)**: fotogrammi al secondo, perché è importante in real-time

**Domanda tipica:** *"Come funziona la cattura video?"*
→ `CameraStream` apre la webcam con OpenCV in un thread separato e legge frame in loop. Il thread è separato perché `read()` è bloccante.

---

#### 1.2 Face Detection

- **Cosa fa**: trova i rettangoli delimitatori (bounding box) dei volti in un'immagine
- **SCRFD**: detector convoluzionale efficiente (*Sample and Computation Redistribution for Face Detection*) usato dal pacchetto buffalo_l di InsightFace — rileva volti a scale diverse
- **Landmark facciali**: 5 punti chiave (occhi, naso, angoli bocca) usati per allineare il volto prima del riconoscimento

**Domanda tipica:** *"Come individua i volti nell'immagine?"*
→ InsightFace usa SCRFD, una rete neurale convoluzionale addestrata su milioni di volti. Output: bbox + 5 landmark per ogni volto trovato.

---

#### 1.3 Face Recognition con ArcFace

- **Embedding (o feature vector)**: vettore numerico che rappresenta le caratteristiche di un volto — nel progetto 512 dimensioni float32
- **ArcFace**: architettura ResNet-100 addestrata con *additive angular margin loss* per massimizzare la distanza tra classi diverse e minimizzarla tra campioni della stessa classe
- **Normalizzazione L2**: tutti gli embedding vengono proiettati sulla sfera unitaria (norma = 1) — e la **media** di più embedding va **rinormalizzata** (la media di vettori unitari è più corta di 1)
- **Perché 512 dimensioni?** Compromesso tra potere discriminativo e velocità di confronto

**Domanda tipica:** *"Come distingue una persona da un'altra?"*
→ ArcFace converte il volto in un vettore di 512 numeri. La "posizione" di questo vettore nello spazio 512D è unica per ogni persona. Confrontiamo le posizioni invece delle immagini.

---

#### 1.4 Cosine Similarity e Distance

- **Prodotto scalare normalizzato**: `similarity = a · b` (con `||a|| = ||b|| = 1`)
- **Cosine distance**: `d = 1 - similarity` → 0 = identici, 2 = opposti
- **Soglia (threshold)**: 0.5 significa "accetto come match se la distanza è < 0.5"
- **Perché cosine e non euclidea?** Gli embedding ArcFace sono ottimizzati per la distanza angolare

**Domanda tipica:** *"Come decide se è la stessa persona?"*
→ Calcola la cosine distance tra l'embedding del volto rilevato e tutti gli embedding nel database. Se la distanza minima è sotto la soglia 0.5, è un match.

---

#### 1.5 Localizzazione su mappa (tracking)

- **Omografia**: trasformazione proiettiva 3×3 tra due piani — qui dal piano del pavimento visto dalla camera alla planimetria; stimata da ≥4 corrispondenze con `cv2.findHomography`
- **Modello pinhole e distanza da dimensione nota**: un oggetto di dimensione fisica nota (il viso) appare grande in pixel in modo inversamente proporzionale alla distanza → `d = k / h_px`
- **Coordinate polari**: posizione = camera + distanza × direzione (angolo dal centro dell'inquadratura)
- **Minimi quadrati**: come si stimano heading/FOV/k da pochi campioni di riferimento
- **EMA (Exponential Moving Average)**: filtro che liscia il rumore delle misure (`s_t = α·x_t + (1−α)·s_{t−1}`)

**Domanda tipica:** *"Come fai a sapere dove si trova la persona nella stanza?"*
→ Se la camera vede il pavimento, un'omografia mappa i pixel a terra sulla planimetria. Se non lo vede (webcam da scrivania), si stima la distanza dalla dimensione del volto (modello pinhole) e l'angolo dalla posizione orizzontale nel frame: coordinate polari rispetto alla camera, calibrate mettendosi in 2-3 punti noti.

---

### LIVELLO 2 — Architettura Software

#### 2.1 Threading in Python

- **GIL (Global Interpreter Lock)**: Python esegue un thread Python alla volta, ma l'I/O e il codice C (OpenCV, NumPy) rilasciano il GIL
- **Thread per la camera**: necessario perché `VideoCapture.read()` blocca in attesa del frame
- **Lock e Queue**: `threading.Lock` per proteggere dati condivisi, `queue.Queue` per comunicazione thread-safe tra worker e Flask
- **Double-checked locking**: l'inizializzazione del modello InsightFace è protetta da lock per evitare doppia init da più camera-thread

**Domanda tipica:** *"Perché usi più thread?"*
→ La camera, la pipeline di riconoscimento e il server web devono girare in parallelo. Un singolo thread sarebbe sequenziale e il sistema andrebbe a 1-2 FPS.

---

#### 2.2 Flask e Protocolli Web

- **MJPEG**: protocollo multipart che invia frame JPEG in sequenza sulla stessa connessione HTTP. Il browser mostra "video" ma in realtà è una serie di immagini
- **SSE (Server-Sent Events)**: connessione HTTP long-lived, il server invia eventi JSON in push al browser. Più semplice di WebSocket per flussi unidirezionali
- **REST API**: endpoints `/api/persons`, `/api/enroll/*`, `/api/calibration/*` seguono architettura REST (GET, POST, DELETE)
- **Lazy encoding**: i frame vengono codificati in JPEG solo se c'è almeno un viewer collegato allo stream

---

#### 2.3 SQLAlchemy ORM

- **ORM (Object-Relational Mapping)**: mappa classi Python a tabelle SQL
- **Repository pattern**: tutta la logica SQL è in `repository.py`, i moduli business logic non scrivono mai SQL grezzo
- **Session e transaction**: `get_session()` è un context manager che fa commit automatico o rollback in caso di errore
- **Tipi nativi**: i valori NumPy (`numpy.int64`, `numpy.float32`) vanno convertiti in tipi Python nativi prima del salvataggio, altrimenti il driver SQLite li serializza come BLOB

---

#### 2.4 ONNX Runtime e GPU

- **ONNX (Open Neural Network Exchange)**: formato portabile per modelli di rete neurale
- **Execution Providers**: ONNX Runtime supporta diversi backend — `CUDAExecutionProvider` (NVIDIA GPU), `CPUExecutionProvider` (fallback)
- **Perché GPU è più veloce?** Migliaia di core paralleli per operazioni matriciali vs decine di core CPU

---

### LIVELLO 3 — Privacy e Sicurezza

#### 3.1 GDPR e Dati Biometrici

- **Articolo 9**: i dati biometrici sono "categoria speciale" — richiedono consenso esplicito
- **Articolo 17**: diritto all'oblio — l'utente può chiedere cancellazione completa dei propri dati
- **Privacy by design (Art. 25)**: la protezione deve essere integrata nel sistema, non aggiunta dopo

**Domanda tipica:** *"Questo sistema è legale?"*
→ Sì se: (1) c'è consenso esplicito per ogni persona, (2) i dati sono cifrati, (3) c'è un meccanismo di cancellazione, (4) i dati vengono eliminati dopo la scadenza configurata. Il tracking della posizione va dichiarato nella finalità del consenso.

---

#### 3.2 Crittografia Fernet

- **AES-128-CBC**: cifratura simmetrica a blocchi — lo stesso segreto cifra e decifra
- **HMAC-SHA256**: codice di autenticazione — garantisce che il ciphertext non sia stato manomesso
- **Fernet**: combina AES + HMAC in un formato standardizzato e sicuro
- **Key derivation**: la password (stringa) viene hashata con SHA-256 per ottenere una chiave di 32 byte uniforme

---

#### 3.3 Sicurezza Web

- **Basic Auth + timing attack**: il confronto della password usa `hmac.compare_digest` (tempo costante) per non rivelare informazioni dal tempo di risposta
- **CSRF e `Sec-Fetch-Site`**: i browser moderni dichiarano l'origine della richiesta in un header non falsificabile — le richieste mutanti cross-site vengono rifiutate
- **Security headers**: `nosniff` (no MIME-sniffing), `X-Frame-Options: DENY` (no clickjacking), `Referrer-Policy`

---

### LIVELLO 4 — Domande Avanzate Possibili

| Domanda | Risposta sintetica |
| --- | --- |
| *Quante persone può gestire?* | Praticamente illimitato. Il confronto è una moltiplicazione matrice-vettore O(N×512), scalabile a migliaia. |
| *Può essere ingannato con una foto?* | ArcFace è vulnerabile a spoofing 2D senza liveness detection. Il progetto non implementa anti-spoofing (fuori scope). |
| *Perché calcoli la media di 5 campioni?* | Un singolo embedding può essere rumoroso (posa, illuminazione). La media di 5 campioni, rinormalizzata L2, è più robusta. |
| *Cosa succede se due persone hanno facce simili?* | Il threshold 0.5 è calibrato per avere falsi positivi <1%. Può essere abbassato (più restrittivo) aumentando però i falsi negativi. |
| *Perché non salvare le immagini invece degli embedding?* | Gli embedding sono più leggeri (2 KB vs centinaia di KB), non permettono di ricostruire il volto originale e sono facili da cifrare. |
| *Cosa succede se si perde la chiave crittografica?* | Tutti gli embedding diventano inutilizzabili. Le persone devono essere re-iscritte. |
| *Perché usare InsightFace invece di face_recognition?* | InsightFace (ArcFace) è più accurato (errore ~0.1% su LFW vs ~0.5% di face_recognition/dlib), supporta GPU nativamente e ha modelli aggiornati. |
| *Quanto è precisa la posizione sulla mappa?* | A livello di zona, non di centimetri: il volto sta ~1.5 m sopra il piano modellato e la sua dimensione varia ±10% tra persone. Lo smoothing EMA riduce il jitter. |
| *Perché due modalità di calibrazione?* | L'omografia richiede punti sul pavimento: se la camera non lo inquadra è geometricamente inapplicabile. La modalità polare usa la dimensione del volto come riferimento metrico e funziona con qualunque inquadratura. |

---

### Schema Riassuntivo Studio

```text
FONDAMENTALI (studia per primo)
│
├── Come funziona un'immagine digitale
├── Cos'è una rete neurale convoluzionale (CNN) — concetto base
├── Face Detection (SCRFD, bounding box, landmark)
├── Face Embedding (ArcFace, vettore 512D, normalizzazione L2)
├── Cosine Similarity / Distance
├── Soglia di matching (threshold, false positive, false negative)
└── Localizzazione: omografia · pinhole/distanza dal viso · coordinate polari · EMA

ARCHITETTURA (studia dopo)
│
├── Threading Python (perché, Lock, Queue, double-checked locking)
├── Flask (HTTP, MJPEG, SSE, REST)
├── SQLAlchemy ORM (tabelle, sessioni, repository pattern)
└── ONNX Runtime + CUDA (execution providers, GPU speedup)

PRIVACY E SICUREZZA (studia per la parte legale)
│
├── GDPR Art. 9 (dati biometrici, consenso)
├── GDPR Art. 17 (diritto all'oblio)
├── GDPR Art. 25 (privacy by design)
├── Cifratura simmetrica (AES, chiave, ciphertext) + HMAC
└── Sicurezza web (Basic Auth, CSRF, security headers)
```

---

*Documentazione aggiornata il 18/06/2026.*
