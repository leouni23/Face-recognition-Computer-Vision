<div align="center">

# 🎯 Face ID — Riconoscimento Facciale in Tempo Reale

**Sistema multi-camera con tracking della posizione su planimetria, storage biometrico cifrato e conformità GDPR**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace_512d-FF6F00)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12-76B900?logo=nvidia&logoColor=white)
![DB](https://img.shields.io/badge/DB-SQLite_%7C_PostgreSQL-316192?logo=postgresql&logoColor=white)
![GDPR](https://img.shields.io/badge/GDPR-compliant-2E7D32)
![License](https://img.shields.io/badge/License-MIT-yellow)

[Funzionalità](#-funzionalità) •
[Quick Start](#-quick-start) •
[Architettura](#%EF%B8%8F-architettura) •
[Tracking & Mappa](#%EF%B8%8F-tracking-posizioni--mappa-live) •
[Sicurezza](#-sicurezza) •
[GDPR](#%EF%B8%8F-privacy-e-gdpr)

</div>

---

## ✨ Funzionalità

| | Funzionalità | Dettagli |
| --- | --- | --- |
| 🎥 | **Riconoscimento in tempo reale** | Una o più webcam locali o stream RTSP/RTMP, elaborazione in thread paralleli |
| 🧠 | **Identificazione biometrica** | Embedding ArcFace a **512 dimensioni** (InsightFace `buffalo_l`: detection SCRFD + recognition ResNet-100) |
| ⚡ | **Accelerazione GPU** | CUDA via `onnxruntime-gpu` con fallback automatico su CPU |
| 🗺️ | **Tracking posizioni su mappa** | Storico delle posizioni nel tempo e vista live dei soggetti sulla planimetria della stanza |
| 📐 | **Doppia calibrazione camera** | **Polare** (distanza dal viso — per camere che *non* vedono il pavimento) oppure **omografia** (per camere che lo vedono) |
| 🖥️ | **Web UI completa** | Stream MJPEG live, eventi SSE in tempo reale, enrollment dal browser, gestione persone |
| 🔐 | **Template cifrati** | AES-128-CBC + HMAC-SHA256 (Fernet) — i dati biometrici non toccano mai il disco in chiaro |
| 🛡️ | **Sicurezza Web UI** | Basic Auth opzionale, guard CSRF, security headers, validazione input |
| ⚖️ | **GDPR compliant** | Consenso esplicito, data retention automatica, diritto all'oblio (Art. 9, 17, 25) |

---

## 🚀 Quick Start

```bash
# 1 · Clona e installa
git clone https://github.com/leouni23/Face-recognition-Computer-Vision.git
cd Face-recognition-Computer-Vision
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2 · Configura
cp .env.example .env
#    → genera la chiave: python -c "import secrets; print(secrets.token_hex(32))"
#    → incollala in BIOMETRIC_SECRET_KEY nel file .env

# 3 · Inizializza il database e avvia
python scripts/init_db.py
python main.py --web
```

Apri **[http://localhost:8000](http://localhost:8000)** → registra un volto col pulsante **"Inizia registrazione"** → il riconoscimento parte subito.

<details>
<summary>⚡ <b>Supporto GPU NVIDIA (opzionale)</b></summary>

```bash
pip install onnxruntime-gpu
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
```

Imposta `USE_GPU=true` nel `.env`. Su Windows le DLL CUDA installate via pip vengono registrate automaticamente all'avvio. Speedup tipico: 5-10× rispetto alla CPU.

</details>

---

## 🏗️ Architettura

```mermaid
flowchart TB
    subgraph WT["🧵 Worker thread — uno per camera"]
        CAM["CameraStream<br/>webcam · RTSP/RTMP"] -->|frame BGR| DET["detector.py<br/>SCRFD + ArcFace 512-d"]
        DET -->|embedding| REC["recognizer.py<br/>cosine distance vettoriale"]
        REC -->|identità| TRK["tracking posizione<br/>polare / omografia + EMA"]
    end

    TRK -->|"eventi · posizioni"| DB[("🗄️ SQLite / PostgreSQL<br/>template cifrati Fernet")]
    WT -->|"frame + eventi"| BC["Broadcaster<br/>bridge thread-safe"]
    BC <--> WEB["Flask<br/>MJPEG · SSE · REST"]
    WEB <--> BR["🌐 Browser<br/>dashboard · mappa live · calibrazione"]
```

<details>
<summary>📁 <b>Struttura del progetto</b></summary>

```text
face_id/
├── config/
│   └── settings.py          # Pydantic settings da .env
├── core/
│   ├── camera.py            # Acquisizione threaded (webcam + RTSP, auto-reconnect)
│   ├── detector.py          # InsightFace buffalo_l: SCRFD detection + ArcFace 512-d
│   ├── recognizer.py        # Matrice template precalcolata, cosine distance
│   ├── geometry.py          # Omografia + calibrazione polare (distanza dal viso)
│   └── pipeline.py          # Orchestrazione: detect → identify → log + proiezione mappa
├── database/
│   ├── models.py            # Person, FaceTemplate, RecognitionEvent, PositionLog, CameraCalibration
│   ├── session.py           # Engine + context manager transazionale
│   └── repository.py        # CRUD, cifratura template, traiettorie, calibrazioni
├── privacy/
│   ├── crypto.py            # Fernet (AES-128-CBC + HMAC-SHA256) + guard chiave placeholder
│   └── retention.py         # GDPR Art. 5 — cancellazione automatica dati scaduti
├── ui/
│   └── display.py           # Finestre OpenCV locali + FPS counter
├── web/
│   ├── app.py               # Flask: stream, SSE, enrollment, mappa, calibrazione + auth/CSRF
│   ├── broadcaster.py       # Bridge thread-safe con conteggio viewer (lazy JPEG encode)
│   └── static/
│       ├── index.html       # Dashboard (Tailwind CSS + vanilla JS)
│       ├── calibrate.html   # Pagina di calibrazione con live feed
│       └── maps/            # Planimetria caricata
├── scripts/
│   ├── enroll.py            # Enrollment da terminale con consenso GDPR
│   ├── delete_person.py     # Cancellazione dati (GDPR Art. 17)
│   └── init_db.py           # Creazione tabelle (idempotente)
└── main.py                  # Entry point (--web / --local)
```

</details>

### Come avviene un riconoscimento

```mermaid
flowchart LR
    A["📷 frame"] --> B["SCRFD<br/>bbox + landmark"]
    B --> C["ArcFace<br/>embedding 512-d<br/>norma L2 = 1"]
    C --> D{"min distanza<br/>coseno < 0.5?"}
    D -->|sì| E["✅ identificato<br/>person_id + confidence"]
    D -->|no| F["❓ Sconosciuto"]
```

Il confronto con tutti gli N template registrati è **una singola moltiplicazione matrice-vettore** NumPy (`matrix @ embedding`): la matrice (N×512) è precalcolata al caricamento, quindi il costo per frame è O(N·512) — veloce anche con centinaia di persone.

### Come viene protetto un template biometrico

```mermaid
flowchart LR
    A["embedding 512-d<br/>(media di 5 campioni,<br/>rinormalizzata L2)"] --> B["serializzazione<br/>2048 byte float32"]
    B --> C["🔐 Fernet<br/>AES-128-CBC + HMAC-SHA256<br/>chiave da SHA-256(secret)"]
    C --> D[("DB: blob cifrato<br/>~2.3 KB")]
```

**Nessuna immagine viene mai salvata** — solo il vettore matematico, sempre cifrato. Ruotare `BIOMETRIC_SECRET_KEY` rende inutilizzabili tutti i template senza cancellare il database. Se la chiave è vuota o è il placeholder, il sistema **si rifiuta di partire**.

---

## 🗺️ Tracking posizioni & mappa live

Il sistema salva nel tempo la posizione di ogni soggetto **identificato** (cadenza configurabile, default 1 punto/s per persona per camera) e la proietta su una planimetria condivisa. Il re-identification tra camere è automatico: la stessa persona ha lo stesso `person_id` ovunque grazie all'embedding ArcFace.

<div align="center">
<img src="docs/floorplan-example.png" width="560" alt="Planimetria di esempio con posizione camera, campo visivo e soggetti tracciati" />
<br/><sub><i>Planimetria di esempio — i soggetti identificati appaiono come pallini verdi nella vista Mappa live.<br/>La propria planimetria si carica dalla pagina Calibra e resta solo in locale (la cartella è esclusa da git).</i></sub>
</div>

### Due modalità di calibrazione (pagina `/calibrate`, con live feed della camera)

| | 📐 Polare — *distanza dal viso* | 🔲 Omografia — *piano del pavimento* |
| --- | --- | --- |
| **Quando usarla** | La camera **non inquadra il pavimento** (es. webcam su scrivania) | La camera **vede il pavimento** (es. telecamera a soffitto) |
| **Come funziona** | Distanza dal modello pinhole `d = k / altezza_volto_px` + angolo dalla posizione orizzontale nel frame | Matrice 3×3 che mappa i pixel del pavimento sulla planimetria (`cv2.findHomography`) |
| **Calibrazione** | Clic sulla mappa dove sta la camera, poi **2-3 campioni**: ti metti in un punto noto e premi *Cattura* | **≥4 coppie di punti a terra** cliccate su snapshot e mappa |
| **Vincoli** | Campioni in zone diverse dell'inquadratura (sinistra/destra) | Tutti i punti devono stare sul pavimento |

Le posizioni sulla mappa sono lisciate con una **media mobile esponenziale** (la dimensione del volto in pixel è rumorosa frame a frame) e hanno precisione *a livello di zona* — adatta a "è vicino al letto / alla porta", non al centimetro.

> 💡 La modalità polare implementa il metodo documentato in letteratura per la stima monoculare della distanza dalla dimensione apparente del volto (dimensione fisica del viso ~costante tra adulti).

### Storico delle posizioni

Dal pannello **Persone iscritte → Storico**: traiettoria disegnata su canvas per ogni camera (gradiente blu→verde dal punto più vecchio al più recente), con intervallo selezionabile da 15 minuti a 24 ore.

---

## 🖥️ Utilizzo

```bash
python main.py --web               # Web UI su http://localhost:8000 (consigliato)
python main.py                     # finestre OpenCV locali
python main.py --web --local       # entrambe
```

| Flag | Default | Descrizione |
| --- | --- | --- |
| `--web` | — | Avvia la Web UI Flask |
| `--local` | — | Finestre OpenCV locali (Q per uscire) |
| `--host` | `127.0.0.1` | Usa `0.0.0.0` per esporre sulla LAN — **solo con `WEB_PASSWORD` impostata** |
| `--port` | `8000` | Porta del server web |

### La Web UI

- **📺 Stream live** — feed MJPEG con bounding box e nome in tempo reale
- **🔔 Riconoscimenti** — log eventi via Server-Sent Events, senza polling
- **➕ Registra nuovo volto** — enrollment guidato dal browser (5 campioni, barra di avanzamento)
- **🗺️ Mappa** — posizione live dei soggetti sulla planimetria
- **📐 Calibra** — pagina di calibrazione camera con video live e flusso guidato
- **👥 Persone** — storico posizioni e cancellazione GDPR per ogni persona

<details>
<summary>⌨️ <b>Strumenti da terminale</b></summary>

```bash
# Enrollment con informativa e consenso GDPR (SPAZIO per catturare i campioni)
python scripts/enroll.py --name "Mario Rossi" [--samples 5] [--camera 0]

# Diritto all'oblio (GDPR Art. 17) — elimina template, eventi e traiettorie
python scripts/delete_person.py --name "Mario Rossi"
```

</details>

<details>
<summary>📡 <b>Sorgenti RTSP/RTMP</b></summary>

```env
# Camera IP singola
CAMERA_SOURCES=rtsp://admin:password@192.168.1.100:554/stream

# Mix webcam locale + IP camera
CAMERA_SOURCES=0,rtsp://admin:password@192.168.1.100:554/stream
```

Backend FFMPEG con buffer di 1 frame per latenza minima e riconnessione automatica in caso di dropout (fino a 30 tentativi, ritardo di 2 s tra l'uno e l'altro).

</details>

<details>
<summary>🔌 <b>API REST principali</b></summary>

| Endpoint | Metodo | Descrizione |
| --- | --- | --- |
| `/stream/<camera>` | GET | Stream MJPEG live |
| `/api/events` | GET | Eventi riconoscimento (SSE) |
| `/api/persons` | GET | Persone iscritte |
| `/api/persons/<nome>` | DELETE | Cancellazione GDPR |
| `/api/persons/<id>/trajectory?minutes=N` | GET | Storico posizioni (pixel + mappa) |
| `/api/positions/map?minutes=N` | GET | Ultima posizione sulla mappa per persona |
| `/api/enroll/start · capture · save · cancel` | POST | Enrollment in 3 step dal browser |
| `/api/map` | GET/POST | Planimetria (lettura / upload) |
| `/api/calibration/<camera>` | POST | Salva calibrazione (polare od omografia) |
| `/api/calibration/<camera>/sample` | POST | Cattura campione per calibrazione polare |

</details>

---

## ⚙️ Configurazione

Tutte le variabili si impostano nel file `.env` (vedi `.env.example`):

| Variabile | Default | Descrizione |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL locale | Stringa di connessione — per SQLite: `sqlite:///./face_id.db` |
| `BIOMETRIC_SECRET_KEY` | — | **Obbligatoria.** Chiave di cifratura dei template biometrici |
| `WEB_PASSWORD` | *(vuota)* | Basic Auth della Web UI — **obbligatoria se esposta oltre localhost** |
| `CAMERA_SOURCES` | `0` | Indici o URL RTSP separati da virgola |
| `MATCH_THRESHOLD` | `0.5` | Soglia cosine distance (più basso = più severo) |
| `USE_GPU` | `true` | CUDA se disponibile, altrimenti fallback CPU |
| `POSITION_LOG_INTERVAL` | `1.0` | Secondi tra i punti di posizione salvati (per persona/camera) |
| `DATA_RETENTION_DAYS` | `365` | Retention GDPR — cancellazione automatica all'avvio |
| `LOG_LEVEL` | `INFO` | Verbosità log |

---

## 🔒 Sicurezza

| Misura | Implementazione |
| --- | --- |
| **Autenticazione** | HTTP Basic Auth opzionale (`WEB_PASSWORD`), confronto in tempo costante (`hmac.compare_digest`) |
| **Default sicuro** | Server in ascolto solo su `127.0.0.1`; warning esplicito se esposto senza password |
| **CSRF** | Richieste mutanti cross-site bloccate via header `Sec-Fetch-Site` |
| **Security headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` |
| **Validazione input** | Nomi persona validati (regex, max 100 caratteri), upload mappa max 10 MB e verificato come immagine |
| **Chiave biometrica** | Avvio rifiutato se `BIOMETRIC_SECRET_KEY` è vuota o è il placeholder |

---

## ⚖️ Privacy e GDPR

I dati biometrici sono **categoria speciale** ai sensi dell'Art. 9 GDPR.

| Requisito | Implementazione |
| --- | --- |
| **Art. 5** — Minimizzazione | Solo embedding numerici cifrati, nessuna immagine salvata |
| **Art. 5** — Limitazione conservazione | `DATA_RETENTION_DAYS` con cancellazione automatica all'avvio |
| **Art. 9** — Consenso esplicito | Richiesto prima di ogni iscrizione (Web UI e CLI) |
| **Art. 17** — Diritto all'oblio | Cancellazione completa (template + eventi + traiettorie, `ON DELETE CASCADE`) |
| **Art. 25** — Privacy by design | Cifratura a riposo di default, localhost di default |
| **Art. 32** — Sicurezza del trattamento | Fernet (AES-128-CBC + HMAC-SHA256), derivazione chiave SHA-256 |

> ⚠️ Per un deployment in produzione valutare una **DPIA** (Data Protection Impact Assessment) ai sensi dell'Art. 35 GDPR. Il tracking della posizione amplia il perimetro del trattamento: assicurarsi che il consenso raccolto lo copra.

---

## 🧰 Stack tecnologico

| Libreria | Ruolo |
| --- | --- |
| `insightface` (`buffalo_l`) | Detection SCRFD + embedding ArcFace 512-d |
| `onnxruntime` / `onnxruntime-gpu` | Inference su CPU o CUDA |
| `opencv-python` | Acquisizione video, annotazione, omografia |
| `flask` | Web server: MJPEG, SSE, REST |
| `SQLAlchemy 2` | ORM (SQLite o PostgreSQL via `psycopg2`) |
| `cryptography` (Fernet) | Cifratura template biometrici |
| `pydantic-settings` | Configurazione type-safe da `.env` |
| `numpy` | Algebra vettoriale (cosine distance, calibrazione) |
| `loguru` | Logging strutturato |

---

## 📚 Documentazione

La documentazione tecnica completa in italiano — architettura interna, spiegazione degli algoritmi, threading, protocolli web e guida allo studio — è in **[DOCUMENTAZIONE.md](DOCUMENTAZIONE.md)**.

---

## 📄 Licenza

MIT — vedi [LICENSE](LICENSE)
