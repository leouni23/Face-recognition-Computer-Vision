# Face Recognition – Computer Vision

Sistema di riconoscimento facciale in tempo reale, multi-camera, con storage biometrico cifrato e conforme al GDPR (Reg. UE 2016/679).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace_512d-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![SQLite / PostgreSQL](https://img.shields.io/badge/DB-SQLite%20%7C%20PostgreSQL-316192?logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Funzionalità

- **Riconoscimento in tempo reale** da una o più webcam locali o stream RTSP/RTMP
- **Identificazione biometrica** tramite embedding ArcFace a **512 dimensioni** (InsightFace `buffalo_l`)
- **Confronto vettoriale ottimizzato** — cosine distance con NumPy vettorializzato, O(N) puro
- **Enrollment via Web UI** — iscrizione nuova persona direttamente dal browser, senza script
- **Web UI** con streaming MJPEG live, log eventi SSE e gestione persone
- **Template cifrati** — i dati biometrici sono salvati con AES-128-CBC + HMAC-SHA256 (Fernet), mai in chiaro
- **Accelerazione GPU** — CUDA via `onnxruntime-gpu`, fallback automatico su CPU
- **RTSP auto-reconnect** con backoff esponenziale
- **GDPR compliant** — consenso esplicito, data retention automatica, diritto all'oblio

---

## Architettura del progetto

```text
face_id/
├── config/
│   └── settings.py          # Pydantic settings da .env
├── core/
│   ├── camera.py            # Acquisizione threaded (webcam + RTSP/RTMP)
│   ├── detector.py          # InsightFace ArcFace 512-d (GPU/CPU)
│   ├── recognizer.py        # Cosine distance vettoriale (FaceRecognizer)
│   └── pipeline.py          # FaceIdPipeline: loop per camera + throttling DB
├── database/
│   ├── models.py            # SQLAlchemy: Person, FaceTemplate, RecognitionEvent
│   ├── session.py           # Engine + context manager
│   └── repository.py        # CRUD + encrypt/decrypt template
├── privacy/
│   ├── crypto.py            # Fernet encrypt/decrypt embedding
│   └── retention.py         # GDPR Art. 5 – cancellazione dati scaduti
├── ui/
│   └── display.py           # Annotazione frame OpenCV (finestre locali)
├── web/
│   ├── app.py               # Flask: MJPEG, SSE, REST API enrollment
│   ├── broadcaster.py       # Bridge thread-safe → Flask (frame + eventi)
│   └── static/index.html   # Web UI (Tailwind CSS, vanilla JS)
├── scripts/
│   ├── enroll.py            # Enrollment da terminale con consenso GDPR
│   ├── delete_person.py     # Cancellazione dati (GDPR Art. 17)
│   └── init_db.py           # Creazione tabelle
└── main.py                  # Entry point (locale e/o web)
```

---

## Flusso dati completo

```text
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          THREAD WORKER (per camera)                     │
  │                                                                         │
  │  CameraStream ──► frame BGR                                             │
  │       │                │                                                │
  │       │      ┌─────────▼──────────┐                                    │
  │       │      │  FaceIdPipeline    │                                     │
  │       │      │  .process_frame()  │                                     │
  │       │      │                    │                                     │
  │       │      │  detect_and_encode │ ← InsightFace ArcFace 512-d         │
  │       │      │  FaceRecognizer    │ ← cosine distance in RAM            │
  │       │      │  .identify()       │                                     │
  │       │      │                    │                                     │
  │       │      │  DB: log_event()   │ ← throttle 10 s/persona            │
  │       │      │       last_seen()  │                                     │
  │       │      └─────────┬──────────┘                                     │
  │       │                │ results [ (loc, pid, name, conf), … ]          │
  │       │                │                                                │
  │       │      ┌─────────▼──────────┐                                    │
  │       │      │  annotate_frame()  │ ← bounding box + nome + conf %     │
  │       │      └─────────┬──────────┘                                    │
  │       │                │                                                │
  │       └────────────────┤                                                │
  │             raw frame  │  annotated frame  +  eventi (pid ≠ None)      │
  └────────────────────────┼────────────────────────────────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │       Broadcaster        │  (thread-safe, singleton)
              │                          │
              │  _raw_frames[cam_id]     │ ← usato dall'enrollment web
              │  _frames[cam_id]  JPEG   │ ← stream MJPEG
              │  _subscribers[]  queue   │ ← eventi SSE
              └────────────┬─────────────┘
                           │
           ┌───────────────┼────────────────────┐
           │               │                    │
  ┌────────▼────────┐  ┌───▼──────────────┐  ┌──▼──────────────────┐
  │  GET /stream/   │  │  GET /api/events │  │  POST /api/enroll/* │
  │  MJPEG live     │  │  SSE eventi      │  │  enrollment web     │
  └────────┬────────┘  └───┬──────────────┘  └──┬──────────────────┘
           │               │                    │
           └───────────────┴────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Browser /  │
                    │  Web UI     │
                    └─────────────┘

  ┌────────────────────────┐
  │  Display locale        │  (solo se --local)
  │  OpenCV windows        │ ← result_queue dalla pipeline
  └────────────────────────┘
```

---

## Come funziona internamente

### 1 · Dal volto al vettore matematico

Ogni frame acquisito dalla camera passa alla rete neurale **ArcFace** (modello `buffalo_l`):

```text
┌─────────────────────────────────────────────────────────────┐
│                     FRAME BGR (OpenCV)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │   RetinaFace    │  rilevamento volti
                  │  (detection)    │  → bounding box (x1,y1,x2,y2)
                  └────────┬────────┘
                           │  crop + allineamento landmarks
                  ┌────────▼────────┐
                  │    ArcFace      │  rete ResNet-50
                  │ (recognition)   │  addestrata su milioni di volti
                  └────────┬────────┘
                           │
              embedding normalizzato L2
         [ 0.12, -0.87, 0.34, … ]  ← 512 × float32 = 2 048 byte
```

L'embedding è una "firma matematica" stabile del volto: invariante a variazioni di luce e posa. Dal vettore non è possibile ricostruire il volto originale.

---

### 2 · Cifratura e salvataggio nel database

```text
  embedding numpy (512 × float32)
           │
           │  .tobytes()              2 048 byte grezzi
           ▼
  [ b'\x3f\x8c\x1a...' ]             ancora in chiaro
           │
           │  BIOMETRIC_SECRET_KEY
           │  → SHA-256 → chiave 32 byte
           │  → Fernet(key).encrypt()
           │     AES-128-CBC   cifratura
           │     HMAC-SHA256   verifica integrità
           ▼
  [ b'gAAAABm...' ]                   blob cifrato, ~2 350 byte
           │
           │  INSERT INTO face_templates
           ▼
  ┌──────────────────────────────────────────────────────┐
  │  face_templates                                      │
  │  ┌──────┬───────────┬────────────────────────────┐  │
  │  │  id  │ person_id │   encoding_encrypted       │  │
  │  ├──────┼───────────┼────────────────────────────┤  │
  │  │   1  │     7     │  gAAAABm... (LargeBinary)  │  │
  │  └──────┴───────────┴────────────────────────────┘  │
  └──────────────────────────────────────────────────────┘
```

Senza `BIOMETRIC_SECRET_KEY`, il blob è inutilizzabile. Nessuna immagine viene mai salvata.

---

### 3 · Enrollment via Web UI

Il metodo principale di iscrizione avviene interamente dal browser in tre chiamate REST:

```text
  Browser
     │
     │  POST /api/enroll/start   { name, samples }
     │  ◄─────────────────────── { message, required }
     │
     │  (ripetuto N volte — utente preme il pulsante)
     │
     │  POST /api/enroll/capture
     │         │
     │         │  broadcaster.get_raw_frame(camera_id)
     │         │       ↑
     │         │  frame raw (last BGR dalla camera live)
     │         │
     │         │  detect_and_encode(frame)
     │         │  → InsightFace → embedding 512-d
     │         │
     │         │  _enroll_session["embeddings"].append(embedding)
     │         │
     │  ◄─────────────────────── { collected, required, done }
     │
     │  POST /api/enroll/save
     │         │
     │         │  mean_embedding = np.mean(embeddings, axis=0)
     │         │
     │         │  encrypt_embedding(mean_embedding, SECRET_KEY)
     │         │  → blob Fernet
     │         │
     │         │  INSERT persons + face_templates
     │         │
     │         │  broadcaster.pipeline.force_reload()
     │         │  → FaceIdPipeline ricarica i template dal DB
     │         │     senza aspettare i 60 s di polling
     │         │
     │  ◄─────────────────────── { message, samples }
```

La sessione di enrollment è protetta da un lock (`_enroll_lock`) — una sola iscrizione alla volta.

---

### 4 · Enrollment da terminale (alternativo)

```bash
python scripts/enroll.py --name "Mario Rossi" [--samples 5] [--camera 0]
```

```text
  Informativa GDPR → consenso s/N
           │
           │  webcam live — premi SPAZIO × N
           ▼
  5 embedding raccolti
           │
           │  np.mean(embeddings, axis=0)   ← media riduce il rumore
           ▼
  encrypt_embedding()  →  INSERT DB
```

---

### 5 · Riconoscimento in tempo reale

```text
  All'avvio (e ogni 60 s):
  DB → decrypt_embedding() × N → matrix (N × 512) in RAM

  Per ogni frame:
  embedding_query (512-d)
           │
           │  distances = 1.0 − (matrix @ embedding_query)
           │                      ↑
           │              dot product = cosine similarity
           │              (embedding già L2-normalizzati)
           │
           │  best_dist = min(distances)
           │  confidence = 1.0 − best_dist
           │
           ├── best_dist < soglia (0.5)  →  RICONOSCIUTO  (person_id, name)
           └── best_dist ≥ soglia        →  "Sconosciuto"
```

Il confronto è una singola moltiplicazione matrice-vettore NumPy — veloce anche con centinaia di persone registrate.

---

### 6 · Sicurezza e privacy dei dati

```text
  ╔══════════════════════════════════════════════════════════════════╗
  ║  NEL DATABASE                      NON NEL DATABASE             ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  ✓ blob cifrato (AES+HMAC)         ✗ immagini o foto           ║
  ║  ✓ nome persona                    ✗ embedding in chiaro        ║
  ║  ✓ timestamp iscrizione / accesso  ✗ dati biometrici leggibili  ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  PER USARE I DATI SERVONO ENTRAMBI:                              ║
  ║    • accesso al database                                         ║
  ║    • BIOMETRIC_SECRET_KEY (env var)                              ║
  ║                                                                  ║
  ║  Ruotare la chiave → tutti i template esistenti                  ║
  ║  diventano inutilizzabili, senza cancellare il DB                ║
  ╚══════════════════════════════════════════════════════════════════╝
```

---

## Requisiti

| Componente | Versione minima |
| --- | --- |
| Python | 3.10+ |
| PostgreSQL (opzionale) | 14+ |
| NVIDIA GPU (opzionale) | CUDA 12 + cuDNN 9 |
| Webcam o stream RTSP | — |

> SQLite è il default — non richiede installazione separata.

---

## Installazione

### 1. Clona la repository

```bash
git clone https://github.com/leouni23/Face-recognition-Computer-Vision.git
cd Face-recognition-Computer-Vision
```

### 2. Crea il virtual environment e installa le dipendenze

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Per il supporto GPU (opzionale):

```bash
pip install onnxruntime-gpu
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
```

### 3. Configura le variabili d'ambiente

```bash
cp .env.example .env
```

Modifica `.env`:

```env
# Database (default PostgreSQL — usa sqlite:///./face_id.db per SQLite)
DATABASE_URL=postgresql://user:password@localhost:5432/face_id

# Chiave cifratura template biometrici (genera una nuova!)
BIOMETRIC_SECRET_KEY=<genera con: python -c "import secrets; print(secrets.token_hex(32))">

# Accelerazione GPU
USE_GPU=false

# Sorgenti camera (indice locale o URL RTSP)
CAMERA_SOURCES=0

# Soglia riconoscimento (cosine distance — più basso = più severo)
MATCH_THRESHOLD=0.5

# Retention GDPR in giorni
DATA_RETENTION_DAYS=365
```

### 4. Inizializza il database

```bash
python scripts/init_db.py
```

---

## Utilizzo

### Avvia il sistema

#### Web UI (enrollment + stream live dal browser)

```bash
python main.py --web
```

Apri **[http://localhost:8000](http://localhost:8000)** per:

- guardare lo stream live con riconoscimenti in overlay
- iscrivere nuove persone cliccando "Enroll" (senza usare script)
- vedere il log eventi in tempo reale
- cancellare persone (GDPR Art. 17)

#### Finestre locali OpenCV

```bash
python main.py
```

#### Entrambe le modalità

```bash
python main.py --web --local
```

| Flag | Descrizione |
| --- | --- |
| `--web` | Avvia Flask (MJPEG + SSE + enrollment API) |
| `--local` | Mostra finestre OpenCV locali |
| `--port 8080` | Porta del server web (default: 8000) |
| `--host 0.0.0.0` | Host del server web |

Premi **Q** nella finestra OpenCV o **Ctrl-C** nel terminale per fermare.

---

### Enrollment da terminale (alternativo alla Web UI)

```bash
python scripts/enroll.py --name "Mario Rossi" [--samples 5] [--camera 0]
```

Mostra un'informativa GDPR, chiede il consenso, poi acquisisce N campioni da webcam premendo **SPAZIO**.

---

### Cancella i dati di una persona (GDPR Art. 17)

Via Web UI: pulsante "Elimina" nella lista persone.

Via terminale:

```bash
python scripts/delete_person.py --name "Mario Rossi"
```

---

## Privacy e GDPR

| Requisito GDPR | Implementazione |
| --- | --- |
| **Art. 5** – Minimizzazione dei dati | Solo embedding numerici cifrati, nessuna immagine salvata |
| **Art. 5** – Limitazione della conservazione | `DATA_RETENTION_DAYS` — cancellazione automatica all'avvio |
| **Art. 9** – Dati biometrici | Consenso esplicito richiesto prima dell'iscrizione |
| **Art. 17** – Diritto all'oblio | Web UI + `scripts/delete_person.py` cancellano tutti i dati |
| **Art. 25** – Privacy by design | Template cifrati con Fernet (AES-128-CBC + HMAC-SHA256) |
| **Invalidazione template** | Ruotare `BIOMETRIC_SECRET_KEY` rende inutilizzabili tutti i template |

> Per un deployment in produzione valutare una DPIA ai sensi dell'Art. 35 GDPR.

---

## Supporto RTSP

```env
CAMERA_SOURCES=rtsp://admin:password@192.168.1.100:554/stream

# Mix webcam locale + RTSP
CAMERA_SOURCES=0,rtsp://admin:password@192.168.1.100:554/stream
```

Il sistema usa buffer = 1 frame per latenza minima e auto-reconnect con backoff in caso di dropout.

---

## Stack tecnologico

| Libreria | Utilizzo |
| --- | --- |
| `insightface` (`buffalo_l`) | Embedding ArcFace 512-d (detection + recognition) |
| `onnxruntime` / `onnxruntime-gpu` | Inference ONNX su CPU o CUDA |
| `opencv-python` | Acquisizione video, annotazione frame |
| `flask` | Web server, MJPEG, SSE, REST API enrollment |
| `SQLAlchemy` + `psycopg2` / `aiosqlite` | ORM + driver PostgreSQL o SQLite |
| `cryptography` (Fernet) | Cifratura AES-128-CBC + HMAC-SHA256 |
| `pydantic-settings` | Configurazione type-safe da `.env` |
| `loguru` | Logging strutturato |
| `numpy` | Algebra lineare vettorializzata (cosine distance) |

---

## Licenza

MIT — vedi [LICENSE](LICENSE)
