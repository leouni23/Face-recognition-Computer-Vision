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
[Installazione](#-installazione) •
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

## 🚀 Installazione

Scegli la tua piattaforma. La **chiave di cifratura** è obbligatoria su tutte: generala con
`python -c "import secrets; print(secrets.token_hex(32))"`.

| Piattaforma | Metodo consigliato | GPU |
| --- | --- | --- |
| 🪟 **Windows x86-64** | **Nativo** (massime prestazioni) | CUDA via pip |
| 🐧 **Linux x86-64 + NVIDIA** | **Docker** (CUDA come nativo) | `--gpus all` |
| 🤖 **Jetson TX2 / Orin Nano** | **Docker, build sul device** | `--runtime nvidia` |

---

### 🪟 Windows — installazione nativa (consigliata)

Su Windows l'installazione nativa offre le **massime prestazioni CUDA**, senza overhead di virtualizzazione.

```powershell
git clone https://github.com/leouni23/Face-recognition-Computer-Vision.git
cd Face-recognition-Computer-Vision
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# GPU NVIDIA (opzionale ma consigliato) — le DLL CUDA via pip sono registrate all'avvio
pip install onnxruntime-gpu nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12

copy .env.example .env
#  → incolla la chiave generata in BIOMETRIC_SECRET_KEY, e per SQLite imposta:
#    DATABASE_URL=sqlite:///./face_id.db

python scripts\init_db.py
python main.py --web
```

Apri **[http://localhost:8000](http://localhost:8000)** → **"Inizia registrazione"** per iscrivere un volto.

---

### 🐧 Linux x86-64 + NVIDIA — Docker

Con il **NVIDIA Container Toolkit** la GPU passa al container con overhead trascurabile: l'accelerazione CUDA (cuDNN incluso) rende come in nativo — **verificato** con inferenza reale su GPU.

**Prerequisiti:** Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Su Windows è possibile anche via Docker Desktop + WSL2 + toolkit.

**Opzione A — immagine pronta (la più rapida, nessuna build):**

```bash
docker run --gpus all -p 8000:8000 \
  -e BIOMETRIC_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
  -v faceid-data:/data -v faceid-models:/home/faceid/.insightface \
  --device /dev/video0 t018/faceid:x86-cuda
```

L'immagine è su Docker Hub: **[hub.docker.com/r/t018/faceid](https://hub.docker.com/r/t018/faceid)**.

**Opzione B — build con docker compose** (dal sorgente):

```bash
git clone https://github.com/leouni23/Face-recognition-Computer-Vision.git
cd Face-recognition-Computer-Vision
BIOMETRIC_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
  docker compose up --build
```

La Web UI è su **[http://localhost:8000](http://localhost:8000)**. DB SQLite, modelli InsightFace e dati persistono nei volumi `faceid-data` / `faceid-models`. Per una **webcam USB** usa `--device /dev/video0` (o scommentalo in `docker-compose.yml`); per **RTSP** imposta `CAMERA_SOURCES`; esponendo oltre il proprio host imposta `WEB_PASSWORD`.

---

### 🤖 Jetson TX2 — immagine unica + profilo prestazioni (build **sul device**)

> ⚠️ **Si costruisce ed esegue SUL Jetson.** Le librerie CUDA/cuDNN/TensorRT arrivano dall'host col runtime NVIDIA (`--runtime nvidia`) — **non** sono nell'immagine. Il cross-build su x86 (buildx/QEMU) non vale: builda sul device.

**Una sola immagine** ([`Dockerfile.jetson`](Dockerfile.jetson)) contiene **due profili** commutabili a runtime dalla dashboard (barra in alto) o da `PERFORMANCE_PROFILE`:

| Profilo | Cosa fa |
| --- | --- |
| **Standard** (Fase 1) | modelli e pipeline attuali, invariati (`buffalo_l`, FP32/CUDA). |
| **Optimized-TX2** (Fase 2) | FP16/**TensorRT** (engine on-device, cache su `/data/engines`), pack leggero `buffalo_s`, downsampling, frame-skip, tracker, batch-embedding. |

Base **`dustynv/onnxruntime:r32.7.1`**: porta già, compilati per L4T r32.7, **CUDA + cuDNN + onnxruntime-gpu + numpy + OpenCV** (gli unici build ARM/L4T che funzionano — i wheel pip x86 no). L'immagine aggiunge solo il codice e le dipendenze pure-Python di `requirements-jetson.txt` (pin Python 3.6). **Modelli ed engine NON inclusi** (scaricati/costruiti al primo avvio su `/data`). Tutti i dati e le cache stanno sul **disco esterno** montato su `/data`; sull'eMMC restano solo OS e immagine.

**1 · Prerequisiti.** JetPack 4.6 (L4T r32.7); runtime NVIDIA per Docker (`/etc/docker/daemon.json` → `"default-runtime": "nvidia"`, poi `sudo systemctl restart docker`); massime prestazioni:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

**2 · Monta il disco esterno** (modelli, engine, validazione, DB). Esempio (disco già formattato):

```bash
sudo mkdir -p /mnt/faceid && sudo mount /dev/sda1 /mnt/faceid   # ext4 consigliato
# FAT32/exFAT/NTFS: vedi DOCUMENTAZIONE §11.2. Niente dati sull'eMMC.
```

**3 · Builda ed esegui** (sul TX2 — onnxruntime/CUDA arrivano dal base, niente wheel da procurare):

```bash
git clone https://github.com/leouni23/Face-recognition-Computer-Vision.git
cd Face-recognition-Computer-Vision
git checkout feat/jetson-optimized-profile

EXT_DISK=/mnt/faceid \
BIOMETRIC_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
PERFORMANCE_PROFILE=standard \
  sudo -E docker compose -f docker-compose.jetson.yml up --build

sudo docker images faceid:jetson-tx2   # dimensione immagine (il base dustynv è ampio: l'immagine non è "piccola")
```

Apri **[http://localhost:8000](http://localhost:8000)** → in alto scegli **Standard** o **Optimized-TX2**. Il primo avvio in Optimized costruisce l'engine TensorRT in `/data/engines` (può richiedere qualche minuto, poi è in cache). I dati di validazione vanno in `/data/validation/<sessione>_<profilo>/` — ogni run è una cartella nuova, mai sovrascritta.

**Confronto Fase 1 vs Fase 2** (offline, dai dati salvati, senza ri-accendere le camere):

```bash
python scripts/compare_sessions.py --json phase.json --csv phase.csv   # oppure il bottone "Confronta" in /validation
```

**Controlli dalla UI (dashboard, in alto):** durante il caricamento modelli (~9 min al primo avvio / al cambio profilo) il feed è **già live** e una **barra mostra il progresso %** (auto-nascosta a fine caricamento; il riconoscimento parte da solo). Il bottone **⏻** in alto a destra → **Riavvia** (uscita pulita, Docker `restart: unless-stopped` riavvia) o **Spegni** (`docker stop`). Entrambi chiedono conferma e, se `WEB_PASSWORD` è impostata, la password. `docker-compose.jetson.yml` **monta già** il socket Docker e imposta `restart: unless-stopped`: con `docker compose -f docker-compose.jetson.yml up --build` **non serve alcun flag extra**. (Se avvii con un `run.sh` manuale invece del compose, aggiungi `-v /var/run/docker.sock:/var/run/docker.sock`.)

> ⚠️ Il mount del socket dà al container accesso root all'host: usalo solo su box mono-utente e con `WEB_PASSWORD` impostata. Senza il socket, **Riavvia** funziona comunque; **Spegni** risponde 501.

> ℹ️ **Nota Python 3.6 (TX2).** L'immagine usa `requirements-jetson.txt` (pin compatibili 3.6: Flask 2.0, SQLAlchemy 1.4, pydantic v1, …) con shim che mantengono il codice identico su x86. Lo stack scientifico (scipy/scikit-image/onnx) su ARM cp36 può richiedere build lunghe se manca il wheel; in tal caso vedi la nota in DOCUMENTAZIONE §11. Se `TensorrtExecutionProvider` non è disponibile nel wheel, l'app ripiega su CUDA (segnalato nei log dei provider attivi).

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

La documentazione tecnica completa in italiano è in **[DOCUMENTAZIONE.md](DOCUMENTAZIONE.md)**: architettura interna, algoritmi, threading e protocolli web, le **feature sperimentali** (strumentazione prestazioni e benchmark, modalità di validazione open-set 1:N con FPIR/FNIR/DET/EER/CMC, bot Telegram, containerizzazione) e la guida allo studio.

---

## 📄 Licenza

MIT — vedi [LICENSE](LICENSE)
