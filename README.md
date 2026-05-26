# Face Recognition – Computer Vision

Sistema di riconoscimento facciale in tempo reale, multi-camera, con storage biometrico cifrato e conforme al GDPR (Reg. UE 2016/679).

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-316192?logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Funzionalità

- **Riconoscimento in tempo reale** da una o più webcam locali o stream RTSP/RTMP
- **Identificazione biometrica** tramite embedding a 128 dimensioni (dlib HOG)
- **Confronto vettoriale ottimizzato** (cosine distance con NumPy vectorised)
- **Web UI** con streaming MJPEG live, log eventi SSE e gestione persone
- **Template cifrati** — i dati biometrici sono salvati con AES-128 + HMAC (Fernet), mai in chiaro
- **RTSP auto-reconnect** con backoff esponenziale
- **GDPR compliant** — consenso esplicito, data retention automatica, diritto all'oblio

---

## Architettura

```
face_id/
├── config/
│   └── settings.py          # Pydantic settings da .env
├── core/
│   ├── camera.py            # Acquisizione threaded (webcam + RTSP/RTMP)
│   ├── detector.py          # Rilevamento facce (dlib HOG, frame scalato)
│   ├── recognizer.py        # Identificazione (cosine distance vettoriale)
│   └── pipeline.py          # Pipeline completa + throttling DB
├── database/
│   ├── models.py            # Modelli SQLAlchemy (Person, FaceTemplate, RecognitionEvent)
│   ├── session.py           # Engine + context manager
│   └── repository.py        # Accesso dati (CRUD + caricamento template)
├── privacy/
│   ├── crypto.py            # Fernet encrypt/decrypt embedding
│   └── retention.py         # GDPR Art. 5 – cancellazione automatica dati scaduti
├── ui/
│   └── display.py           # Annotazione frame OpenCV + finestre locali
├── web/
│   ├── app.py               # FastAPI: MJPEG, SSE, REST API
│   ├── broadcaster.py       # Bridge thread-safe → asyncio per frame ed eventi
│   └── static/index.html   # Web UI (Tailwind CSS, vanilla JS)
├── scripts/
│   ├── enroll.py            # Iscrizione persona con consenso GDPR
│   ├── delete_person.py     # Cancellazione dati (GDPR Art. 17)
│   └── init_db.py           # Creazione tabelle
└── main.py                  # Entry point (locale e/o web)
```

### Flusso dati

```
Camera (thread) ──► FaceIdPipeline ──► FaceRecognizer ──► DB log
                          │
                    annotate_frame
                          │
             ┌────────────┴────────────┐
             │                         │
       Display locale             Broadcaster
       (OpenCV window)        (MJPEG + SSE eventi)
                                        │
                                   Web UI browser
```

---

## Requisiti

| Componente | Versione minima |
|---|---|
| Python | 3.10+ |
| PostgreSQL | 14+ |
| CMake | 3.20+ (per compilare dlib) |
| Webcam o stream RTSP | — |

---

## Installazione

### 1. Clona la repository

```bash
git clone https://github.com/leouni23/Face-recognition-Computer-Vision.git
cd Face-recognition-Computer-Vision
```

### 2. Crea il virtual environment e installa le dipendenze

> **Nota:** `dlib` viene compilato da sorgente — l'installazione richiede 3–5 minuti.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install cmake numpy           # prerequisiti dlib
pip install -r requirements.txt
```

### 3. Configura le variabili d'ambiente

```bash
cp .env.example .env
```

Modifica `.env`:

```env
# Connessione PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/face_id

# Chiave cifratura template biometrici (genera una nuova!)
BIOMETRIC_SECRET_KEY=<genera con: python -c "import secrets; print(secrets.token_hex(32))">

# Sorgenti camera (indice locale o URL RTSP)
CAMERA_SOURCES=0
# CAMERA_SOURCES=0,1
# CAMERA_SOURCES=rtsp://user:pass@192.168.1.100/stream

# Soglia riconoscimento (cosine distance — più basso = più severo)
MATCH_THRESHOLD=0.4

# Retention GDPR in giorni
DATA_RETENTION_DAYS=365
```

### 4. Inizializza il database

```bash
python scripts/init_db.py
```

---

## Utilizzo

### Iscrivi una persona (training)

```bash
python scripts/enroll.py --name "Mario Rossi"
```

Il programma mostrerà un'**informativa GDPR** e chiederà il consenso esplicito.  
Una volta accettato, aprirà la webcam: premi **SPAZIO** per acquisire 5 campioni del volto, poi **Q** per annullare.

```
Campione 1/5 acquisito.
Campione 2/5 acquisito.
...
✓ Persona 'Mario Rossi' iscritta con successo.
```

Opzioni disponibili:

```
--name    Nome della persona (obbligatorio)
--samples Numero di campioni (default: 5)
--camera  Indice/URL camera (default: 0)
```

---

### Avvia il riconoscimento

#### Modalità locale (finestre OpenCV)

```bash
python main.py
```

#### Modalità Web UI

```bash
python main.py --web
```

Apri il browser su **[http://localhost:8000](http://localhost:8000)**

#### Entrambe le modalità

```bash
python main.py --web --local
```

| Flag | Descrizione |
|---|---|
| `--web` | Avvia la Web UI (FastAPI + MJPEG) |
| `--local` | Mostra finestre OpenCV locali |
| `--port 8080` | Porta del server web (default: 8000) |
| `--host 0.0.0.0` | Host del server web |

Premi **Q** nella finestra OpenCV o **Ctrl-C** nel terminale per fermare.

---

### Cancella i dati di una persona (GDPR Art. 17)

```bash
python scripts/delete_person.py --name "Mario Rossi"
```

Elimina permanentemente tutti i template biometrici e gli eventi associati.

---

## Web UI

La dashboard web è accessibile su `http://localhost:8000` in modalità `--web`.

- **Stream live** — feed MJPEG con bounding box e nome identificato in tempo reale
- **Riconoscimenti** — log in diretta via Server-Sent Events (SSE), senza polling
- **Gestione persone** — lista con data ultimo accesso e pulsante di cancellazione GDPR

---

## Supporto RTSP

Il sistema supporta stream RTSP/RTMP tramite backend FFMPEG con:

- **Buffer = 1 frame** per latenza minima
- **Auto-reconnect** in caso di dropout di rete (fino a 30 tentativi)
- **Backoff automatico** di 2 secondi tra i tentativi

```env
# Camera singola RTSP
CAMERA_SOURCES=rtsp://admin:password@192.168.1.100:554/stream

# Mix webcam locale + RTSP
CAMERA_SOURCES=0,rtsp://admin:password@192.168.1.100:554/stream
```

---

## Privacy e GDPR

| Requisito GDPR | Implementazione |
|---|---|
| **Art. 5** – Minimizzazione dei dati | Solo embedding numerici, nessuna immagine salvata |
| **Art. 5** – Limitazione della conservazione | `DATA_RETENTION_DAYS` — cancellazione automatica all'avvio |
| **Art. 9** – Dati biometrici | Consenso esplicito richiesto prima dell'iscrizione |
| **Art. 17** – Diritto all'oblio | `scripts/delete_person.py` cancella tutti i dati |
| **Art. 25** – Privacy by design | Template cifrati con Fernet (AES-128-CBC + HMAC-SHA256) |
| **Invalidazione template** | Rotazione di `BIOMETRIC_SECRET_KEY` rende inutilizzabili tutti i template salvati |

> **Nota:** per un deployment in produzione valutare una DPIA (Data Protection Impact Assessment) ai sensi dell'Art. 35 GDPR.

---

## Stack tecnologico

| Libreria | Utilizzo |
|---|---|
| `face-recognition` | Embedding biometrici 128-d (dlib HOG) |
| `opencv-python` | Acquisizione video, annotazione frame |
| `SQLAlchemy` + `psycopg2` | ORM + driver PostgreSQL |
| `cryptography` (Fernet) | Cifratura AES-128 template |
| `FastAPI` + `uvicorn` | Web server async, MJPEG, SSE |
| `pydantic-settings` | Configurazione type-safe da `.env` |
| `loguru` | Logging strutturato |

---

## Licenza

MIT — vedi [LICENSE](LICENSE)
