# Face Recognition System — Documentazione Tecnica Completa

## Indice
1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Architettura del Sistema](#2-architettura-del-sistema)
3. [Componenti Principali](#3-componenti-principali)
4. [Flusso Dati End-to-End](#4-flusso-dati-end-to-end)
5. [Stack Tecnologico](#5-stack-tecnologico)
6. [Conformità GDPR](#6-conformità-gdpr)
7. [Configurazione e Deploy](#7-configurazione-e-deploy)
8. [Argomenti da Studiare per la Presentazione](#8-argomenti-da-studiare-per-la-presentazione)

---

## 1. Panoramica del Progetto

Sistema di riconoscimento facciale in tempo reale con interfaccia web. Acquisisce il video dalla webcam, rileva i volti, li confronta con un database di persone iscritte e mostra i risultati live nel browser.

**Funzionalità principali:**
- Rilevamento e riconoscimento volti in tempo reale (~25 FPS)
- Accelerazione GPU tramite CUDA (NVIDIA)
- Web UI per monitoraggio, iscrizione e gestione persone
- Embedding biometrici cifrati a riposo (AES-128 + HMAC)
- Conformità GDPR: consenso, data retention, diritto all'oblio

---

## 2. Architettura del Sistema

```
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
         │  frame BGR      │  results           │  MJPEG / SSE / REST
         └────────────────►│                    │
                           │◄───────────────────┤ broadcaster
                           │                    │  (thread-safe bridge)
                    ┌──────▼──────┐             │
                    │  Database   │             │
                    │  SQLite     │             │
                    │  SQLAlchemy │             │
                    └─────────────┘             │
                                                ▼
                                        ┌──────────────┐
                                        │   Browser    │
                                        │  index.html  │
                                        └──────────────┘
```

### Struttura cartelle

```
Face-recognition-Computer-Vision/
├── config/
│   └── settings.py          # Configurazione centralizzata (Pydantic)
├── core/
│   ├── camera.py            # Acquisizione video threaded
│   ├── detector.py          # Rilevamento volti + embedding ArcFace
│   ├── recognizer.py        # Matching embedding (cosine distance)
│   └── pipeline.py          # Pipeline completa (detect → recognize → log)
├── database/
│   ├── models.py            # Tabelle ORM: Person, FaceTemplate, RecognitionEvent
│   ├── session.py           # Engine SQLAlchemy + context manager
│   └── repository.py        # CRUD e query specializzate
├── privacy/
│   ├── crypto.py            # Cifratura Fernet (AES-128-CBC + HMAC-SHA256)
│   └── retention.py         # Pulizia automatica dati scaduti (GDPR Art. 5)
├── ui/
│   └── display.py           # Annotazione frame OpenCV + FPS counter
├── web/
│   ├── app.py               # Flask endpoints (stream, SSE, enrollment, REST)
│   ├── broadcaster.py       # Bridge thread-safe tra worker e Flask
│   └── static/index.html    # Single-page app (Tailwind CSS + Vanilla JS)
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
- Per RTSP: riconnessione automatica con backoff esponenziale (fino a 30 tentativi)

**Perché un thread separato?**  
`cv2.VideoCapture.read()` è bloccante: senza thread, ogni chiamata aspetterebbe il frame successivo dalla camera, introducendo latenza variabile e bloccando tutto il resto del programma.

---

### 3.2 Rilevamento e Embedding — `core/detector.py`

Usa **InsightFace** con modello **buffalo_l** (RetinaFace + ArcFace).

**Due fasi:**

1. **Face Detection** (RetinaFace): localizza i volti nel frame → bbox `(x1, y1, x2, y2)` + 5 landmark facciali
2. **Face Recognition** (ArcFace): estrae l'**embedding**: un vettore di 512 numeri float32 che rappresenta univocamente le caratteristiche del volto

**Embedding ArcFace:**
- Output di una ResNet-100 addestrata con *ArcFace loss* (additive angular margin)
- Già normalizzato L2: `||e|| = 1` → vettore sulla sfera unitaria in 512 dimensioni
- Due volti della stessa persona → embedding vicini; persone diverse → embedding lontani

**GPU vs CPU:**
- Con `USE_GPU=true`: usa `CUDAExecutionProvider` (ONNX Runtime → NVIDIA CUDA)
- Su Windows: le DLL CUDA vengono caricate dai pacchetti `nvidia-*-cu12` installati via pip
- Speedup tipico: 5-10× rispetto a CPU

---

### 3.3 Riconoscimento — `core/recognizer.py`

`FaceRecognizer` confronta l'embedding del volto rilevato con tutti gli embedding iscritti nel database.

**Algoritmo:**

```python
# Template: matrice N×512 (N = persone iscritte)
# Query:    vettore 1×512 (volto da identificare)

cosine_distances = 1 - (template_matrix @ query_embedding)
# dato che tutti i vettori sono normalizzati L2:
# dot product = cosine similarity
# distance = 1 - similarity

best_idx = argmin(cosine_distances)
best_distance = cosine_distances[best_idx]

if best_distance < MATCH_THRESHOLD:   # default: 0.5
    return (person_id, name, confidence=1 - best_distance)
else:
    return (None, "Sconosciuto", confidence)
```

**Perché cosine distance?**  
Gli embedding ArcFace sono ottimizzati per la similarità coseno. Due embedding della stessa persona hanno cosine similarity tipicamente > 0.6 (distanza < 0.4). `MATCH_THRESHOLD=0.5` è il valore consigliato da InsightFace.

**Performance:** l'operazione è una moltiplicazione matrice-vettore vectorizzata da NumPy → O(N×512), molto veloce anche con centinaia di persone.

---

### 3.4 Pipeline — `core/pipeline.py`

`FaceIdPipeline` orchestra l'intero flusso per ogni frame:

1. **Reload template** (ogni 60s): rilegge gli embedding dal database e li carica nel recognizer. Permette di iscrivere nuove persone a caldo senza riavviare.
2. **Detect & Encode**: chiama `detect_and_encode(frame)`
3. **Identify**: per ogni volto rilevato, chiama `recognizer.identify(embedding)`
4. **Log event**: se la persona è stata riconosciuta e non è stata loggata negli ultimi 10s, registra l'evento nel database e aggiorna `last_seen`

**Throttling dei log** (`_LOG_COOLDOWN=10s`): evita di scrivere migliaia di righe al secondo nel database quando una persona è ferma davanti alla camera.

---

### 3.5 Database — `database/`

**Tre tabelle:**

| Tabella | Scopo |
|---|---|
| `Person` | Dati anagrafici: nome, consenso, date |
| `FaceTemplate` | Embedding cifrati (AES-128) |
| `RecognitionEvent` | Log: chi, quando, con quale confidenza, su quale camera |

**ORM:** SQLAlchemy v2 con pattern Repository — tutta la logica di accesso ai dati è in `repository.py`, mai SQL grezzo nei moduli di business logic.

**SQLite vs PostgreSQL:** Il progetto supporta entrambi tramite `DATABASE_URL`. SQLite non richiede installazioni aggiuntive ed è sufficiente per uso personale/locale. Per deployment multi-utente si switcherebbe a PostgreSQL cambiando solo la variabile d'ambiente.

---

### 3.6 Cifratura — `privacy/crypto.py`

Gli embedding biometrici non vengono mai salvati in chiaro nel database.

**Schema di cifratura:**
```
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

Se `BIOMETRIC_SECRET_KEY` viene ruotata, tutti i template esistenti diventano automaticamente inutilizzabili (le persone devono essere re-iscritte).

---

### 3.7 Web UI — `web/`

**Flask** espone tre categorie di endpoint:

**MJPEG Streaming** (`/stream/<camera_id>`):  
Protocollo multipart/x-mixed-replace — il server invia frame JPEG in sequenza nella stessa connessione HTTP. Il browser li interpreta come video live senza plugin.

**Server-Sent Events** (`/api/events`):  
Connessione HTTP long-lived: il server invia eventi JSON ogni volta che viene riconosciuta una persona. Il browser aggiorna la lista riconoscimenti in tempo reale senza polling.

**Enrollment API** (4 endpoint POST):  
Processo stateful in 3 step:
1. `/api/enroll/start` — inizializza sessione con nome e numero campioni richiesti
2. `/api/enroll/capture` — cattura e analizza un frame dal broadcaster (senza aprire una seconda connessione alla camera)
3. `/api/enroll/save` — calcola la media degli embedding raccolti e salva nel database

**Broadcaster** (`web/broadcaster.py`):  
Bridge thread-safe tra i worker thread (che elaborano i frame) e Flask (che li serve al browser). Usa `threading.Lock` per proteggere l'accesso ai frame condivisi e `queue.Queue` per distribuire gli eventi SSE a tutti i subscriber connessi.

---

## 4. Flusso Dati End-to-End

### 4.1 Riconoscimento in tempo reale

```
[Webcam] 
    │ frame BGR 1920×1080
    ▼
[CameraStream thread]
    │ frame BGR
    ▼
[FaceIdPipeline.process_frame()]
    │
    ├─[detect_and_encode(frame)]
    │     InsightFace: RetinaFace detection → ArcFace embedding
    │     Output: [(top,right,bottom,left), embedding_512d]
    │
    ├─[recognizer.identify(embedding)]
    │     cosine_distance = 1 - (templates @ embedding)
    │     Output: (person_id, name, confidence)
    │
    ├─[repository.log_event()] → SQLite
    │
    └─ Output: [(location, person_id, name, confidence), ...]
         │
         ├─► broadcaster.push_frame(camera_id, annotated_frame)
         │       → JPEG encode → stored in memory
         │
         └─► broadcaster.push_event(camera_id, name, confidence)
                 → messo in queue dei subscriber SSE

[Browser — MJPEG]
    GET /stream/0
    └─► broadcaster.get_frame() → JPEG bytes → img src
    
[Browser — SSE]
    GET /api/events
    └─► broadcaster.subscribe() → queue
        ogni evento: data: {"camera":"0","name":"Leo","confidence":0.91}
```

### 4.2 Iscrizione nuova persona (Web UI)

```
[Browser] POST /api/enroll/start {"name": "Leo", "samples": 5}
    └─► _enroll_session = {name: "Leo", embeddings: [], required: 5}

[Browser] POST /api/enroll/capture (×5)
    └─► broadcaster.get_raw_frame("0") → frame ndarray
        detect_and_encode(frame) → embedding
        _enroll_session["embeddings"].append(embedding)
        
[Browser] POST /api/enroll/save
    └─► mean_embedding = np.mean(embeddings, axis=0)
        normalize L2
        repository.create("Leo") → Person
        repository.give_consent(person)
        repository.add_template(person, mean_embedding)
            → encrypt_embedding() → FaceTemplate → SQLite
        pipeline.force_reload() → template ricaricati subito
```

---

## 5. Stack Tecnologico

| Componente | Tecnologia | Versione |
|---|---|---|
| Face Detection | RetinaFace (InsightFace) | buffalo_l |
| Face Recognition | ArcFace ResNet-100 | buffalo_l |
| Inference Runtime | ONNX Runtime | ≥1.18.0 |
| GPU Acceleration | CUDA | 12.6 |
| Acquisizione Video | OpenCV | ≥4.8.0 |
| Web Framework | Flask | ≥3.0.0 |
| ORM | SQLAlchemy | ≥2.0.0 |
| Database | SQLite (locale) / PostgreSQL | — |
| Crittografia | cryptography (Fernet) | ≥41.0.0 |
| Configurazione | Pydantic Settings | ≥2.0.0 |
| Logging | Loguru | ≥0.7.0 |
| Runtime Python | Python | 3.10+ |

---

## 6. Conformità GDPR

I dati biometrici (embedding facciali) rientrano nella **categoria speciale** ai sensi dell'Art. 9 GDPR.

| Articolo GDPR | Requisito | Implementazione nel progetto |
|---|---|---|
| Art. 5 — Minimizzazione | Solo dati strettamente necessari | Nessuna immagine salvata, solo vettori numerici |
| Art. 5 — Limitazione conservazione | Scadenza dati | `DATA_RETENTION_DAYS` + `run_retention()` automatico all'avvio |
| Art. 9 — Consenso esplicito | Per dati biometrici | `consent_given=True` prima di salvare qualsiasi template |
| Art. 17 — Diritto all'oblio | Cancellazione su richiesta | `DELETE /api/persons/{name}` + `delete_person.py` |
| Art. 25 — Privacy by design | Protezione by default | Embedding cifrati a riposo, nessun log immagini |
| Art. 32 — Sicurezza del trattamento | Misure tecniche | AES-128-CBC + HMAC-SHA256 (Fernet), key derivation SHA-256 |

---

## 7. Configurazione e Deploy

### Variabili d'ambiente (`.env`)

| Variabile | Default | Descrizione |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./face_id.db` | Stringa connessione database |
| `BIOMETRIC_SECRET_KEY` | — | **Obbligatoria.** Chiave 32+ byte per cifratura embedding |
| `CAMERA_SOURCES` | `0` | Indici/URL camere separati da virgola (es. `0,1` o `rtsp://...`) |
| `MATCH_THRESHOLD` | `0.5` | Soglia cosine distance (0=identici, 1=completamente diversi) |
| `USE_GPU` | `true` | Usa CUDA se disponibile |
| `DATA_RETENTION_DAYS` | `365` | Giorni di conservazione dati biometrici |
| `LOG_LEVEL` | `INFO` | Verbosità log (DEBUG/INFO/WARNING/ERROR) |

### Avvio

```bash
# Solo web UI (consigliato)
python main.py --web

# Solo finestre OpenCV locali
python main.py

# Entrambe
python main.py --web --local

# Iscrizione nuova persona da CLI
python scripts/enroll.py --name "Nome Cognome" --samples 5

# Cancellazione dati (GDPR Art. 17)
python scripts/delete_person.py --name "Nome Cognome"
```

---

## 8. Argomenti da Studiare per la Presentazione

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
- **RetinaFace**: detector convoluzionale che usa Feature Pyramid Network (FPN) per rilevare volti a scale diverse
- **Landmark facciali**: 5 punti chiave (occhi, naso, angoli bocca) usati per allineare il volto prima del riconoscimento

**Domanda tipica:** *"Come individua i volti nell'immagine?"*  
→ InsightFace usa RetinaFace, una rete neurale convoluzionale addestrata su milioni di volti. Output: bbox + 5 landmark per ogni volto trovato.

---

#### 1.3 Face Recognition con ArcFace
- **Embedding (o feature vector)**: vettore numerico che rappresenta le caratteristiche di un volto — nel progetto 512 dimensioni float32
- **ArcFace**: architettura ResNet-100 addestrata con *additive angular margin loss* per massimizzare la distanza tra classi diverse e minimizzarla tra campioni della stessa classe
- **Normalizzazione L2**: tutti gli embedding vengono proiettati sulla sfera unitaria (norma = 1)
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

### LIVELLO 2 — Architettura Software

#### 2.1 Threading in Python
- **GIL (Global Interpreter Lock)**: Python esegue un thread Python alla volta, ma l'I/O e il codice C (OpenCV, NumPy) rilasciano il GIL
- **Thread per la camera**: necessario perché `VideoCapture.read()` blocca in attesa del frame
- **Lock e Queue**: `threading.Lock` per proteggere dati condivisi, `queue.Queue` per comunicazione thread-safe tra worker e Flask

**Domanda tipica:** *"Perché usi più thread?"*  
→ La camera, la pipeline di riconoscimento e il server web devono girare in parallelo. Un singolo thread sarebbe sequenziale e il sistema andrebbe a 1-2 FPS.

---

#### 2.2 Flask e Protocolli Web
- **MJPEG**: protocollo multipart che invia frame JPEG in sequenza sulla stessa connessione HTTP. Il browser mostra "video" ma in realtà è una serie di immagini.
- **SSE (Server-Sent Events)**: connessione HTTP long-lived, il server invia eventi JSON in push al browser. Più semplice di WebSocket per flussi unidirezionali.
- **REST API**: endpoints `/api/persons`, `/api/enroll/*` seguono architettura REST (GET, POST, DELETE)

---

#### 2.3 SQLAlchemy ORM
- **ORM (Object-Relational Mapping)**: mappa classi Python a tabelle SQL
- **Repository pattern**: tutta la logica SQL è in `repository.py`, i moduli business logic non scrivono mai SQL grezzo
- **Session e transaction**: `get_session()` è un context manager che fa commit automatico o rollback in caso di errore

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
→ Sì se: (1) c'è consenso esplicito per ogni persona, (2) i dati sono cifrati, (3) c'è un meccanismo di cancellazione, (4) i dati vengono eliminati dopo la scadenza configurata.

---

#### 3.2 Crittografia Fernet
- **AES-128-CBC**: cifratura simmetrica a blocchi — lo stesso segreto cifra e decifra
- **HMAC-SHA256**: codice di autenticazione — garantisce che il ciphertext non sia stato manomesso
- **Fernet**: combina AES + HMAC in un formato standardizzato e sicuro
- **Key derivation**: la password (stringa) viene hasciata con SHA-256 per ottenere una chiave di 32 byte uniforme

---

### LIVELLO 4 — Domande Avanzate Possibili

| Domanda | Risposta sintetica |
|---|---|
| *Quante persone può gestire?* | Praticamente illimitato. Il confronto è una moltiplicazione matrice-vettore O(N×512), scalabile a migliaia. |
| *Può essere ingannato con una foto?* | ArcFace è vulnerabile a spoofing 2D senza liveness detection. Il progetto non implementa anti-spoofing (fuori scope). |
| *Perché calcoli la media di 5 campioni?* | Un singolo embedding può essere rumoroso (posa, illuminazione). La media di 5 campioni è più robusta. |
| *Cosa succede se due persone hanno facce simili?* | Il threshold 0.5 è calibrato per avere falsi positivi <1%. Può essere abbassato (più restrittivo) aumentando però i falsi negativi. |
| *Perché non salvare le immagini invece degli embedding?* | Gli embedding sono più leggeri (2 KB vs centinaia di KB), sono già "astratti" (non ricostruiscono il volto originale), e sono più facili da cifrare. |
| *Cosa succede se si perde la chiave crittografica?* | Tutti gli embedding diventano inutilizzabili. Le persone devono essere re-iscritte. |
| *Perché usare InsightFace invece di face_recognition?* | InsightFace (ArcFace) è più accurato (errore ~0.1% su LFW benchmark vs ~0.5% di face_recognition/dlib), supporta GPU nativamente, e ha modelli pre-addestrati aggiornati. |

---

### Schema Riassuntivo Studio

```
FONDAMENTALI (studia per primo)
│
├── Come funziona un'immagine digitale
├── Cos'è una rete neurale convoluzionale (CNN) — concetto base
├── Face Detection (RetinaFace, bounding box, landmark)
├── Face Embedding (ArcFace, vettore 512D, normalizzazione L2)
├── Cosine Similarity / Distance
└── Soglia di matching (threshold, false positive, false negative)

ARCHITETTURA (studia dopo)
│
├── Threading Python (perché, Lock, Queue)
├── Flask (HTTP, MJPEG, SSE, REST)
├── SQLAlchemy ORM (tabelle, sessioni, repository pattern)
└── ONNX Runtime + CUDA (execution providers, GPU speedup)

PRIVACY (studia per la parte legale)
│
├── GDPR Art. 9 (dati biometrici, consenso)
├── GDPR Art. 17 (diritto all'oblio)
├── GDPR Art. 25 (privacy by design)
├── Cifratura simmetrica (AES, chiave, ciphertext)
└── HMAC (integrità dei dati)
```

---

*Documentazione generata il 27/05/2026 — versione progetto `53f28cb`*
