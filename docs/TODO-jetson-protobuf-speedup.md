# TODO — Azzerare i ~9 min di caricamento modelli su Jetson TX2

## Problema
All'avvio (vedi `core/detector.warmup`, lanciato in thread da `main.py`) InsightFace
carica i modelli `buffalo_l` (`w600k_r50` 174 MB + `1k3d68` 143 MB). Il caricamento passa
da `onnx.load_model` → `protobuf ParseFromString`, e nell'immagine attuale
(`dustynv/onnxruntime:r32.7.1`, Python 3.6, protobuf 3.19.6) **il backend protobuf è
pure-Python** (`api_implementation.Type() == "python"`, manca `google.protobuf.pyext._message`).
Parsare ~317 MB di ONNX in pure-Python costa **~550 s** ad ogni avvio del container
(misurato: `pre-caricamento completato in 550.2s`).

Il pre-warm in background tiene la UI reattiva ma **non riduce** il costo: ogni restart
ripaga i ~9 min. Finché il warmup non finisce la camera risulta `connected` ma il feed è nero.

## Feature da aggiungere
Abilitare il **backend C++ di protobuf** (`_message` / upb) nell'immagine così il parse
scende da minuti a ~secondi (~100x).

### Approccio (in `Dockerfile.jetson`)
1. Installare una build di protobuf con estensione C++ per aarch64 / cp36, p.es.:
   ```dockerfile
   RUN pip3 install --no-cache-dir "protobuf==3.19.6" \
       && python3 -c "from google.protobuf.internal import api_implementation as a; assert a.Type()=='cpp', a.Type()"
   ```
   (la wheel `manylinux2014_aarch64 cp36` include `pyext/_message`; verificare che venga
   scelta al posto della sdist pure-Python).
2. Forzare il backend C++ a runtime:
   ```dockerfile
   ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
   ```
3. Verificare compatibilità con la versione protobuf richiesta da onnx/onnxruntime nel base
   image (pin coerente per non rompere onnxruntime).

### Alternative
- Saltare del tutto `onnx.load_model` in fase di metadata e passare il file direttamente a
  onnxruntime (parser C++ nativo), rimuovendo i modelli non usati (`1k3d68`, `2d106det`,
  `genderage`) dal pack — `allowed_modules=["detection","recognition"]` non li usa. Rischio:
  insightface può ri-scaricare il pack se rileva file mancanti.
- Convertire i modelli in un formato a load rapido / cache serializzata in RAM.

## Verifica attesa
Dopo il fix: log `pre-caricamento completato in <pochi secondi>`, feed camera attivo entro
secondi dall'avvio invece di ~9 min.
