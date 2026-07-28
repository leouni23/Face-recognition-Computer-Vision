#!/usr/bin/env python3
"""Diagnostica portata del rilevatore: quanto e' grande un volto nel frame e quale
cancello lo scarta (rete che non lo vede / score sotto soglia / min_face_px).

Eseguire DENTRO il container, con il soggetto in posa alla distanza da testare:
  python3 /app/scripts/diag_face_size.py --camera cam3 --frames 5
  python3 /app/scripts/diag_face_size.py --image /tmp/frame.jpg

Il frame arriva da GET /api/cameras/<id>/snapshot (l'ultimo frame gia' decodificato dal
worker: nessuna seconda connessione RTSP alla camera). Il rilevatore gira su
CPUExecutionProvider di proposito, per non contendere GPU/memoria al container in
esercizio ne' innescare build TensorRT.

Per ogni combinazione (risoluzione input del rilevatore x soglia di confidenza) stampa
le altezze bbox MISURATE IN PIXEL DEL FRAME ORIGINALE: e' il numero con cui si tara
MIN_FACE_PX.
"""
import argparse
import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Risoluzioni input: 640x640 e' il vecchio default (44% del canvas sprecato in padding su
# un 16:9); le altre sono 16:9 con entrambi i lati multipli di 32 (vincolo delle griglie
# di anchor SCRFD, che sono calcolate come input_size // stride con stride fino a 32).
DET_INPUTS = [(640, 640), (960, 544), (1280, 736), (1920, 1088)]
DET_THRESHOLDS = [0.5, 0.3, 0.15]


def fetch_frame(camera_id, port=8000):
    """Ultimo frame grezzo della camera via API (usa il worker gia' attivo)."""
    try:
        from urllib.request import Request, urlopen
    except ImportError:  # py2 — non succede nel container, ma non fallire in modo oscuro
        raise SystemExit("Serve Python 3")
    from config.settings import get_settings

    url = "http://127.0.0.1:%d/api/cameras/%s/snapshot" % (port, camera_id)
    req = Request(url)
    password = get_settings().web_password
    if password:
        token = base64.b64encode((":" + password).encode()).decode()
        req.add_header("Authorization", "Basic " + token)
    raw = urlopen(req, timeout=15).read()
    if raw[:2] != b"\xff\xd8":
        raise SystemExit("Risposta non JPEG da %s: %s" % (url, raw[:200]))
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit("Frame non decodificabile")
    return frame


def build_detector(pack, gpu=False):
    """FaceAnalysis con il solo rilevatore. CPU di default: non tocca GPU/TensorRT del
    container in esercizio. NB: con il container che satura i core, il caricamento su CPU
    può richiedere ~10 minuti (misurato sul TX2); l'inferenza poi è 0,25-1,1 s/frame.
    Con --gpu si carica in secondi ma si contende memoria GPU al servizio."""
    from insightface.app import FaceAnalysis

    t0 = time.perf_counter()
    providers = ["CUDAExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=pack, allowed_modules=["detection"], providers=providers)
    app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
    print("Rilevatore '%s' pronto su %s in %.1fs"
          % (pack, "GPU" if gpu else "CPU", time.perf_counter() - t0), flush=True)
    return app.det_model


def run_sweep(det_model, frames, min_face_px):
    """Sweep risoluzione x soglia. Ritorna {(wh, thr): [altezze bbox nel frame originale]}."""
    results = {}
    for size in DET_INPUTS:
        for thr in DET_THRESHOLDS:
            det_model.det_thresh = thr
            heights, scores, ms = [], [], []
            for frame in frames:
                t0 = time.perf_counter()
                bboxes, _ = det_model.detect(frame, input_size=size, max_num=0,
                                             metric="default")
                ms.append((time.perf_counter() - t0) * 1000.0)
                for i in range(bboxes.shape[0]):
                    x1, y1, x2, y2 = bboxes[i, 0:4]
                    heights.append(float(y2 - y1))
                    scores.append(float(bboxes[i, 4]))
            results[(size, thr)] = (heights, scores, ms)
            kept = [h for h in heights if h >= min_face_px]
            print("  input %4dx%-4d thr %.2f -> %2d volti  h_px=%s  score=%s  %6.0f ms/frame"
                  " | dopo min_face_px=%d ne restano %d"
                  % (size[0], size[1], thr, len(heights),
                     _fmt([round(h) for h in heights]), _fmt([round(s, 2) for s in scores]),
                     float(np.mean(ms)) if ms else 0.0, min_face_px, len(kept)), flush=True)
    return results


def _fmt(values, limit=6):
    if not values:
        return "[]"
    head = values[:limit]
    return "[" + ", ".join(str(v) for v in head) + ("...]" if len(values) > limit else "]")


def verdict(results, min_face_px, baseline=(640, 640), strict=0.5):
    """Attribuisce il fallimento al cancello giusto, confrontando la configurazione in
    esercizio (baseline @ soglia stretta) con la piu' permissiva dello sweep."""
    base_h = results[(baseline, strict)][0]
    best_key = max(results, key=lambda k: len(results[k][0]))
    best_h, best_s, _ = results[best_key]
    print("\n== VERDETTO ==")
    if not best_h:
        print("Nessun volto rilevato in NESSUNA configurazione: nel frame non c'e' un volto,"
              " oppure e' troppo piccolo/sfocato/di profilo anche a piena risoluzione.")
        print("Ripetere con il soggetto in posa frontale alla distanza da testare.")
        return
    med = float(np.median(best_h))
    print("Volto piu' grande visto: %d px di altezza (mediana %d px) con input %dx%d @ thr %.2f"
          % (round(max(best_h)), round(med), best_key[0][0], best_key[0][1], best_key[1]))
    if not base_h:
        # La rete a 640x640 non lo vede proprio: il volto arriva alla rete gia' sotto il
        # pavimento di rilevabilita' SCRFD (~16-20 px sull'input).
        print("CANCELLO 1 (risoluzione input): a %dx%d @ thr %.2f il volto NON viene rilevato,"
              " mentre a %dx%d si'. Il collo di bottiglia e' la risoluzione del rilevatore:"
              " alzare DET_INPUT_WIDTH/HEIGHT." % (baseline[0], baseline[1], strict,
                                                   best_key[0][0], best_key[0][1]))
    else:
        loose = results[(baseline, DET_THRESHOLDS[-1])][0]
        if len(loose) > len(base_h):
            print("CANCELLO 2 (soglia): a %dx%d il volto compare solo abbassando det_thresh"
                  " (%d volti a %.2f contro %d a %.2f). Abbassare DET_THRESHOLD."
                  % (baseline[0], baseline[1], len(loose), DET_THRESHOLDS[-1],
                     len(base_h), strict))
        elif max(base_h) < min_face_px:
            print("CANCELLO 3 (min_face_px): il volto E' rilevato (%d px) ma viene scartato in"
                  " silenzio da min_face_px=%d. Abbassare MIN_FACE_PX sotto %d."
                  % (round(max(base_h)), min_face_px, round(max(base_h))))
        else:
            print("Nella configurazione in esercizio il volto passa tutti i cancelli"
                  " (%d px >= min_face_px=%d): il problema non e' il rilevamento."
                  % (round(max(base_h)), min_face_px))
    suggested = max(16, int(med * 0.6) // 8 * 8)
    print("MIN_FACE_PX suggerito per questa distanza: %d (60%% della mediana misurata)."
          % suggested)


def main():
    from config.settings import get_settings
    from core.profile import get_profile

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="cam3", help="camera_id da interrogare")
    parser.add_argument("--frames", type=int, default=3, help="quanti frame campionare")
    parser.add_argument("--interval", type=float, default=1.0, help="secondi tra i frame")
    parser.add_argument("--image", help="usa un JPEG su disco invece della camera")
    parser.add_argument("--pack", default=None, help="model pack (default: quello attivo)")
    parser.add_argument("--save-dir", default=None, help="salva i frame campionati qui")
    parser.add_argument("--gpu", action="store_true",
                        help="carica su GPU (secondi invece di ~10 min) contendendola al servizio")
    parser.add_argument("--loop", type=int, default=0, metavar="N",
                        help="ripeti lo sweep ogni N secondi (il modello si carica una volta "
                             "sola): permette di misurare più distanze in una sessione")
    args = parser.parse_args()

    pack = args.pack or get_profile().model_pack
    min_face_px = settings.min_face_px

    def grab(round_no=0):
        frames = []
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise SystemExit("Immagine non leggibile: %s" % args.image)
            frames.append(frame)
        else:
            for i in range(max(1, args.frames)):
                frames.append(fetch_frame(args.camera))
                if i + 1 < args.frames:
                    time.sleep(args.interval)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(args.save_dir, "diag_%02d_%02d.jpg" % (round_no, i)), f)
        return frames

    frames = grab()
    h, w = frames[0].shape[:2]
    print("Frame: %dx%d  (%d campioni)  pack=%s  min_face_px=%d  det_threshold=%.2f"
          % (w, h, len(frames), pack, min_face_px, settings.det_threshold), flush=True)

    det_model = build_detector(pack, gpu=args.gpu)
    round_no = 0
    while True:
        print("\nSweep (altezze bbox in px del FRAME ORIGINALE):", flush=True)
        results = run_sweep(det_model, frames, min_face_px)
        verdict(results, min_face_px)
        if not args.loop:
            break
        print("\n--- prossimo sweep tra %ds (spostarsi alla distanza successiva) ---"
              % args.loop, flush=True)
        time.sleep(args.loop)
        round_no += 1
        frames = grab(round_no)


if __name__ == "__main__":
    main()
