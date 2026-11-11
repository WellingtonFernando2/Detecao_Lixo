import argparse
import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from camera import Camera, tipo_camera

DEMO_CLASS_MAP = {
    "bottle": "Vidro",
    "wine glass": "Vidro",
    "cup": "Vidro",
    "book": "Papel",
    "cell phone": "Metal",
    "fork": "Metal",
    "knife": "Metal",
    "spoon": "Metal",
    "scissors": "Metal",
}

CLASSES_ORDENADAS = ["Metal", "Papel", "Plastico", "Vidro"]

CLASS_COLORS = {
    "Metal": (60, 180, 255),
    "Papel": (80, 220, 80),
    "Plastico": (200, 100, 255),
    "Vidro": (255, 160, 40),
}
DEFAULT_COLOR = (200, 200, 200)

PANEL_WIDTH = 320
BG_COLOR = (30, 30, 30)


def parse_args():
    parser = argparse.ArgumentParser(description="Deteccao e contagem de residuos em tempo real")
    parser.add_argument("--weights", default="yolov8n.pt", help="Caminho dos pesos do YOLO (.pt)")
    parser.add_argument("--camera", type=tipo_camera, default=0,
                         help="Indice da webcam (0, 1, ...) ou o nome: celular (Iriun)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confianca minima de deteccao")
    parser.add_argument("--demo", action="store_true",
                         help="Usa o mapeamento de classes do COCO so para testar a interface")
    parser.add_argument("--linha", choices=["vertical", "horizontal", "off"], default="vertical",
                         help="Linha de contagem: so conta o objeto ao cruzar a linha (off = conta ao aparecer)")
    parser.add_argument("--posicao", type=float, default=0.5,
                         help="Posicao da linha, de 0 a 1 (0.5 = meio da imagem)")
    parser.add_argument("--sem-recorte", action="store_true",
                         help="Desliga o recorte automatico de faixas pretas da camera")
    parser.add_argument("--width", type=int, default=960, help="Largura do video exibido")
    parser.add_argument("--height", type=int, default=540, help="Altura do video exibido")
    return parser.parse_args()


def resolve_label(raw_name: str, demo: bool) -> Optional[str]:
    if demo:
        label = DEMO_CLASS_MAP.get(raw_name)
    else:
        label = raw_name.capitalize()

    if label not in CLASSES_ORDENADAS:
        return None
    return label


def draw_panel(canvas, counts_total, counts_now, frame_h):
    x0 = canvas.shape[1] - PANEL_WIDTH
    cv2.rectangle(canvas, (x0, 0), (canvas.shape[1], frame_h), BG_COLOR, -1)
    cv2.putText(canvas, "Contagem de Residuos", (x0 + 16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(canvas, (x0 + 16, 55), (canvas.shape[1] - 16, 55), (90, 90, 90), 1)

    y = 100
    for cls in CLASSES_ORDENADAS:
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        total = counts_total.get(cls, 0)
        now = counts_now.get(cls, 0)
        cv2.circle(canvas, (x0 + 26, y), 7, color, -1)
        cv2.putText(canvas, cls, (x0 + 44, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, str(total), (canvas.shape[1] - 60, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if now:
            cv2.putText(canvas, f"(+{now} na tela)", (x0 + 44, y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        y += 70

    total_items = sum(counts_total.values())
    cv2.line(canvas, (x0 + 16, y), (canvas.shape[1] - 16, y), (90, 90, 90), 1)
    cv2.putText(canvas, f"Total: {total_items}", (x0 + 16, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(canvas, "[q] sair [r] reset [s] csv [f] foto", (x0 + 16, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)


def save_report(counts_total, out_dir="relatorios"):
    Path(out_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"contagem_{ts}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["classe", "quantidade"])
        for cls in CLASSES_ORDENADAS:
            writer.writerow([cls, counts_total.get(cls, 0)])
    return path


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if not args.demo:
        CLASSES_ORDENADAS[:] = [nome.capitalize() for nome in model.names.values()]
        extras = iter([(0, 200, 255), (200, 100, 255), (255, 200, 0), (100, 255, 200)])
        for cls in CLASSES_ORDENADAS:
            CLASS_COLORS.setdefault(cls, next(extras, DEFAULT_COLOR))

    try:
        camera = Camera(args.camera, recortar=not args.sem_recorte)
    except RuntimeError as erro:
        print(erro)
        return

    counts_total = defaultdict(int)
    seen_track_ids = set()
    last_side = {}

    print("Iniciando webcam... clique na janela de video e pressione 'q' para sair.")

    while True:
        raw = camera.read()
        if raw is None:
            print("Falha ao capturar frame da camera.")
            break

        result = model.track(raw, conf=args.conf, persist=True,
                             tracker="bytetrack.yaml", verbose=False)[0]

        orig_h, orig_w = raw.shape[:2]
        out_h = args.height
        out_w = int(orig_w * out_h / orig_h)
        if out_w > args.width:
            out_w = args.width
            out_h = int(orig_h * out_w / orig_w)
        frame = cv2.resize(raw, (out_w, out_h))
        scale_x = out_w / orig_w
        scale_y = out_h / orig_h

        counts_now = defaultdict(int)
        boxes = result.boxes

        if args.linha == "vertical":
            line_pos = int(out_w * args.posicao)
            cv2.line(frame, (line_pos, 0), (line_pos, out_h), (0, 255, 255), 2)
            cv2.putText(frame, "LINHA DE CONTAGEM", (line_pos + 8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif args.linha == "horizontal":
            line_pos = int(out_h * args.posicao)
            cv2.line(frame, (0, line_pos), (out_w, line_pos), (0, 255, 255), 2)
            cv2.putText(frame, "LINHA DE CONTAGEM", (8, line_pos - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if boxes is not None and boxes.id is not None:
            names = result.names
            for box, cls_idx, track_id, conf in zip(
                boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                boxes.id.cpu().numpy(), boxes.conf.cpu().numpy()
            ):
                label = resolve_label(names[int(cls_idx)], args.demo)
                if label is None:
                    continue

                x1, y1, x2, y2 = box
                x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

                counts_now[label] += 1
                tid = int(track_id)
                key = (label, tid)
                if args.linha == "off":
                    if key not in seen_track_ids:
                        seen_track_ids.add(key)
                        counts_total[label] += 1
                else:
                    if args.linha == "vertical":
                        lado = 1 if (x1 + x2) / 2 > line_pos else -1
                    else:
                        lado = 1 if (y1 + y2) / 2 > line_pos else -1
                    anterior = last_side.get(tid)
                    if anterior is not None and anterior != lado and key not in seen_track_ids:
                        seen_track_ids.add(key)
                        counts_total[label] += 1
                    last_side[tid] = lado

                color = CLASS_COLORS.get(label, DEFAULT_COLOR)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf * 100:.0f}% #{int(track_id)}", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=BG_COLOR)
        draw_panel(canvas, counts_total, counts_now, out_h)

        cv2.imshow("TCC - Deteccao e Contagem de Residuos", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counts_total.clear()
            seen_track_ids.clear()
        elif key == ord("s"):
            saved_path = save_report(counts_total)
            print(f"Relatorio salvo em: {saved_path}")
        elif key == ord("f"):
            erros_dir = Path("fotos_brutas") / "erros"
            erros_dir.mkdir(parents=True, exist_ok=True)
            foto = erros_dir / f"erro_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(foto), raw)
            print(f"Foto salva para anotar: {foto}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
