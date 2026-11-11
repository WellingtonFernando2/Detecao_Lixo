import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from camera import Camera, tipo_camera


def parse_args():
    parser = argparse.ArgumentParser(description="Captura fotos da webcam para o dataset")
    parser.add_argument("--saida", required=True, help="Pasta onde salvar as fotos")
    parser.add_argument("--camera", type=tipo_camera, default=0, help="Indice da webcam ou o nome: celular (Iriun)")
    return parser.parse_args()


def main():
    args = parse_args()
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    try:
        cap = Camera(args.camera)
    except RuntimeError as erro:
        print(erro)
        return

    print(f"Salvando fotos em: {saida.resolve()}")
    print("Pressione [espaco] ou [s] para salvar, [q] para sair.\n")

    contador = len(list(saida.glob("*.jpg")))

    while True:
        frame = cap.read()
        if frame is None:
            print("Falha ao capturar frame.")
            break

        preview = frame.copy()
        cv2.putText(preview, f"Fotos salvas: {contador}  [espaco/s] salvar  [q] sair",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Captura de fotos", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") or key == ord(" "):
            contador += 1
            nome = f"foto_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(saida / nome), frame)
            print(f"  salva: {nome}  (total: {contador})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal de fotos em {saida}: {contador}")


if __name__ == "__main__":
    main()
