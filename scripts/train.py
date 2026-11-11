import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Treino do YOLO para classificacao de residuos")
    parser.add_argument("--data", default="data/dataset_pequeno.yaml", help="Caminho do arquivo .yaml do conjunto de dados")
    parser.add_argument("--model", default="yolov8n.pt", help="Pesos base (yolov8n.pt, yolov8s.pt, ...)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'0' para GPU, 'cpu' para CPU")
    parser.add_argument("--name", default="train", help="Nome da pasta de saida em runs/detect/")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
