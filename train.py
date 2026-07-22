import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11-RepGhost-imMSAM-imShapeIoU.")
    parser.add_argument("--model", default="configs/yolo11-repghost-immsam-imshapeiou.yaml")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
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
        workers=args.workers,
        seed=0,
        deterministic=True,
        shapeiou=True,
        project="runs",
        name="repghost-immsam-imshapeiou",
    )


if __name__ == "__main__":
    main()
