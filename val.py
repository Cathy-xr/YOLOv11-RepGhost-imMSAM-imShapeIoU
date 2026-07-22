import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO11-RepGhost-imMSAM-imShapeIoU.")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="val",
        plots=True,
        save_json=False,
    )


if __name__ == "__main__":
    main()
