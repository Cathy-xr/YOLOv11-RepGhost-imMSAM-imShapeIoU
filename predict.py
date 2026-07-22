import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction with YOLO11-RepGhost-imMSAM-imShapeIoU.")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project="runs",
        name="predict",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
