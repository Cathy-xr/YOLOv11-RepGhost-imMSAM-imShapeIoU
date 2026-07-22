# YOLOv11-RepGhost-imMSAM-imShapeIoU

This repository provides the source code, model configuration file, trained weights, validation scripts, and result logs used in this study.

## Repository Structure

```text
YOLOv11-RepGhost-imMSAM-imShapeIoU/
├─ README.md
├─ requirements.txt
├─ data.yaml.template
├─ train.py
├─ val.py
├─ predict.py
├─ configs/
│  └─ yolo11-repghost-immsam-imshapeiou.yaml
├─ ultralytics/
├─ weights/
│  └─ best.pt
└─ results/
   ├─ res.txt
   ├─ repghost-immsam-imshapeiou.csv
   ├─ PR_curve.png
   ├─ F1_curve.png
   ├─ P_curve.png
   ├─ R_curve.png
   ├─ confusion_matrix.png
   └─ results.png
```

## Environment

Please install the required dependencies before running the code:

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.11 and Ultralytics 8.3.49.

## Dataset

The dataset is not included in this repository.

Please organize the dataset in YOLO format and modify `data.yaml` according to the local dataset path:

```text
dataset/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

The class names are:

```text
0: coal
1: gangue
2: backfill
3: marked backfill
```

An example dataset configuration is provided in `data.yaml.template`.

## Model Configuration

The proposed model configuration is provided at:

```text
configs/yolo11-repghost-immsam-imshapeiou.yaml
```

The implementation includes the following components:

```text
RepGhost backbone
imMSAM attention module
imShapeIoU loss
```

The modified Ultralytics source code is included in the `ultralytics/` directory.

## Training

To train the model, update `data.yaml` according to your local dataset path and run:

```bash
python train.py --data data.yaml --model configs/yolo11-repghost-immsam-imshapeiou.yaml
```

Default training settings:

```text
epochs: 100
imgsz: 640
batch: 64
seed: 0
deterministic: True
imShapeIoU loss: enabled
```

Note: the internal configuration flag in the code is named `shapeiou=True`, while the method is referred to as imShapeIoU in this repository.

## Validation

The trained model weights are provided at:

```text
weights/best.pt
```

To validate the model, run:

```bash
python val.py --weights weights/best.pt --data data.yaml
```

## Inference

To perform prediction on images, run:

```bash
python predict.py --weights weights/best.pt --source path/to/images
```

## Reported Results

The validation log corresponding to the reported results is provided in:

```text
results/res.txt
```

Main validation results:

```text
mAP@0.5 = 0.939
mAP@0.5:0.95 = 0.826
```

The training and validation metric records are provided in:

```text
results/repghost-immsam-imshapeiou.csv
```

## Notes

The dataset should be prepared locally due to data availability restrictions. This repository provides the source code, trained weights, model configuration file, validation scripts, and result logs required for verification of the reported model performance.
