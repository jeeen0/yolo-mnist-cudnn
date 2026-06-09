"""
Train the single-class digit detector and export it.

Run on the DESKTOP GPU (training is too heavy for the Orin). YOLOv8n is tiny;
139 images train in a few minutes on a desktop CUDA GPU.

  python scripts/02_train_yolo.py

Outputs:
  runs/detect/digit/weights/best.pt        <- copy this to the Orin
On the Orin you then build the TensorRT engine:
  yolo export model=best.pt format=engine half=True device=0

We keep export() here as well (ONNX) so the desktop run is self-checking, but
the .engine must be built ON the Orin because TensorRT engines are not portable
across devices.
"""

import os
import sys

from ultralytics import YOLO

EPOCHS = 80
IMGSZ = 640
BATCH = 16
MODEL = "yolov8n.pt"        # nano: fast on Orin, plenty for one class
NAME = "digit"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    data = "data/yolo/data.yaml"
    if not os.path.exists(data):
        print("Missing", data, "- run scripts/01_make_dataset.py first.")
        sys.exit(1)

    model = YOLO(MODEL)
    model.train(
        data=data,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=NAME,
        device=0,           # desktop GPU
        patience=20,
        verbose=True,
    )

    best = os.path.join("runs", "detect", NAME, "weights", "best.pt")
    print("\nbest weights:", best)

    # Self-check export to ONNX (portable). Engine is built on the Orin.
    try:
        YOLO(best).export(format="onnx", imgsz=IMGSZ, half=False)
        print("ONNX export ok (sanity check).")
    except Exception as e:
        print("ONNX export skipped:", e)

    print("\nNext: copy", best, "to the Orin, then:")
    print("  yolo export model=best.pt format=engine half=True device=0")


if __name__ == "__main__":
    main()
