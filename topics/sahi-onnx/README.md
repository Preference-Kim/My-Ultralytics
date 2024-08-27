# SAHI-Enabled Detection Model Templating with ONNX Format

- *Last Updated: Aug. 27, 2024*

## List of Milestones

- [X] Convert `.pt` format to `.onnx`
    - Create examples and document them in a notebook file: [`pt2onnxruntime.ipynb`](pt2onnxruntime.ipynb)
- [X] Construct an inference session with onnxruntime
    - Make a reusable function: [`main.py`](main.py)
- [X] Add a minimal slicing inference feature refering to SAHI
    - [ ] Compare the existing method using a sample video
- [X] Make a class to easily use for applications: [`main.py`](main.py)
    - [X] Write example scripts that utilize these features
    - [ ] Incorporate related features into `myultralytics`

## References

- yolo with onnxruntime: https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py
- base model template for sahi: https://github.com/obss/sahi/blob/main/sahi/models/base.py
- sahi with onnx-yolo: https://github.com/Kazuhito00/sahi-yolox-onnx-sample/blob/main/sample_sliced_prediction.py
- dynamic batch: https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7-Dynamic-Batch-ONNXRUNTIME.ipynb