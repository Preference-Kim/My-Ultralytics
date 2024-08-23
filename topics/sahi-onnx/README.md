# SAHI-Enabled Detection Model Templating with ONNX Format

- *Last Updated: Aug. 23, 2024*

## List of Milestones

- [ ] Convert `.pt` format to `.onnx`
    - Create examples and document them in a notebook file
- [ ] Construct an inference session with onnxruntime
    - Make a reusable function
- [ ] Add a minimal slicing inference feature refering to SAHI
    - Compare the existing method using a sample video
- [ ] Make a class to easily use for applications
    - Incorporate related features into `myultralytics` and write example scripts that utilize these features

## References

- yolo with onnxruntime: https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py
- base model template for sahi: https://github.com/obss/sahi/blob/main/sahi/models/base.py
- sahi with onnx-yolo: https://github.com/Kazuhito00/sahi-yolox-onnx-sample/blob/main/sample_sliced_prediction.py