# SAHI-Enabled Detection Model Templating with ONNX Format

- *Last Updated: Aug. 30, 2024*

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

## Onnxruntime-GPU with Docker

### Prerequisites

- Docker, Nvidia-container-toolkit

### Build

```bash
    $ cd onnxruntime/dockerfiles && docker build -t onnxruntime-cuda -f Dockerfile.cuda ..
```

### Runtime in Container

#### Terminal

```bash
    $ docker run --gpus all -it \
        -v /local/path/to/mount:/data \
        onnxruntime-cuda:latest
```

#### Additional Settings in Container

1. Copy the main script, model weight, and sample video on mounted volume
2. Install PIL and CV2

    ```bash
        root@container_id:/# apt-get update && apt-get install libgl1-mesa-glx
        root@container_id:/# pip install pillow opencv-python
    ```

3. Run inference session

    ```bash
        root@container_id:/# cd data
        root@container_id:/data/# time python3 main.py --vid_path input.mp4 --onnx_weight yolov10n.onnx --nms True --conf 0.001 --suffix output
    ```

## References

- yolo with onnxruntime: https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py
- base model template for sahi: https://github.com/obss/sahi/blob/main/sahi/models/base.py
- sahi with onnx-yolo: https://github.com/Kazuhito00/sahi-yolox-onnx-sample/blob/main/sample_sliced_prediction.py
- dynamic batch: https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7-Dynamic-Batch-ONNXRUNTIME.ipynb