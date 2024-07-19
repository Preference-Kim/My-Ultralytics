# Custom Dataset Integration

- *Last Updated: July 19, 2024*

## Introduction

We developed a method to efficiently use large datasets with non-YOLO format label files, such as `.json` files with xyxy-coordinates. Instead of changing the file structure or converting labels to the YOLO format, we created custom Dataset and Trainer classes that allow seamless data loading and training.

## Motivation

I determined this task was necessary for the following reasons:

1. **Avoid Modifying the Dataset:** If you plan to use the same dataset for training other models besides YOLO, you may be reluctant to modify the dataset structure or annotation files solely for YOLO. This would complicate data and model maintenance.
2. **Large Dataset Size:** Creating separate YOLO format labels was impractical due to the large number of images. We aimed to train with a dataset containing over one million images and wanted to preserve the server storage used for training.
3. **Integration with Ultralytics:** We wanted to continue utilizing the diverse features provided by Ultralytics.

## Methods

We created custom Dataset and Trainer classes by modifying several classes within Ultralytics used for model training.

### Dataset

The target dataset for this work is the [Fire Prediction Videos](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=176) provided by [AI-Hub](https://www.aihub.or.kr/), Korea.

#### Directory Structure

The dataset structure is as follows:

```bash
/path/to/dataset$ tree -d
.
├── train
│   ├── fire_scene
│   │   ├── images
│   │   └── labels
│   ├── similar_scene
│   │   ├── images
│   │   └── labels
│   └── unrelated_scene
│       ├── images
│       └── labels
└── val
    ├── fire_scene
    │   ├── images
    │   └── labels
    ├── similar_scene
    │   ├── images
    │   └── labels
    └── unrelated_scene
        ├── images
        └── labels
```

#### Annotation File Format

The annotation files are in `.json` format, with each image having its own file. The annotations contain a mix of bounding box and polygon shapes. An example of the files is as follows:

1. **Box-Shaped Label**

    ```json
    {
    "image": {
        "date": "20201129",
        "path": "S3-N0877MF02525",
        "filename": "S3-N0877MF02651.jpg",
        "copyrighter": "Media Group Saram & Forest (Con)",
        "H_DPI": 96,
        "location": "08",
        "V_DPI": 96,
        "bit": "24",
        "resolution": [
        1920,
        1080
        ]
    },
    "annotations": [
        {
        "data ID": "S3",
        "middle classification": "01",
        "flags": "not occluded, not truncated",
        "box": ["min-x", "min-y", "max-x", "max-y"],
        "class": "02"
        },
        {
        "data ID": "S3",
        "middle classification": "01",
        "flags": "not occluded, not truncated",
        "box": ["min-x", "min-y", "max-x", "max-y"],
        "class": "04"
        }
    ]
    }
    ```

2. **Polygon-Shaped Label**

    ```json
    {
    "image": {
        "date": "20201209",
        "path": "S3-N0869MF02101",
        "filename": "S3-N0869MF02247.jpg",
        "copyrighter": "Media Group Saram & Forest (Con)",
        "H_DPI": 96,
        "location": "08",
        "V_DPI": 96,
        "bit": "24",
        "resolution": [
        1920,
        1080
        ]
    },
    "annotations": [
        {
        "polygon": [
            ["x", "y"]
        ],
        "data ID": "S3",
        "middle classification": "01",
        "flags": "not occluded, not truncated",
        "class": "03"
        }
    ]
    }
    ```

## Outcome

Training can be conducted in a Python environment with Ultralytics installed by running the following command:

```bash
    path/to/this/repo$ python -m topics.custom_dataset_integration.train
```

The relevant code is stored in [`topics/custom_dataset_integration/trainer`](topics/custom_dataset_integration/trainer).
