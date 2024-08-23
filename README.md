# My Ultralytics

This repository contains documentation and customizations made to the Ultralytics engine. Descriptions and codes for each topic are stored in the [`topics`](topics) folder.

The work in this repository requires the [Ultralytics](ultralytics) package. Additional prerequisites are specified in README files for each task.

## Installation

All the examples in this repository were implemented in a **Python 3.12.4** environment with [**Ultralytics**](ultralytics) included. Additionally, newly created modules are packaged under the name `myultralytics` and can be installed using the following command:

```bash
path/to/this/repo$ pip install -e .
```

## [Custom Dataset Integration][custom_dataset_integration]

- **Overview:** We developed a method to efficiently use large datasets with non-YOLO format label files, e.g., `.json` file with xyxy-coordinates. Instead of changing the file structure or converting labels to the YOLO format, we created custom Dataset and Trainer classes that allow seamless data loading and training. For more information, please refer to the README and code in [`topics/custom_dataset_integration`][custom_dataset_integration].
- **Work Done:** Created custom Dataset and Trainer classes by modifying several classes within Ultralytics used for model training.
- **Benefits:**
    - Enables data loading and training without altering the dataset or generating YOLO format labels.
    - Continues to leverage the powerful features provided by Ultralytics!

## License

This repository uses the same AGPL-3.0 License as the Ultralytics package. See the [LICENSE](LICENSE) file for details.

[custom_dataset_integration]: topics/custom_dataset_integration
