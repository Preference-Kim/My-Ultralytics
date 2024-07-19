# My Ultralytics

This repository contains documentation and customizations made to the Ultralytics engine. Descriptions and codes for each topic are stored in the [`topics`](topics) folder.

The work in this repository requires the [Ultralytics](ultralytics) package. Additional prerequisites are specified in README files for each task.

## [Custom Dataset Integration][custom_dataset_integration]

- **Overview:** We developed a method to efficiently use large datasets with non-YOLO format label files, e.g., `.json` file with xyxy-coordinates. Instead of changing the file structure or converting labels to the YOLO format, we created custom Dataset and Trainer classes that allow seamless data loading and training. For more information, please refer to the README and code in [`topics/custom_dataset_integration`][custom_dataset_integration].
- **Work Done:** Created custom Dataset and Trainer classes by modifying several classes within Ultralytics used for model training.
- **Benefits:**
    - Enables data loading and training without altering the dataset or generating YOLO format labels.
    - Continues to leverage the powerful features provided by Ultralytics!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

[custom_dataset_integration]: topics/custom_dataset_integration
