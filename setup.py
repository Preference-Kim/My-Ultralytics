from setuptools import setup, find_packages

setup(
    name="myultralytics",
    version="0.1.0",
    description="A custom YOLO implementation for object detection",
    packages=find_packages(where='./myultralytics'),
    package_dir={'': 'myultralytics'},
    install_requires=[
        "ultralytics",
    ],
    python_requires=">=3.12.4",
)