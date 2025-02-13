import os
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ultralytics.data.utils import exif_size, IMG_FORMATS, VID_FORMATS, PIN_MEMORY, FORMATS_HELP_MSG


def img2jsonlabel_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".json" for x in img_paths]


def get_common_path(paths):
    """Get a common path among the list of Paths"""
    str_paths = [str(path) for path in paths]
    common_path = os.path.commonpath(str_paths)
    return Path(common_path)


def convert_json_to_yolo(json_data, image_width, image_height):
    # List to store YOLO format annotations
    yolo_annotations = []

    # Convert annotation data
    for annotation in json_data.get('annotations'):
        class_id = int(annotation['class']) - 1  # Class ID (YOLO uses numbers)

        if 'box' in annotation:
            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = annotation['box']
        elif 'polygon' in annotation:
            x_min, y_min, x_max, y_max = segment2boxcoords(annotation['polygon'])
        else:
            raise ValueError(json_data['image']['filename'] + ": Annotation must contain either 'box' or 'polygon'")
        
        # Calculate center coordinates and width & height
        x_center = (x_min + x_max) / 2.0 / image_width
        y_center = (y_min + y_max) / 2.0 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Convert to YOLO format
        yolo_annotation = [class_id, x_center, y_center, width, height]
        yolo_annotations.append(yolo_annotation)

    return yolo_annotations


def segment2boxcoords(seg):
    """
    A function that calculates and returns x_min, y_min, x_max, y_max from a given list of coordinates.
    
    Parameters:
    seg (list): A list of coordinates in the form [x, y].
    
    Returns:
    list: [x_min, y_min, x_max, y_max]
    """

    def filter_coordinates(lst):
        """
        Filters a list to only include elements that are [x, y] coordinates.

        Parameters:
        lst (list): A list that may contain various elements.

        Returns:
        list: A new list with only [x, y] coordinate elements.
        """
        return [item for item in lst if isinstance(item, list) and len(item) == 2 and all(isinstance(i, (int, float)) for i in item)]
    
    try:
        coords = np.array(seg)
    except ValueError:
        coords = np.array(filter_coordinates(seg))

    x, y = coords.T  # Separate the list of coordinates into x and y
    x_min = x.min()
    y_min = y.min()
    x_max = x.max()
    y_max = y.max()
    return x_min, y_min, x_max, y_max


def verify_image_label_json(args):
    """Reference module: `verify_image_label` from `ultralytics.data.utils`"""
    """Verify one image-label pair."""
    
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
                    
        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            try:
                with open(lb_file, 'r') as f:
                    json_data = json.load(f)
            except json.JSONDecodeError: # Unexpected UTF-8 BOM (decode using utf-8-sig)
                with open(lb_file, 'r', encoding='utf-8-sig') as f:
                    json_data = json.load(f)
                with open(lb_file, 'w', encoding='utf-8') as file:
                    json.dump(json_data, file, ensure_ascii=False, indent=4)

            # Get image resolution
            image_width, image_height = json_data['image']['resolution']

            # Convert JSON data to YOLO format
            lb = convert_json_to_yolo(json_data, image_width, image_height)    
            lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls <= num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]