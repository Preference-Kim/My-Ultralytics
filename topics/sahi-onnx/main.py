from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Optional, List, Tuple
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import onnxruntime as ort


EP_LIST = [
    ('CUDAExecutionProvider', {'device_id': 0}), 
    'CPUExecutionProvider',
] # prefer CUDA Execution Provider over CPU Execution Provider


@dataclass
class YOLOConfig:
    weight_path: Union[str, Path]

    # Use default_factory for mutable default values to avoid shared references
    classes: List[str] = field(default_factory=lambda: [
        'black_smoke',
        'gray_smoke',
        'white_smoke',
        'flame',
        'cloud',
        'fog',
        'lamp_light',
        'sun_light',
        'shaky_object',
        'wind-swayed_leaves',
        'irrelevant',
    ])
    
    classes_aligned: List[Optional[str]] = field(default_factory=lambda: [
        'Smoke', 'Flame', None
    ])
    
    classes_map: List[int] = field(default_factory=lambda: [
        0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2  # 0: smoke, 1: flame, 2: none
    ])
    
    color_palette: List[tuple] = field(default_factory=lambda: [  # BGR
        (255, 0, 0),  # Smoke
        (0, 0, 255),  # Flame
        (0, 0, 0)     # None
    ])


class MyModel(ABC):
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass
    
    
class YOLOFireOnnx(MyModel):


    INPUT_SHAPE = (640,640)
    NUM_SLICES = (3,2) # horizontal, width

    def __init__(self, model_config:YOLOConfig, execution_provider:Union[list,str]=EP_LIST, resolution:tuple[int]=(2160,3840)) -> None:
        self.provider = execution_provider # prefer CUDA Execution Provider over CPU Execution Provider
        self.model = None
        self.model_path:Path = Path(model_config.weight_path)
        self.model_format:str = self.model_path.suffix
        assert self.model_format == '.onnx', f"This class only supports '.onnx' format.\nPlease check the model file: {str(self.model_path.resolve())}"
        
        self.image_height, self.image_width = resolution
        
        self.slice_length = int(self.image_height * 11 / 18)
        self.ratio_input2slice = float(self.slice_length/self.INPUT_SHAPE[0])
        self.offsets = (-int(self.slice_length / 22), -int(self.slice_length * 4 / 11)) # horizontal, width
        self.slice_positions = []
        for i in range(self.NUM_SLICES[1]):
            for j in range(self.NUM_SLICES[0]):
                x = j * (self.slice_length+self.offsets[0])
                y = i * (self.slice_length+self.offsets[1])
                self.slice_positions.append((x, y))
                
        self.classes:list       = model_config.classes
        self.classes_aligned    = model_config.classes_aligned
        self.classes_map        = model_config.classes_map
        self.color_palette      = model_config.color_palette
        

    def load_model(self):
        
        self.model = ort.InferenceSession(self.model_path, providers=self.provider)

        # Get the model inputs
        model_inputs = self.model.get_inputs()

        # Store the shape of the input for later use
        self.batch_shape = model_inputs[0].shape
        self.input_width = self.batch_shape[2]
        self.input_height = self.batch_shape[3]
        assert self.INPUT_SHAPE == (self.input_width, self.input_height), f"Unexpected input shape: {(self.input_width, self.input_height)}"
        
        return self
    
    
    def preprocess(self, im: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed batch ready for inference.
        """

        # Convert the image color space from BGR to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        slices, _ = self.slice_image(im)

        # Resize the image to match the input shape
        # Normalize the image data by dividing it by 255.0
        slices_resized = np.array([cv2.resize(slc, self.INPUT_SHAPE, interpolation=cv2.INTER_LINEAR) for slc in slices]) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(slices_resized, (0, 3, 1, 2)).astype(np.float32)  # Batch, Channel, H, W

        # Return the preprocessed image data
        return np.ascontiguousarray(image_data)
    
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """
        Batch Inference with preprocessed data

        Returns:
            (6, 300, 6) shaped numpy array
        """
        return self.model.run(None, {'images': batch})[0]


    def postprocess(self, original_img:np.ndarray, prediction_result:np.ndarray, confidence_thres:float=0.05, iou_thres:float=0.5) -> np.ndarray:
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            original_img (numpy.ndarray): The original image.
            prediction_result (numpy.ndarray): The outputs of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        for outputs_slice, pos_slice in zip(prediction_result, self.slice_positions):
            # Get the number of rows in the outputs array
            rows = outputs_slice.shape[0]
            
            # Iterate over each row in the outputs array
            for i in range(rows):
                # Extract the score from the current row
                best_score = outputs_slice[i][4]

                # If the maximum score is above the confidence threshold
                if best_score >= confidence_thres:
                    # Get the class ID with the highest score
                    class_id = int(outputs_slice[i][-1]) # detected class
                    class_id_aligned = self.classes_map[class_id]
                    
                    # Neglact `None` class
                    if self.classes_aligned[class_id_aligned]:
                        pass
                    else:
                        continue

                    # Extract the bounding box coordinates from the current row
                    x1, y1, x2, y2 = outputs_slice[i][0], outputs_slice[i][1], outputs_slice[i][2], outputs_slice[i][3]

                    # Calculate the scaled coordinates of the bounding box
                    left = int(x1 * self.ratio_input2slice) + pos_slice[0]
                    top = int(y1 * self.ratio_input2slice) + pos_slice[1]
                    width = int((x2-x1) * self.ratio_input2slice)
                    height = int((y2-y1) * self.ratio_input2slice)

                    # Add the class ID, score, and box coordinates to the respective lists
                    class_ids.append(class_id_aligned)
                    scores.append(best_score)
                    boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)


        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(original_img, box, score, class_id)

        # Return the modified input image
        return original_img


    """ ============== PRE-PROCESSING UTILS ============== """
    
    
    def slice_image(self,
        image: np.ndarray,
        output_file_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Tuple[List[np.ndarray], Optional[str]]:
        """
        Slice a large image (numpy array) into a fixed number of windows: 3 horizontal slices and 2 vertical slices.
        Each slice size is determined as 11/18 of the image's height.

        Args:
            image (np.ndarray): Image to be sliced as a numpy array (height x width x channels).
            output_file_name (str, optional): Root name of output files.
            output_dir (str, optional): Output directory.

        Returns:
            Tuple: A list of sliced images as numpy arrays and the directory of the exported images if applicable.
                sliced_images[idx]: slice at (idx//NUM_SLICES[1], idx%NUM_SLICES[1])-th order
        """

        sliced_images = []
        for (x, y) in self.slice_positions:
            slice_image = image[y:y + self.slice_length, x:x + self.slice_length]
            sliced_images.append(slice_image)

            # Save image if output_dir and output_file_name are provided
            if output_file_name and output_dir:
                if not Path(output_dir).exists():
                    Path(output_dir).mkdir(parents=True)
                slice_file_name = f"{output_file_name}_{x}_{y}.png"
                slice_file_path = Path(output_dir) / slice_file_name
                Image.fromarray(slice_image).save(slice_file_path)

        return sliced_images, output_dir if output_file_name and output_dir else None
    
    
    """ ============== POST-PROCESSING UTILS ============== """
    
    
    def draw_detections(self, img:np.ndarray, box, score, class_id) -> None:
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        if class_name := self.classes_aligned[class_id]:
            pass
        else:
            return

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{class_name}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    
""" ============== MAIN ============== """


if __name__=='__main__':
    
    model = None
    
    working_dir = Path(__file__).parent
    
    vid_path = working_dir / 'samples/sample1.mp4'
    cap = cv2.VideoCapture(vid_path)
    ret, im = cap.read()
    
    onnx_weight = working_dir / 'models/yolov10n.onnx'
    config = YOLOConfig(weight_path=onnx_weight)
    my_model = YOLOFireOnnx(model_config=config, resolution=im.shape[:2])
    model = my_model.load_model()
    
    batch = model.preprocess(im)
    batch_results = model.predict(batch)
    result_image = model.postprocess(original_img=im, prediction_result=batch_results, confidence_thres=0.05, iou_thres=0.5)

    cv2.imwrite(working_dir / 'main_result.jpg', result_image)