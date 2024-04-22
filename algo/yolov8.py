from infer import infer
import cv2
import numpy as np


class infer_yolov8(infer):
    def __init__(self, model_path, backend):
        super().__init__(model_path, backend)

    def preprocess(self, img_path, info):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        img = cv2.imread(img_path)

        # Get the height and width of the input image
        img_height, img_width = img.shape[:2]
        info = {"img_height": img_height, "img_width": img_width}

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (640, 640))

        # Normalize the image data by dividing it by 255.0
        img = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        img = np.transpose(img, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return [img], info

    def postprocess(self, outputs):
        class_id = np.argmax(outputs[0], axis=1)
        return class_id

