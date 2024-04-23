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
        info.update({"img_height": img_height, "img_width": img_width})

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (info["input_width"], info["input_height"]))

        # Normalize the image data by dividing it by 255.0
        img = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        img = np.transpose(img, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return [img], info

    def postprocess(self, outputs, info):
        '''
        return detections = [detection]

        each detection is [id, name, score, x_lt, y_lt, w, h]
        lt is left-top
        '''

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = info["img_width"] / info["input_width"]
        y_factor = info["img_height"] / info["input_height"]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= info["confidence_thres"]:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = (x - w / 2) * x_factor
                top = (y - h / 2) * y_factor
                width = w * x_factor
                height = h * y_factor

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, info["confidence_thres"], info["iou_thres"])

        detections = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            detection = [class_ids[i], info["class_names"][class_ids[i]], scores[i].astype(np.float64), boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
            detections.append(detection)
        return detections, info
