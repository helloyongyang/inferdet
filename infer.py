import os
from loguru import logger
from abc import abstractmethod, ABCMeta
import cv2
from backend import *


class infer(metaclass=ABCMeta):
    def __init__(self, model_path, backend):
        self.model_path = model_path
        if os.path.exists(self.model_path):
            logger.info(f"model_path : {self.model_path}")
        else:
            raise Exception(f"{self.model_path} does not exists.")

        self.backend = backend
        if self.backend == "onnx":
            self.infer_model_func = infer_onnx
            self.load_model_func = load_onnx
        elif self.backend == "tensorrt":
            self.infer_model_func = infer_tensorrt
            self.load_model_func = load_tensorrt
        elif self.backend == "openvino":
            self.infer_model_func = infer_openvino
            self.load_model_func = load_openvino
        else:
            raise Exception(f"Not support {self.backend} backend.")

        logger.info("Loading model...")
        self.model = self.load_model(self.model_path)
        logger.info("Loading model is finished.")

    def load_model(self, model_path):
        return self.load_model_func(model_path)

    @abstractmethod
    def preprocess(self, img_path):
        pass

    def infer_model(self, inputs, info):
        return self.infer_model_func(inputs, self.model, info)

    @abstractmethod
    def postprocess(self, outputs, info):
        pass
    
    def infer(self, img_path, info):
        inputs, info = self.preprocess(img_path, info)
        outputs, info = self.infer_model(inputs, info)
        results, info = self.postprocess(outputs, info)
        return results, info

    def show_results_single_img(self, img_path, results, class_names, save_path):
        img = cv2.imread(img_path)
        for result in results:
            class_id, class_name, x1, y1, w, h = result
            (label_width, label_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
            cv2.putText(img, class_name, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(save_path, img)
