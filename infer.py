import os
from loguru import logger
from abc import abstractmethod, ABCMeta
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
