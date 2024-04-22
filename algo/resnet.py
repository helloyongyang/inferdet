from infer import infer
from PIL import Image
import numpy as np


class infer_resnet(infer):
    def __init__(self, model_path, backend):
        super().__init__(model_path, backend)

    def preprocess(self, img_path, info):
        img = Image.open(img_path)
        img = img.resize((256, 256), Image.BILINEAR)
        img = img.crop((16, 16, 240, 240)) # center crop to (224, 224)
        img = np.array(img, dtype='float32') / 255.0
        img -= [0.485, 0.456, 0.406]
        img /= [0.229, 0.224, 0.225]
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) # NCHW, N=1, img shape is (1, 3, 224, 224)
        return [img], info

    def postprocess(self, outputs, info):
        class_id = np.argmax(outputs[0], axis=1)
        return class_id, info

