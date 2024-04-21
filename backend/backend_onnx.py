import onnxruntime as ort
import numpy as np


def infer_onnx(inputs, model):
    outputs = model.run(["output"], {"x": inputs})
    return outputs

def load_onnx(model_path):
    model = ort.InferenceSession(model_path)
    return model
