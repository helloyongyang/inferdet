import onnxruntime as ort
import numpy as np


def load_onnx(model_path):
    model = ort.InferenceSession(model_path)
    return model

def infer_onnx(inputs, model, info):
    outputs_name = info["outputs_name"]
    inputs_name = info["inputs_name"]
    inputs_dict = dict(zip(inputs_name, inputs))
    outputs = model.run(outputs_name, inputs_dict)
    return outputs, info
