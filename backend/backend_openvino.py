import openvino.runtime as ov
import ipywidgets as widgets

core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

def load_openvino(model_path, info):
    model = core.read_model(model_path)
    model = core.compile_model(model, device.value)
    return model, info

def infer_openvino(inputs, model, info):
    outputs = model(inputs[0])
    return outputs, info
