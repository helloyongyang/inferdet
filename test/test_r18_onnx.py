from algo import infer_resnet
from loguru import logger


model_path = "/home/yongyang/work/projects/infer_det/test/resnet18.onnx"
backend = "onnx"

info = {
    "inputs_name": ["x"],
    "outputs_name" : ["output"],
    "providers": ["CPUExecutionProvider"]
}

infer_instance = infer_resnet(model_path, backend)

infer_instance.load_model(info)

results, info = infer_instance.infer("/home/yongyang/work/projects/infer_det/test/cat.jpg", info)
logger.info(f"results : {results}")
logger.info(f"info : {info}")
