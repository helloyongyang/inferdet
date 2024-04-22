from algo import infer_yolov8
from loguru import logger


model_path = "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/yolov8n.onnx"
backend = "onnx"

class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush')

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.5,
    "iou_thres": 0.5,
    "class_names": class_names,
    "providers": ["CPUExecutionProvider"]
}

infer_instance = infer_yolov8(model_path, backend)

infer_instance.load_model(info)

img_path = "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/bus.jpg"
results, info = infer_instance.infer(img_path, info)
logger.info(f"results : {results}")
logger.info(f"info : {info}")
infer_instance.show_results_single_img(img_path, results, info, "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/bus_res.jpg")
