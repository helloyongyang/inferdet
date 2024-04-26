from algo import infer_yolov8
from loguru import logger
from pycocotools.coco import COCO
import os
import json


model_path = "../../models/yolov8n.onnx"
val_path = "../../dataset/val2017"
annFile = "../../dataset/annotations/instances_val2017.json"

backend = "openvino"

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
    "confidence_thres": 0.001,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
    "providers": ["CUDAExecutionProvider"]
}

# Load model
infer_instance = infer_yolov8(model_path, backend)
infer_instance.load_model(info)


with open(annFile, "r") as fp_gt:
    gt_data = json.load(fp_gt)

detection_out_dict = {
   "images": gt_data["images"],
   "annotations": [],
   "categories": gt_data["categories"]
}

# Load all COCO val images
coco = COCO(annFile)
image_ids = coco.getImgIds()
images = coco.loadImgs(image_ids)


ann_idx = 0
for img_idx in range(len(images)):

    logger.info(img_idx)

    file_name = images[img_idx]["file_name"]
    img_path = os.path.join(val_path, file_name)
    results, info = infer_instance.infer(img_path, info)

    for result in results:
        detection_out_dict['annotations'].append(
            {
                "image_id": images[img_idx]["id"],
                "bbox": [
                    result[3],
                    result[4],
                    result[5],
                    result[6]
                ],
                "category_id": gt_data["categories"][result[0]]["id"],
                "id": ann_idx,
                "score": result[2],
                "area": result[5] * result[6]
            }
        )
        ann_idx += 1

with open("./res.json", "w") as fp_out:
    json.dump(detection_out_dict, fp_out, ensure_ascii=False, indent=4)
