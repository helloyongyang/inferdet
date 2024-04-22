from algo import infer_yolov8
from loguru import logger
from pycocotools.coco import COCO
import os
import json


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


val_path = "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/coco2017/val2017"
annFile = "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/coco2017/annotations/instances_val2017.json"

with open(annFile, "r") as fp_gt:
    gt_data = json.load(fp_gt)

detection_out_dict = {
   "images": gt_data["images"],
   "annotations": [],
   "categories": gt_data["categories"]
}

coco = COCO(annFile)

image_ids = coco.getImgIds()
images = coco.loadImgs(image_ids)


ann_idx = 0

for img_idx in range(len(images)):

    logger.info(img_idx)

    file_name = images[img_idx]["file_name"]
    img_path = os.path.join(val_path, file_name)
    results, info = infer_instance.infer(img_path, info)
    # logger.info(f"results : {results}")
    # logger.info(f"info : {info}")
    # infer_instance.show_results_single_img(img_path, results, info, "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/ans.jpg")

    for result in results:
        detection_out_dict['annotations'].append(
            {
                "image_id": images[img_idx]["id"],
                "bbox": [
                    result[2],
                    result[3],
                    result[4],
                    result[5]
                ],
                "category_id": gt_data["categories"][result[0]]["id"],
                "id": ann_idx,
                "score": 1.0,
                "area": result[4] * result[5]
            }
        )
        ann_idx += 1

with open("res.json", "w") as fp_out:
    json.dump(detection_out_dict, fp_out, ensure_ascii=False, indent=4)
