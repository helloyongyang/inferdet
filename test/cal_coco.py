from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno_json = "../../dataset/annotations/instances_val2017.json"
pred_json = "res.json"

anno = COCO(anno_json)
pred = COCO(pred_json)
eval = COCOeval(anno, pred, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()

print(eval.stats)
