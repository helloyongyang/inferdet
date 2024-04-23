from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno_json = "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/coco2017/annotations/instances_val2017.json"
pred_json = "/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/inferdet/test/res.json"

anno = COCO(anno_json)
pred = COCO(pred_json)
eval = COCOeval(anno, pred, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()

print(eval.stats)
