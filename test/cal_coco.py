from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    anno_json = r'/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/coco2017/annotations/instances_val2017.json'
    pred_json = r'/mnt/nvme1/yongyang/projects/mqb/shenlan_quant/L6/inferdet/test/res.json'
    anno = COCO(anno_json)    # init annotations api
    pred = COCO(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    # map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    print(eval.stats)