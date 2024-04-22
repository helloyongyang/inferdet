from algo import infer_yolov8


model_path = "/home/yongyang/work/projects/infer_det/test/yolov8n.onnx"
backend = "onnx"

infer_instance = infer_yolov8(model_path, backend)

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.5,
    "iou_thres": 0.5
}
results, info = infer_instance.infer("/home/yongyang/work/projects/infer_det/test/bus.jpg", info)
print(results)
print(info)
