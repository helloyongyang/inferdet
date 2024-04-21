from algo import infer_resnet


model_path = "/home/yongyang/work/projects/infer_det/test/resnet18.onnx"
backend = "onnx"

infer_instance = infer_resnet(model_path, backend)
ans = infer_instance.infer("/home/yongyang/work/projects/infer_det/test/cat.jpg")
print(ans)

