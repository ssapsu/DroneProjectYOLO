from ultralytics import YOLO
model = YOLO("./runs/detect/train4/weights/best.pt")
model.export(format='torchscript')
