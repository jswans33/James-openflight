from ultralytics import YOLO

model = YOLO('models/golf_ball_yolo11n_new.pt')
model.export(format='ncnn', imgsz=640)