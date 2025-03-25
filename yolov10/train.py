from ultralytics import YOLOv10

dataset_yaml_path = "dataset/data.yaml"

if __name__ == '__main__':
    # 预训练模型
    pre_model_name = 'yolov10s.pt'
    model = YOLOv10(pre_model_name)
    # model = YOLOv10(model_yaml_path).load(pre_model_name)
    model.train(data=dataset_yaml_path,
                epochs=150,
                imgsz=640,
                device=0,
                workers=0,
                optimizer='SGD')
