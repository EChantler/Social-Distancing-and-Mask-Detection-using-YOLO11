from ultralytics import YOLO

if __name__ == "__main__":
    # Load the YOLO model
    model = YOLO("yolo11n.pt")

    # Train the model
    model.train(data="datasets/mask-dataset/mask-dataset.yaml", epochs=10, imgsz=640, freeze=10)

    model.val(data="datasets/mask-dataset/mask-dataset.yaml")