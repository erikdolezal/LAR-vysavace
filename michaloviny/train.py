from ultralytics import YOLO

if __name__ == '__main__':
# Create a YOLO model instance with a pre-trained weight file
    model = YOLO('yolo11n.pt')

    # Train the model using the data configuration file
    model.train(data="yolo/data.yaml", epochs=120, imgsz=640,  device=0)

    model.export(format="pt")

    results = model.val()

    # Print validation results
    print(results)