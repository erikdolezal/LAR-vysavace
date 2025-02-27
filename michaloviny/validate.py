from ultralytics import YOLO
import cv2
import os

def load_yolo_model(model_path='best_ones/v11n_120.pt'):
    return YOLO(model_path)

def process_image(model, input_path, output_path):
    results = model(input_path)
    print(results)
    for result in results:
        image = result.plot()
        cv2.imwrite(output_path, image)
    print(output_path)
    
if __name__ == "__main__":
    model = load_yolo_model()
    
    for file in os.listdir('validate/np'):
        print(file)
        input_path = os.path.join('validate/np', file)
        output_path = os.path.join('validate/np/output', file)
        process_image(model, input_path, output_path)