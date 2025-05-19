import cv2
from ultralytics import YOLO

model_path = "yolov8n_openvino_model/"
model = YOLO('yolov8n_openvino_model/',task='detect')

# Read image
img = cv2.imread('frameaj211_110.jpg')  # replace with your image path

# Run inference on the image
results = model(img)

# Draw bounding boxes and labels on the image
for box, score, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
    x1, y1, x2, y2 = map(int, box)
    label = f"{model.names[int(cls)]} {score:.2f}"
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Put label text above rectangle
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow('YOLOv8 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

