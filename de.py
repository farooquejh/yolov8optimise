
import cv2
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO Core
core = Core()

# Load OpenVINO IR model
model_path = "yolov8n_openvino_model/yolov8n.xml"
model = core.read_model(model_path)

# Compile model for CPU (change to "GPU" if you have OpenVINO GPU support)
compiled_model = core.compile_model(model, "CPU")

# Get input and output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Load and preprocess image
image_path = "frameaj211_110.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# YOLOv8 expects 640x640 input, resize accordingly
input_size = (640, 640)
resized_image = cv2.resize(image, input_size)

# Convert to RGB
resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# Normalize if needed (YOLOv8 typically expects 0-1 float input)
input_data = resized_image.astype(np.float32) / 255.0

# Change shape from HWC to CHW
input_data = np.transpose(input_data, (2, 0, 1))

# Add batch dimension
input_data = np.expand_dims(input_data, axis=0)

# Run inference
results = compiled_model([input_data])[output_layer]

# Process results
# results shape is (1, N, 6), where N = number of detections,
# each detection: [x1, y1, x2, y2, confidence, class_id]
detections = results[0]  # remove batch dim

# Filter detections by confidence threshold
conf_threshold = 0.5
detections = detections[detections[:, 4] > conf_threshold]

# Draw detections on original image
for *box, conf, cls in detections:
    x1, y1, x2, y2 = map(int, box)
    label = f"{int(cls)}:{conf:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow("YOLOv8 OpenVINO Inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

