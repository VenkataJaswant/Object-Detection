import os
from ultralytics import YOLO

# Step 1: Load YOLOv8 Model
model_path = "yolov8l.pt"  # Path to trained YOLOv8 model
model = YOLO(model_path)

# Step 2: Define Output Folder
output_folder = "converted_models"
os.makedirs(output_folder, exist_ok=True)

# Step 3: Export Model to ONNX, OpenVINO, and MNN
export_formats = ["onnx", "openvino", "mnn"]
for fmt in export_formats:
    print(f"Converting YOLOv8 to {fmt} format...")
    model.export(format=fmt)

print("\n Model conversion complete for ONNX, OpenVINO, and MNN.")

