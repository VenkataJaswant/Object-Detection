import os
import json

# Define paths
COCO_JSON = r"C:\Users\Harini Pantangi\Desktop\Object Detection\yolo-object-detection\yolo\final\annotations\instances_default.json"
YOLO_LABELS_DIR = "dataset/labels"  # Change to where you want to save YOLO labels

# Create YOLO labels directory if not exists
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Load COCO JSON file
with open(COCO_JSON, "r") as f:
    data = json.load(f)

# Create a category mapping (COCO class IDs to YOLO class indices)
category_mapping = {cat["id"]: idx for idx, cat in enumerate(data["categories"])}

# Process each annotation
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    # Find image filename
    image_filename = None
    for img in data["images"]:
        if img["id"] == image_id:
            image_filename = img["file_name"]
            img_width, img_height = img["width"], img["height"]
            break

    if image_filename is None:
        print(f"Skipping annotation {annotation['id']} (no matching image)")
        continue

    # Convert segmentation to YOLO format
    yolo_segments = []
    for segment in segmentation:
        yolo_segment = []
        for i in range(0, len(segment), 2):
            x = segment[i] / img_width
            y = segment[i + 1] / img_height
            yolo_segment.append(f"{x:.6f} {y:.6f}")
        yolo_segments.append(" ".join(yolo_segment))

    # Save as YOLO .txt file
    label_filename = os.path.join(YOLO_LABELS_DIR, image_filename.replace(".jpg", ".txt"))
    with open(label_filename, "w") as f:
        for segment in yolo_segments:
            f.write(f"{category_mapping[category_id]} {segment}\n")

print(f"✅ Conversion complete! YOLO segmentation labels saved in: {YOLO_LABELS_DIR}")
