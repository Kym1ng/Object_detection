import os
import json
from pathlib import Path
from PIL import Image

def kitti_to_coco(image_folder, label_folder, output_json_path):
    """
    Converts KITTI annotations to COCO-like format and saves them as a JSON file.
    
    Args:
        image_folder (str or Path): Path to the folder containing KITTI images (.png).
        label_folder (str or Path): Path to the folder containing KITTI label text files.
        output_json_path (str or Path): Path to save the generated COCO-like JSON file.
    """
    # Initialize a COCO-like dictionary structure.
    coco_dict = {
        "info": {
            "description": "KITTI dataset in COCO format",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define KITTI label map.
    KITTI_LABEL_MAP = {
        "Car": 1,
        "Pedestrian": 2,
        "Cyclist": 3,
        "Truck": 4,
        "Misc": 5,
        "Van": 6,
        "Tram": 7,
        "Person_sitting": 8,
        "DontCare": 255  # 'DontCare' objects are skipped
    }
    
    # Create categories list for the COCO file (ignoring 'DontCare').
    for cat, cat_id in KITTI_LABEL_MAP.items():
        if cat == "DontCare":
            continue
        coco_dict["categories"].append({
            "id": cat_id,
            "name": cat,
            "supercategory": "kitti"
        })
    
    # Get sorted list of image files.
    image_files = sorted(os.listdir(image_folder))
    annotation_id = 0
    
    for image_idx, image_file in enumerate(image_files):
        # Get full image path and assign an image ID.
        image_id = image_idx + 1
        image_path = os.path.join(image_folder, image_file)
        # Corresponding label file: replace extension .png with .txt.
        label_file = image_file.replace(".png", ".txt")
        label_path = os.path.join(label_folder, label_file)

        # Skip images without a matching label file.
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_file} not found for image {image_file}. Skipping...")
            continue

        # Read image dimensions.
        with Image.open(image_path) as img:
            width, height = img.size

        # Append image info in COCO format.
        coco_dict["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        # Open and parse each line of the label file.
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                category = parts[0]
                # Skip any "DontCare" annotations.
                if category == "DontCare":
                    continue

                # The KITTI label file usually has the bounding box in parts[4:8]:
                # [xmin, ymin, xmax, ymax]
                try:
                    xmin, ymin, xmax, ymax = map(float, parts[4:8])
                except ValueError:
                    print(f"Warning: Could not parse bbox for image {image_file} in {label_file}. Skipping line...")
                    continue

                # Ensure the bounding box is within image boundaries.
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)

                # Skip invalid bounding boxes.
                if xmax <= xmin or ymax <= ymin:
                    continue

                annotation_id += 1
                coco_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": KITTI_LABEL_MAP[category],
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO format: [x, y, width, height]
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0
                })

    # Save the COCO-like annotations to the specified JSON file.
    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"COCO-like annotations saved to {output_json_path}")


# Example usage:
if __name__ == "__main__":
    # Replace these paths with the paths to your KITTI dataset folders.
    image_folder = r"E:\Projects\dataset\KittiDetection\raw\val\image_2"
    label_folder = r"E:\Projects\dataset\KittiDetection\raw\val\label_2"
    output_json_path = "kitti_annotations_val.json"
    
    kitti_to_coco(image_folder, label_folder, output_json_path)
