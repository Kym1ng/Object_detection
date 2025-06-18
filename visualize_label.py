import os
import cv2
from tqdm import tqdm

def get_kitti_filename(idx):
    """
    Return the 6-digit file name (without extension) for a given index.
    e.g., idx=10 -> '000010'
    """
    return f"{idx:06d}"

def parse_kitti_label(label_path):
    """
    Parse a KITTI label file and return the 2D bounding boxes and class names.
    """
    boxes = []
    classes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            obj_type = parts[0]
            if obj_type == "DontCare":
                continue
            left = float(parts[4])
            top = float(parts[5])
            right = float(parts[6])
            bottom = float(parts[7])
            boxes.append([int(left), int(top), int(right), int(bottom)])
            classes.append(obj_type)
    return boxes, classes

def draw_boxes_on_image(image_path, boxes, classes, output_path):
    """
    Draw bounding boxes and class names on the image and save the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image:", image_path)
        return
    for box, cls in zip(boxes, classes):
        left, top, right, bottom = box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, cls, (left, max(top-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Example usage
    kitti_root = r"E:\Projects\dataset\KittiDetection\raw\val" 
    base_output_dir = r"inference\kitti_label"
    os.makedirs(base_output_dir, exist_ok=True)
    image_dir = os.path.join(kitti_root, "image_2")
    label_dir = os.path.join(kitti_root, "label_2")

    idx = 0  # which sample you want to visualize

    for img_file in tqdm(os.listdir(image_dir)):
        if not img_file.endswith('.png'):
            continue
        idx = int(img_file.split('.')[0])

        # Generate the 6-digit base name
        base_name = get_kitti_filename(idx)  # e.g. "000010"

        # Construct paths to image and label
        image_path = os.path.join(image_dir, base_name + ".png")
        label_path = os.path.join(label_dir, base_name + ".txt")

        # Parse 2D bounding boxes from label
        boxes, classes = parse_kitti_label(label_path)

        # Output path for visualization
        current_output_file_path = os.path.join(base_output_dir, base_name + "_label.png")

        # Draw and save
        draw_boxes_on_image(image_path, boxes, classes, current_output_file_path)
