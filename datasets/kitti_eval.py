import torch
import json
import tempfile
from pycocotools.coco import COCO
from .kitti import KittiDetection, KITTI_LABEL_MAP

def get_kitti_api_from_dataset(dataset):
    """
    Convert a KittiDetection dataset (whose targets have normalized boxes)
    into a COCO-style dataset dictionary and return a COCO API object.
    """

    # Unwrap Subset if necessary
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    if not isinstance(dataset, KittiDetection):
        raise TypeError("This function only works with KittiDetection datasets.")

    def denormalize_boxes(normalized_boxes, size):
        """
        Denormalize boxes that are in normalized (cx, cy, w, h) format.
        'size' is the resized image size as [height, width].
        Returns boxes in COCO format: [xmin, ymin, width, height].
        """
        h, w = size  # size is [height, width]
        # Multiply by image dimensions to get absolute values in (cx, cy, w, h)
        abs_boxes = normalized_boxes * torch.tensor([w, h, w, h], dtype=normalized_boxes.dtype, device=normalized_boxes.device)
        # Convert from center coordinates to (xmin, ymin, width, height)
        cx, cy, bw, bh = abs_boxes.unbind(1)
        xmin = cx - bw / 2
        ymin = cy - bh / 2
        return torch.stack([xmin, ymin, bw, bh], dim=1)

    # Create the base dictionary in COCO format
    kitti_dict = {
        "info": {
            "description": "KITTI converted to COCO format",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "2025-03-01"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Build the 'categories' list from KITTI_LABEL_MAP (excluding "DontCare")
    label_map = {k: v for k, v in KITTI_LABEL_MAP.items() if k != "DontCare"}
    for label_name, label_id in label_map.items():
        kitti_dict["categories"].append({
            "id": label_id,
            "name": label_name,
            "supercategory": "kitti"
        })

    ann_id = 0
    # Loop over the dataset
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        image_id = int(target["image_id"].item())
        # Use the "size" key, which is the current image size after transforms, [height, width]
        h, w = target["size"].tolist()
        file_name = f"{image_id:06d}.png"

        # Add image info
        kitti_dict["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": w,
            "height": h
        })

        # Denormalize boxes that are stored in normalized (cx,cy,w,h) format.
        normalized_boxes = target["boxes"]
        if idx == 0:
            print("Normalized boxes:", normalized_boxes)
        # abs_boxes = denormalize_boxes(normalized_boxes, (h, w))
        for box, label in zip(normalized_boxes, target["labels"]):
            x1, y1, bw, bh = box.tolist()
            ann_id += 1
            kitti_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x1, y1, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0
            })

    # Write the dictionary to a temporary JSON file and load it as a COCO API object.
    tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp_json.name, 'w') as f:
        json.dump(kitti_dict, f)
    coco_api = COCO(tmp_json.name)
    tmp_json.close()

    return coco_api
