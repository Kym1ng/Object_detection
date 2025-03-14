from pathlib import Path
import torch
import torch.utils.data
import torchvision
import datasets.transforms as T

KITTI_LABEL_MAP = {
    "Car": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
    "Truck": 4,
    "Misc": 5,
    "Van": 6,
    "Tram": 7,
    "Person_sitting": 8,
    "DontCare": 255  # 'DontCare' objects are ignored
}

class KittiDetection(torchvision.datasets.Kitti):
    def __init__(self, root, training_flag, transforms=None):
        super(KittiDetection, self).__init__(root, train=training_flag, download=False)
        self._transforms = transforms


    def __getitem__(self, idx):
        img, ann = super(KittiDetection, self).__getitem__(idx)
        # Convert annotations to DETR-compatible format
        target = self.prepare_kitti_target(idx, ann, img)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def prepare_kitti_target(self, idx, ann, img):
        """ Converts KITTI annotations into COCO-like format for DETR. """
        w, h = img.size  # Image width and height
        image_id = torch.tensor([idx])  # Unique image ID

        boxes = []
        labels = []

        for obj in ann:
            category = obj["type"]
            if category == "DontCare":
                continue  # Ignore 'DontCare' objects

            bbox = obj["bbox"]  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox

            # Ensure box is within image bounds
            xmin, xmax = max(0, xmin), min(w, xmax)
            ymin, ymax = max(0, ymin), min(h, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue  # Ignore invalid boxes

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(KITTI_LABEL_MAP.get(category, 5))  # Default to 'Misc'

        # Convert to PyTorch tensors
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": image_id,
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
        }

        return target


def make_kitti_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([600, 800, 1000]),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f"Unknown image_set: {image_set}")


def build(image_set, args):
    root = Path(args.kitti_path)
    assert root.exists(), f"Provided KITTI path {root} does not exist"
    if image_set == "train":
        training_flag = True
    else:
        training_flag = False
    dataset = KittiDetection(root=root, training_flag=training_flag, transforms=make_kitti_transforms(image_set))
    return dataset
