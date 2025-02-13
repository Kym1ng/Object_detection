from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

# init with image_folder, ann_file, transforms, return_masks
# do the transformation and prepare like convert coco polys to mask
class Kitti(torchvision.datasets.Kitti):
    def __init__(self, img_folder, label_folder, transforms):
        super(Kitti, self).__init__(img_folder, label_folder)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(Kitti, self).__getitem__(idx)
        # prepare the target
        img, target = self.Kittiprepare(img, target)
        # do the transformation
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


    def Kittiprepare(image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_kitti_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        # value from ImageNet (pretrained model), using this value implys that the kitti distribution is similar to ImageNet
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    # right now its kitti path
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    img_folder, label_folder = root / "Kitti/training/image_2", root / "Kitti/training/label_2"
    dataset = Kitti(img_folder, label_folder, transforms = make_kitti_transforms(image_set))
    return dataset
