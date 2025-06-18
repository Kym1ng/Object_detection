# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import os # Added
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, get_kitti_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from pycocotools.coco import COCO
import cv2 # Added
import torchvision.transforms.functional as TF # Added


# read the parameters from the command line
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--kitti2coco', type=bool, default=False,
                        help="If true, means current COCO is converted from KITTI dataset to COCO format")
    parser.add_argument('--kitti_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',default=True)
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# Added: KITTI class names if kitti2coco is used
KITTI_CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

# Added: Helper function to denormalize image tensor for visualization
def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes a tensor image with mean and standard deviation."""
    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# Added: Helper function to draw predictions on an image and save it
def draw_and_save_predictions(image_tensor, predictions, output_path, class_names=None, original_size=None, confidence_threshold=0.55):
    """Draws bounding boxes and labels on an image and saves it."""
    img = denormalize_image(image_tensor.cpu())
    img_pil = TF.to_pil_image(img)

    if original_size is not None:
        original_h, original_w = original_size
        img_pil = img_pil.crop((0, 0, int(original_w), int(original_h)))

    frame = np.array(img_pil)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # PIL is RGB, OpenCV is BGR

    scores = predictions['scores']
    labels = predictions['labels']
    boxes = predictions['boxes']

    for score, label, box in zip(scores, labels, boxes):
        if score < confidence_threshold:
            continue
        
        box_coords = box.cpu().numpy().astype(int)
        x1, y1, x2, y2 = box_coords
        class_id = label.item()
        class_name = str(class_id) if class_names is None or not (0 <= class_id < len(class_names)) else class_names[class_id]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, frame)

def main(args):
    # initialize the distributed training
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # set the device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # initialize the model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # load the dataset
    model_without_ddp = model

    # print the number of parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    # load the dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # initialize the batch sampler for training
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    # initialize the data loaders for training and validation
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    if args.dataset_file == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)


    # create the output directory
    output_dir = Path(args.output_dir)

    # load the model from the checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
            state_dict = checkpoint['model']
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model']

        model_without_ddp.load_state_dict(state_dict)

    start_time = time.time()
    # evaluate the model
    if args.eval:
        # --- Visualize the first batch ---
        print("Visualizing the first batch...")
        vis_output_dir = output_dir / "visualizations_first_batch"
        vis_output_dir.mkdir(parents=True, exist_ok=True)

        # Get an iterator for the validation data_loader
        data_loader_val_iter = iter(data_loader_val)
        try:
            samples_vis, targets_vis = next(data_loader_val_iter)
        except StopIteration:
            print("Validation data loader is empty. Cannot visualize.")
            samples_vis, targets_vis = None, None

        if samples_vis is not None:
            samples_vis = samples_vis.to(device)
            targets_vis = [{k: v.to(device) for k, v in t.items()} for t in targets_vis]
            orig_target_sizes_vis = torch.stack([t["orig_size"] for t in targets_vis], dim=0)

            model.eval() # Ensure model is in eval mode
            with torch.no_grad():
                outputs_vis = model(samples_vis)
                # Assuming 'bbox' postprocessor is always available and desired for visualization
                results_vis = postprocessors['bbox'](outputs_vis, orig_target_sizes_vis)

            for i in range(samples_vis.tensors.shape[0]):
                img_tensor_vis = samples_vis.tensors[i]
                # The mask could be used: samples_vis.mask[i]
                # For cropping to original size before saving:
                original_h, original_w = orig_target_sizes_vis[i].tolist()
                
                predictions_vis = results_vis[i] # {'scores': s, 'labels': l, 'boxes': b}
                
                current_class_names = None
                if args.kitti2coco: # Check if this arg is correctly propagated or available
                    current_class_names = KITTI_CLASSES
                
                vis_path = vis_output_dir / f"visualization_batch_0_img_{i}.png"
                draw_and_save_predictions(img_tensor_vis, predictions_vis, str(vis_path), 
                                          current_class_names, original_size=(original_h, original_w),
                                          confidence_threshold=0.55) # You can adjust threshold
            
            print(f"First batch visualizations saved to {vis_output_dir}")
        
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                            data_loader_val, base_ds, device, args.output_dir)

        # save the evaluation results
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print(test_stats)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('eval time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
