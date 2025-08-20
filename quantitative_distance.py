"""
Qualitative test script of distance estimation for OmniDet.

# usage: ./qualitative_distance.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from matplotlib.colors import ListedColormap
from torchvision import transforms

from main import collect_args
from models.normnet_decoder import NormDecoder
from models.resnet import ResnetEncoder
from utils import Tupperware

FRAME_RATE = 1


def scale_norm(norm, min_distance, max_distance):
    """Convert network's sigmoid output into distance prediction"""
    return min_distance + max_distance * norm


def pre_image_op(args, index, frame_index, cam_side):
    cropped_coords = dict(FV=(0, 0, 1280, 920),
                          MVL=(0, 0, 1280, 960),
                          MVR=(0, 0, 1280, 960),
                          RV=(0, 0, 1280, 792))
    if args.crop:
        cropped_coords = cropped_coords[cam_side]
    else:
        cropped_coords = None
    cropped_image = get_image(args, index, cropped_coords, frame_index, cam_side)
    return cropped_image, cropped_coords


def get_image(args, index, cropped_coords, frame_index, cam_side):
    recording_folder = "rgb_images" if index == 0 else "previous_images"
    file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
    path = os.path.join(args.dataset_dir, recording_folder, file)
    image = Image.open(path).convert('RGB')
    if args.crop:
        return image.crop(cropped_coords)
    return image


def get_mask(args, cropped_coords, frame_index, cam_side):
    path = os.path.join(args.dataset_dir, "masks", f"Car3_{cam_side}.png")
    image = Image.open(path).convert('L')
    if args.crop:
        return image.crop(cropped_coords)
    return image


def pre_dis_op(args, frame_index, cam_side):
    gt_path = os.path.join(args.dataset_dir, 'raw_data', f"{frame_index}_{cam_side}.npy")
    dis_gt = np.load(gt_path)
    cropped_coords = dict(FV=(0, 0, 1280, 920),
                          MVL=(0, 0, 1280, 960),
                          MVR=(0, 0, 1280, 960),
                          RV=(0, 0, 1280, 792))
    if args.crop:
        cropped_coords = cropped_coords[cam_side]
    else:
        cropped_coords = None
    dis_gt_crop = dis_gt[cropped_coords[1]:cropped_coords[3], cropped_coords[0]:cropped_coords[2]]
    return dis_gt_crop


def metric(dis_gt, dis_pred, mask):
    mask = mask*(dis_gt < 100)*(dis_gt > 0.1).astype(int)
    abs_rel = np.sum(np.abs(dis_gt-dis_pred)/dis_gt*mask)/np.sum(mask)
    sq_rel = np.sum((dis_gt-dis_pred)**2/dis_gt*mask)/np.sum(mask)
    rmse = np.sqrt(np.sum((dis_gt-dis_pred)**2*mask)/np.sum(mask))
    rmse_log = np.sqrt(np.sum((np.log(dis_gt) - np.log(dis_pred)) ** 2 * mask) / np.sum(mask))
    delta1 = np.sum(mask*(np.maximum(dis_gt/dis_pred, dis_pred/dis_gt) < 1.25))/np.sum(mask)
    delta2 = np.sum(mask*(np.maximum(dis_gt/dis_pred, dis_pred/dis_gt) < (1.25**2)))/np.sum(mask)
    delta3 = np.sum(mask*(np.maximum(dis_gt/dis_pred, dis_pred/dis_gt) < (1.25**3)))/np.sum(mask)
    return [abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3]


@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    depth_decoder_path = os.path.join(args.pretrained_weights, "norm.pth")

    print("=> Loading pretrained encoder")
    encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
    loaded_dict_enc = torch.load(encoder_path, map_location=args.device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()

    print("=> Loading pretrained decoder")
    decoder = NormDecoder(encoder.num_ch_enc).to(args.device)
    loaded_dict = torch.load(depth_decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    image_paths = [line.rstrip('\n') for line in open(args.val_file)]
    print(f"=> Predicting on {len(image_paths)} validation images")

    distances = list()
    metrics = list()
    for idx, image_path in enumerate(image_paths[205:]):
        frame_index, cam_side = image_path.split('.')[0].split('_')
        input_image, cropped_coords = pre_image_op(args, 0, frame_index, cam_side)
        input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        dis_gt = pre_dis_op(args, frame_index, cam_side)
        dis_gt = cv2.resize(dis_gt, (feed_width, feed_height))
        #input_image = Image.open("D:/Projects/Paper12/source_files/imgs/target.png").convert('RGB')
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        if args.input_mask:
            mask = get_mask(args, cropped_coords, frame_index, cam_side)
            mask = mask.resize((feed_width, feed_height), Image.NEAREST)
            mask = transforms.ToTensor()(mask).unsqueeze(0)
            #input_image = input_image*mask
            mask = mask.squeeze().cpu().numpy()
        else:
            mask = np.ones([feed_height, feed_width], dtype=np.uint8)


        # PREDICTION
        input_image = input_image.to(args.device)
        features = encoder(input_image)
        outputs = decoder(features)
        norm = outputs[("norm", 0)]
        inv_norm = 1 / norm

        # Saving numpy file
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        scaled_dist = scale_norm(norm, 0.1, 100)
        dis_pred = scaled_dist.squeeze().cpu().numpy()
        distances.append(scaled_dist.cpu().numpy())
        metrics.append(metric(dis_gt, dis_pred, mask))
    print(np.sum(np.array(metrics), axis=0)/len(metrics))


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)