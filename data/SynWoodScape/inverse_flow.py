import sys
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submission import get_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
import cv2
import math
import os.path as osp

from core.FlowFormer import build_flowformer

from utils.utils import InputPadder, forward_interpolate
import itertools

TRAIN_SIZE = [432, 960]


def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1]

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights


def compute_flow(model, image1, image2, weights=None):
    print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().detach().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow


def compute_inverse_optical_flow(flow):
    h, w, _ = flow.shape
    flow_inv = np.zeros_like(flow, dtype=np.float32)

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    x_new = x + flow[..., 0]
    y_new = y + flow[..., 1]

    for i in range(h):
        for j in range(w):
            x2, y2 = int(round(x_new[i, j])), int(round(y_new[i, j]))

            if 0 <= x2 < w and 0 <= y2 < h:
                flow_inv[y2, x2, 0] = -flow[i, j, 0]
                flow_inv[y2, x2, 1] = -flow[i, j, 1]

    flow_inv[..., 0] = cv2.inpaint(flow_inv[..., 0], (flow_inv[..., 0] == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    flow_inv[..., 1] = cv2.inpaint(flow_inv[..., 1], (flow_inv[..., 1] == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return flow_inv


if __name__ == '__main__':
    data_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0'
    flow_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/flow_annotations/gtLabels'
    viz_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/flow_annotations/rgbLabels'
    flow_gt_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/optical_flow/raw_data'
    flow_gt_list = sorted(glob(osp.join(flow_gt_path, '*.flo')))
    num_image = len(flow_gt_list)
    keep_size = False
    for i in range(num_image):
        image_name = osp.split(flow_gt_list[i])[1]
        flow_gt = frame_utils.read_gen(osp.join(flow_gt_path, flow_gt_list[i]))
        flow_inv = compute_inverse_optical_flow(flow_gt)
        flow_img = flow_viz.flow_to_image(flow_inv)
        cv2.imwrite(osp.join(viz_path, osp.splitext(image_name)[0]+'.png'), flow_img[:, :, [2,1,0]])
        frame_utils.writeFlow(osp.join(flow_path, osp.splitext(image_name)[0]+'.flo'), flow_inv)
