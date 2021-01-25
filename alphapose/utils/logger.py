# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from alphapose.opt import opt
from alphapose.utils.transforms import get_max_pred


def board_writing(writer, loggers, iterations, dataset='Train'):
    for name, logger in loggers.items():
        writer.add_scalar(
            f'{dataset}/{name}', logger.value, iterations)
        writer.add_scalar(
            f'{dataset}_avg/{name}', logger.avg, iterations)


def debug_writing(writer, joint_map, joint_radius, joint_map_gt, joint_radius_gt, inputs, iterations):
    batch_index = 0
    input_map = torch.unsqueeze(joint_map_gt.cpu().data[batch_index], dim=1)
    output_map = torch.unsqueeze(joint_map.cpu().data[batch_index], dim=1)

    image = inputs.cpu().data[batch_index]
    image[0] += 0.406
    image[1] += 0.457
    image[2] += 0.480

    image = image.detach()

    input_image = 0.5 * image.clone()
    input_map4 = F.interpolate(input_map, scale_factor=4, mode='bilinear', align_corners=False)
    input_image[0] += 0.5 * torch.sum(input_map4, dim=0)[0]
    input_image.clamp_(0, 1)

    output_image = 0.5 * image.clone()
    output_map = F.interpolate(output_map.type(torch.float32), scale_factor=4, mode='bilinear')
    output_image[0] += (0.25 * torch.sum(output_map, dim=0)[0]).clamp_(0, 0.5)

    writer.add_image('Data/input_map', input_image, iterations)
    writer.add_image('Data/output_map', output_image, iterations)
    save_image(f"input_map_{iterations:06}", (255 * input_image.numpy().transpose([1, 2, 0])).astype(np.uint8).copy())
    save_image(f"output_map_{iterations:06}", (255 * output_image.numpy().transpose([1, 2, 0])).astype(np.uint8).copy())

    coords = []
    for map in input_map:
        if map[0].max() == 0:
            coords.append(None)
            continue
        c = np.unravel_index(map[0].argmax(), map.shape[1:])
        coords.append(c)

    keypoints_image = (255 * image.numpy().transpose([1, 2, 0])).astype(np.uint8).copy()
    for coord, radius in zip(coords, joint_radius_gt[batch_index]):
        if coord is None:
            continue
        if radius < 0:
            radius = 0
        center = 4 * coord[1], 4 * coord[0]
        keypoints_image = cv2.circle(keypoints_image, center, int(radius * image.shape[2]), (255, 0, 0), -1)

    input_keypoints = keypoints_image.astype(np.float32) / 255
    writer.add_image('Data/input_keypoints', input_keypoints, iterations, dataformats="HWC")
    save_image(f"input_keypoints_{iterations:06}", keypoints_image)

    coords = []
    for map in output_map:
        if map[0].max() == 0:
            coords.append(None)
            continue
        c = np.unravel_index(map[0].argmax(), map.shape[1:])
        coords.append(c)

    keypoints_image = (255 * image.numpy().transpose([1, 2, 0])).astype(np.uint8).copy()
    keypoints_image_2 = (255 * image.numpy().transpose([1, 2, 0])).astype(np.uint8).copy()
    for coord, radius in zip(coords, joint_radius[batch_index]):
        if coord is None:
            continue
        if radius < 0:
            radius = 0
        center = coord[1], coord[0]
        keypoints_image = cv2.circle(keypoints_image, center, int(radius * image.shape[2]), (255, 0, 0), -1)
        keypoints_image_2 = cv2.circle(keypoints_image_2, center, 3, (255, 0, 0), -1)

    output_keypoints = keypoints_image.astype(np.float32) / 255
    writer.add_image('Data/output_keypoints', output_keypoints, iterations, dataformats="HWC")
    save_image(f"output_keypoints_{iterations:06}", keypoints_image)
    save_image(f"output_keypoints_no_radius_{iterations:06}", keypoints_image_2)


def save_image(name, img):
    img_path = opt.experiment_path / f"debug_img/{name}.png"
    img_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(img_path), img[:, :, ::-1])
