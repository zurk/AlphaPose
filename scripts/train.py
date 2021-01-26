"""Script for multi-gpu training."""
import json
import os
from pathlib import Path
from pprint import pformat
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from tqdm import tqdm

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d


def train(opt, train_loader, m, criterion, optimizer, writer, scaler):
    loggers = {
        'joint_loss': DataLogger(),
        'radius_loss': DataLogger(),
        'loss': DataLogger(),
        'acc': DataLogger(),
        'acc_radius': DataLogger(),
    }
    m.train()
    train_dataset = train_loader.dataset
    train_loader = tqdm(train_loader, dynamic_ncols=True)

    radius_loss_item = -1
    acc_radius = -1

    for i, (inps, labels, label_masks, joint_radius_gt, _, bboxes) in enumerate(train_loader):
        if isinstance(inps, list):
            if opt.device.type != 'cpu':
                inps = [inp.cuda() for inp in inps]
            inps = [inp.requires_grad_() for inp in inps]
        else:
            if opt.device.type != 'cpu':
                inps = inps.cuda()
            inps = inps.requires_grad_()
        if opt.device.type != 'cpu':
            labels = labels.cuda()
            label_masks = label_masks.cuda()
            joint_radius_gt = joint_radius_gt.cuda()
        with autocast():
            full_output = m(inps)
            joint_map = full_output['joints_map']
            joints_radius = full_output['joints_radius']

            if cfg.LOSS.get('TYPE') == 'MSELoss':
                assert criterion.reduction == "sum"
                coef = 1000
                joint_loss = 0.5 * coef * criterion(joint_map.mul(label_masks), labels.mul(label_masks))
                joint_loss /= label_masks.sum() * joint_map.shape[2] * joint_map.shape[3]
                loss = joint_loss
                if opt.fit_radius:
                    radius_masks = label_masks[:, :, 0, 0] * (joint_radius_gt != -1)
                    radius_loss = 0.5 * criterion(joint_radius_gt.mul(radius_masks),
                                                  joints_radius.mul(radius_masks))
                    joint_loss /= radius_masks.sum()
                    loss += radius_loss
                    radius_loss_item = radius_loss.item()
                    acc_radius = ((joint_radius_gt.mul(radius_masks) - joints_radius.mul(radius_masks)) < 1).sum() / (
                            joint_radius_gt.shape[0] * joint_radius_gt.shape[0])
                acc = calc_accuracy(joint_map.mul(label_masks), labels.mul(label_masks))
            else:
                raise NotImplementedError()
                loss = criterion(joint_map, labels, label_masks)
                acc = calc_integral_accuracy(joint_map, labels, label_masks, output_3d=False, norm_type=norm_type)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loggers["joint_loss"].update(joint_loss.item(), batch_size)
        loggers["radius_loss"].update(radius_loss_item, batch_size)
        loggers["loss"].update(loss.item(), batch_size)
        loggers["acc"].update(acc, batch_size)
        loggers["acc_radius"].update(acc_radius, batch_size)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        opt.trainIters += 1
        # Tensorboard
        if opt.board:
            board_writing(writer, loggers, opt.trainIters, 'Train')

        # Debug
        if opt.debug and not i % 100:
            debug_image_index = 2526
            debug_data = train_dataset[debug_image_index]
            (inps, labels, label_masks, joint_radius_gt, _, bboxes) = debug_data
            inps = inps[None, :]
            full_output = m(inps)
            joint_map = full_output['joints_map']
            joints_radius = full_output['joints_radius']

            debug_writing(
                writer, joint_map, joints_radius, labels[None, :], joint_radius_gt[None, :], inps, opt.trainIters)

        # TQDM
        train_loader.set_description(
            " | ".join(f"{name}:{logger.avg:.05f}" for name, logger in loggers.items())
        )

    train_loader.close()

    return loggers


def validate(m, opt, heatmap_to_coord, batch_size=64):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.nThreads, drop_last=False)
    kpt_json = []
    eval_joints = det_dataset.EVAL_JOINTS

    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for index, (inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds) in tqdm(enumerate(det_loader), dynamic_ncols=True):
        if opt.device.type != "cpu":
            if isinstance(inps, list):
                inps = [inp.cuda() for inp in inps]
            else:
                inps = inps.cuda()
        full_output = m(inps)
        joints_map = full_output['joints_map']

        pred = joints_map
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(joints_map.shape[0]):
            bbox = crop_bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i][det_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open(os.path.join(opt.work_dir, 'test_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_kpt.json'), ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res


def validate_gt(m, opt, cfg, heatmap_to_coord, batch_size=64):
    joint_radius_mse = DataLogger()
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.nThreads, drop_last=False)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    mse_loss = nn.MSELoss()
    for index, (inps, labels, label_masks, joint_radius_gt, img_ids, bboxes) in tqdm(enumerate(gt_val_loader), dynamic_ncols=True):
        if opt.device.type != 'cpu':
            if isinstance(inps, list):
                inps = [inp.cuda() for inp in inps]
            else:
                inps = inps.cuda()
        full_output = m(inps)
        joints_map = full_output['joints_map']
        joints_radius = full_output['joints_radius']

        pred = joints_map
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(joints_map.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i][gt_val_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

        radius_masks = (label_masks[:, :, 0, 0] != 0) & (joint_radius_gt != -1)
        joints_radius = joints_radius[:, eval_joints]
        joint_radius_gt = joint_radius_gt[:, eval_joints]
        joint_radius_gt = joint_radius_gt[radius_masks]
        joints_radius = joints_radius[radius_masks]
        joints_radius_error = mse_loss(joint_radius_gt, joints_radius.cpu())
        joint_radius_mse.update(joints_radius_error, joint_radius_gt.shape[0])

    with open(os.path.join(opt.work_dir, 'test_gt_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_gt_kpt.json'), ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return {
        "map": res,
        "radius_mse": joint_radius_mse.avg,
    }

def main():
    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    # Model Initialize
    m = preset_model(cfg)
    # todo: try to replace with distributedDataParallel to see if it is faster
    m = nn.DataParallel(m)
    if opt.device.type != 'cpu':
        m = m.cuda()

    criterion = builder.build_loss(cfg.LOSS)
    if opt.device.type != 'cpu':
        criterion=criterion.cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    if opt.clean:
        if opt.tensorboard_path.exists():
            shutil.rmtree(opt.tensorboard_path)
        if opt.experiment_path.exists():
            shutil.rmtree(opt.experiment_path)
    opt.tensorboard_path.mkdir(exist_ok=True, parents=True)
    opt.experiment_path.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(str(opt.tensorboard_path))

    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * max(1, num_gpu), shuffle=True, num_workers=opt.nThreads)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    opt.trainIters = 0

    scaler = GradScaler()

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loggers = train(opt, train_loader, m, criterion, optimizer, writer, scaler)
        logger.info(f'Train-{opt.epoch:d} epoch | '
                    f'{" | ".join(f"{name}:{l.avg:.07f}" for name, l in loggers.items())}')

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            torch.save(m.module.state_dict(), str(opt.experiment_path / f'model_{opt.epoch}.pth'))
            # Prediction Test
            with torch.no_grad():
                metrics_on_true_box = validate_gt(m.module, opt, cfg, heatmap_to_coord)
                gt_AP = metrics_on_true_box["map"]
                gt_radius_mse = metrics_on_true_box["radius_mse"]
                rcnn_AP = validate(m.module, opt, heatmap_to_coord)
                logger.info(f'##### Epoch {opt.epoch} | '
                            f'gt mAP: {gt_AP} | '
                            f'rcnn mAP: {rcnn_AP} | '
                            f'gt radius_mse {gt_radius_mse}'
                            f' #####')

            writer.add_scalar(f'Validation/mAP_on_gt_box', gt_AP, opt.trainIters)
            writer.add_scalar(f'Validation/mAP_on_pred_box', rcnn_AP, opt.trainIters)
            writer.add_scalar(f'Validation/radius_mse_on_gt_box', gt_radius_mse, opt.trainIters)

        # Time to add DPG
        if i == cfg.TRAIN.DPG_MILESTONE:
            torch.save(m.module.state_dict(), str(opt.experiment_path / "final.pth"))
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.TRAIN.LR
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
            # Reset dataset
            train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, dpg=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * max(1, num_gpu), shuffle=True, num_workers=opt.nThreads)

    torch.save(m.module.state_dict(), str(opt.experiment_path /'final_DPG.pth'))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED), strict=False)
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":
    main()
