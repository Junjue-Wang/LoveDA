'''
define the convolutinal gaussian blur
define the softmax loss

'''
from data.loveda import LoveDALoader
from ever.core.iterator import Iterator
import argparse
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
import pandas as pd
from utils.tools import *
from module.Encoder import Deeplabv2
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils import clip_grad
import os.path as osp
import torch
from eval import evaluate

parser = argparse.ArgumentParser(description='Run CLAN methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)


def main():
    """Create the model and start the training."""
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='PyCDA', logdir=cfg.SNAPSHOT_DIR)
    
    # Create network

    model = Deeplabv2(
        dict(
        backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
        ),
        multi_layer=False,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
            norm_layer=nn.BatchNorm2d,
            
        ),
        inchannels=2048,
        num_classes=7
    ))
    model.train()
    model.cuda()
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)
    

    count_model_parameters(model, logger)

    weighted_softmax = pd.read_csv("./weighted_loss.txt", header=None)
    weighted_softmax = weighted_softmax.values
    weighted_softmax = torch.from_numpy(weighted_softmax)
    weighted_softmax = weighted_softmax / torch.sum(weighted_softmax)
    weighted_softmax = weighted_softmax.cuda().float()

    model.train()

    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    targetloader_iter = Iterator(targetloader)

    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)


    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()


    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        lr = adjust_learning_rate(optimizer, i_iter, cfg)
        batch = trainloader_iter.next()
        images, labels = batch[0]
        src_images = Variable(images).cuda()
        batch = targetloader_iter.next()
        images, _ = batch[0]
        tar_images = Variable(images).cuda()
        B, C, H, W = tar_images.shape

        feat, aux_feat = model(src_images)
        loss_seg1 = loss_calc(feat, labels['cls'].cuda())
        #loss_seg2 = loss_calc(aux_feat, labels['cls'].cuda())

        #loss = cfg.LAMBDA_TRADE_OFF*(loss_seg2+cfg.LAMBDA_SEG * loss_seg1)
        loss = cfg.LAMBDA_TRADE_OFF* loss_seg1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fake_labels = torch.zeros((B, H, W), dtype=torch.long).cuda()

        # images = Variable(images).cuda()
        tar_results = model(tar_images)
        target_seg = tar_results[0]
        conf_tea, pseudo_label = torch.max(nn.functional.softmax(target_seg), dim=1)
        pseudo_label = pseudo_label.detach()
        # pseudo label hard
        loss_pseudo = loss_calc(target_seg, pseudo_label, reduction='none')
        fake_mask = (fake_labels!=255).float().detach()
        conf_mask = torch.gt(conf_tea, cfg.CONF_THRESHOLD).float().detach()


        loss_pseudo = loss_pseudo * conf_mask.detach() * fake_mask.detach()
        loss_pseudo = loss_pseudo.view(-1)
        loss_pseudo = loss_pseudo[loss_pseudo!=0]

        # class balance loss
        predict_class_mean = torch.mean(nn.functional.softmax(target_seg), dim=0).mean(1).mean(1)
        
        equalise_cls_loss = robust_binary_crossentropy(predict_class_mean, weighted_softmax)

        equalise_cls_loss = torch.mean(equalise_cls_loss)

        loss_bbx_att = []
        for box_idx, box_size in enumerate(cfg.BOX_SIZE):
            pooling = torch.nn.AvgPool2d(box_size)
            pooling_result_i = pooling(target_seg)
            pooling_conf_mask, pooling_pseudo = torch.max(nn.functional.softmax(pooling_result_i), dim=1)
            pooling_conf_mask = torch.gt(pooling_conf_mask, cfg.CONF_THRESHOLD).float().detach()
            fake_mask_i = pooling(fake_labels.unsqueeze(1).float())
            fake_mask_i = fake_mask_i.squeeze(1)
            fake_mask_i = (fake_mask_i!=255).float().detach()
            loss_bbx_att_i = loss_calc(pooling_result_i, pooling_pseudo, reduction='none')
            loss_bbx_att_i = loss_bbx_att_i * pooling_conf_mask * fake_mask_i
            loss_bbx_att_i = loss_bbx_att_i.view(-1)
            loss_bbx_att_i = loss_bbx_att_i[loss_bbx_att_i!=0]
            loss_bbx_att.append(loss_bbx_att_i)

        bounding_num = 0
        if len(cfg.BOX_SIZE) > 0:
            if cfg.MERGE_1X1:
                loss_bbx_att.append(loss_pseudo)
            loss_bbx_att = torch.cat(loss_bbx_att, dim=0)
            bounding_num = loss_bbx_att.size(0) / float(H*W*B)
            loss_bbx_att = torch.mean(loss_bbx_att)


        pseudo_num = loss_pseudo.size(0) / float(H*W*B)
        loss_pseudo = torch.mean(loss_pseudo)
        loss = cfg.LAMBDA_BALANCE * equalise_cls_loss
        if not cfg.MERGE_1X1:
            loss += cfg.LAMBDA_PSEUDO * loss_pseudo
        if not isinstance(loss_bbx_att, list):
            loss += cfg.LAMBDA_PSEUDO * loss_bbx_att

        optimizer.zero_grad()
        loss.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        optimizer.step()


        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info('iter = %d \t'
                        'loss_seg1 = %.3f\t'
                        #'loss_bbx_att = %s\t'
                        'loss_pseudo = %.3f\t'
                        'loss_equalise_cls = %.3f\t'
                        'bounding_num = %.3f\t'
                        'pseudo_num = %.3f\t'
                        'lr = %.3f' % (
                i_iter, loss_seg1, loss_pseudo, equalise_cls_loss, bounding_num, pseudo_num, lr))

        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path=ckpt_path, logger=logger)
            break

        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path=ckpt_path, logger=logger)
            model.train()


if __name__ == '__main__':
    seed_torch(2333)
    main()
