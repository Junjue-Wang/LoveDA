from collections import OrderedDict
import os.path as osp
# from module.CLAN_G import Deeplabv2
from module.Encoder import Deeplabv2
from module.Discriminator import PixelDiscriminator
from data.loveda import LoveDALoader
from ever.core.iterator import Iterator
from utils.tools import *
import argparse
from tqdm import tqdm
from torch.nn.utils import clip_grad
import torch.nn.functional as F
from eval import evaluate




parser = argparse.ArgumentParser(description='Run CLAN methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)



def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))



def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='FADA', logdir=cfg.SNAPSHOT_DIR)

    # model = Deeplabv2(
    #     dict(
    #     backbone=dict(
    #         resnet_type='resnet50',
    #         output_stride=16,
    #         pretrained=True,
    #         multi_layer=False,
    #         cascade=False
    #     )))
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=False,
        cascade=False,
        use_ppm=False,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7
    ))
    model.train()
    model.cuda()
    # Init D
    model_D = PixelDiscriminator(1024, num_classes=7)

    model_D.train()
    model_D.cuda()
    count_model_parameters(model, logger)
    count_model_parameters(model_D, logger)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()

    
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()


    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    targetloader_iter = Iterator(targetloader)

    logger.info("Start training")
    model.train()
    model_D.train()
    # for i, ((src_input, src_label, src_name), (tgt_input, _, _)) in enumerate(zip(src_train_loader, tgt_train_loader)):
    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        optimizer_D.zero_grad()
        lr_D = adjust_learning_rate_D(optimizer_D, i_iter, cfg)

        batch = trainloader_iter.next()
        src_input, src_label = batch[0]
        b, c, h, w = src_input.shape
        batch = targetloader_iter.next()
        tar_input, _ = batch[0]

        #Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        source_pred, src_feat = model(src_input.cuda())
        temperature = 1.8
        source_pred = source_pred.div(temperature)
        loss_seg = loss_calc(source_pred, src_label['cls'].cuda()) * cfg.LAMBDA_SEG
        loss_seg.backward()

        # generate soft labels
        src_soft_label = F.softmax(source_pred, dim=1).detach()
        src_soft_label[src_soft_label>0.9] = 0.9
        
        tar_pred, tar_feat = model(tar_input.cuda())
        tar_pred = tar_pred.div(temperature)
        tar_soft_label = F.softmax(tar_pred, dim=1)

        tar_soft_label = tar_soft_label.detach()
        tar_soft_label[tar_soft_label>0.9] = 0.9
        
        tar_D_pred = model_D(tar_feat)
        tar_D_pred = F.interpolate(tar_D_pred, size=(h, w), mode='bilinear', align_corners=True)
        loss_adv_tgt = cfg.LAMBDA_ADV * soft_label_cross_entropy(tar_D_pred, torch.cat((tar_soft_label, torch.zeros_like(tar_soft_label)), dim=1))
        loss_adv_tgt.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        optimizer.step()

        
        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True
        optimizer_D.zero_grad()

        src_D_pred = model_D(src_feat.detach())
        src_D_pred = F.interpolate(src_D_pred, size=(h, w), mode='bilinear', align_corners=True)
        loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
        loss_D_src.backward()

        tgt_D_pred = model_D(tar_feat.detach())
        tgt_D_pred = F.interpolate(tgt_D_pred, size=(h, w), mode='bilinear', align_corners=True)
        loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tar_soft_label), tar_soft_label), dim=1))
        loss_D_tgt.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D.parameters()), max_norm=35, norm_type=2)
        optimizer_D.step()



        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            text = 'i_iter = %d, loss_seg = %.4f loss_adv = %.4f, loss_D_s = %.4f loss_D_t = %.4f G_lr = %.5f D_lr = %.5f' % (
                i_iter, loss_seg, loss_adv_tgt, loss_D_src, loss_D_tgt, lr, lr_D)
            logger.info(text)

        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '_D.pth'))
            evaluate(model, cfg, True, ckpt_path, logger)
            break

        if i_iter % cfg.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '_D.pth'))
            evaluate(model, cfg, True, ckpt_path, logger)
            model.train()



if __name__ == "__main__":
    seed_torch(2333)
    main()
