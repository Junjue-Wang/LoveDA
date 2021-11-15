import torch
from torch.autograd import Variable
import torch.optim as optim
import os.path as osp
# from module.CLAN_G import Deeplabv2
from module.Encoder import Deeplabv2
from module.Discriminator import FCDiscriminator
from module.loss import WeightedBCEWithLogitsLoss
from data.loveda import LoveDALoader
from ever.core.iterator import Iterator
from utils.tools import *
import argparse
from tqdm import tqdm
from torch.nn.utils import clip_grad
import torch.nn.functional as F
from eval import evaluate

Lambda_weight = 0.01
Lambda_local = 10
Epsilon = 0.4


parser = argparse.ArgumentParser(description='Run CLAN methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)




def weightmap(pred1, pred2):
    b, c, h, w = pred1.shape
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(b, 1, h, w) / \
    (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(b, 1, h, w)
    return output


def main():
    """Create the model and start the training."""
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='CLAN', logdir=cfg.SNAPSHOT_DIR)

    # Create Network
    # model = Deeplabv2(dict(
    #     backbone=dict(
    #         resnet_type='resnet50',
    #         output_stride=16,
    #         pretrained=True,
    #         multi_layer=True,
    #         cascade=False
    #         )
    # ))
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
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
    model_D = FCDiscriminator(7)
# =============================================================================
#    #for retrain     
#    saved_state_dict_D = torch.load(RESTORE_FROM_D)
#    model_D.load_state_dict(saved_state_dict_D)
# =============================================================================
    model_D.train()
    model_D.cuda()
    count_model_parameters(model, logger)
    count_model_parameters(model_D, logger)

    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    targetloader_iter = Iterator(targetloader)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    
    
    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    # interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    # interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    
    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        optimizer_D.zero_grad()
        lr_D = adjust_learning_rate_D(optimizer_D, i_iter, cfg)
        
        damping = (1 - i_iter/cfg.NUM_STEPS_STOP)

        #======================================================================================
        # train G
        #======================================================================================

        #Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # Train with Source
        batch = trainloader_iter.next()
        images_s, labels_s = batch[0]

        pred_source1, pred_source2 = model(images_s.cuda())
		
        #Segmentation Loss
        loss_seg = (loss_calc(pred_source1, labels_s['cls'].cuda()) + loss_calc(pred_source2, labels_s['cls'].cuda())) 
        #loss_seg = loss_calc(pred_source1, labels_s['cls'].cuda()) * cfg.LAMBDA_SEG
        loss_seg.backward()

        # Train with Target
        batch = targetloader_iter.next()
        images_t, _ = batch[0]

        pred_target1, pred_target2 = model(images_t.cuda())


        weight_map = weightmap(F.softmax(pred_target1, dim = 1), F.softmax(pred_target2, dim = 1))
        
        D_out = model_D(F.softmax(pred_target1 + pred_target2, dim = 1))
        D_out = F.interpolate(D_out, scale_factor=32, mode='bilinear', align_corners=True)
        #Adaptive Adversarial Loss
        if(i_iter > cfg.PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out, 
                                    Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(), weight_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out,
                          Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

        loss_adv = loss_adv * cfg.Lambda_adv * damping
        loss_adv.backward()
        
        #Weight Discrepancy Loss
        W5 = None
        W6 = None

        for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
            if W5 is None and W6 is None:
                W5 = w5.view(-1)
                W6 = w6.view(-1)
            else:
                W5 = torch.cat((W5, w5.view(-1)), 0)
                W6 = torch.cat((W6, w6.view(-1)), 0)
        
        loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1) # +1 is for a positive loss
        loss_weight = loss_weight * Lambda_weight * damping * 2
        loss_weight.backward()
        
        #======================================================================================
        # train D
        #======================================================================================
        
        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True
            
        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()
        
        D_out_s = model_D(F.softmax(pred_source1 + pred_source2, dim = 1))

        loss_D_s = bce_loss(D_out_s,
                          Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda()) * 0.5

        loss_D_s.backward()

        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        weight_map = weight_map.detach()
        
        D_out_t = model_D(F.softmax(pred_target1 + pred_target2, dim = 1))
        D_out_t = F.interpolate(D_out_t, scale_factor=32, mode='bilinear', align_corners=True)
        #Adaptive Adversarial Loss
        if(i_iter > cfg.PREHEAT_STEPS):
            loss_D_t = weighted_bce_loss(D_out_t, 
                                    Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(), weight_map, Epsilon, Lambda_local) * 0.5
        else:
            loss_D_t = bce_loss(D_out_t,
                          Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda()) * 0.5
            
        loss_D_t.backward()

        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D.parameters()), max_norm=35, norm_type=2)
        optimizer.step()
        optimizer_D.step()

        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            text = 'iter = %d, loss_seg = %.4f loss_adv = %.4f, loss_weight = %.4f, loss_D_s = %.4f loss_D_t = %.4f G_lr = %.5f D_lr = %.5f' % (
                i_iter, loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t, lr, lr_D)
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

if __name__ == '__main__':
    seed_torch(2333)
    main()
