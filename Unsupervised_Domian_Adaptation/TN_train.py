import argparse
from torch.autograd import Variable
import torch.optim as optim
import os.path as osp
from module.Encoder import Deeplabv2
from module.Discriminator import FCDiscriminator
from data.loveda import LoveDALoader
from ever.core.iterator import Iterator
from utils.tools import *
from tqdm import tqdm
from eval import evaluate
from torch.nn.utils import clip_grad
from module.trans_norm import TransNorm2d

parser = argparse.ArgumentParser(description='Run AdaptSeg methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)


def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='TN', logdir=cfg.SNAPSHOT_DIR)

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
            norm_layer=TransNorm2d,
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
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)
    # init D
    model_D1 = FCDiscriminator(cfg.NUM_CLASSES)
    model_D2 = FCDiscriminator(cfg.NUM_CLASSES)

    model_D1.train()
    model_D1.cuda()

    model_D2.train()
    model_D2.cuda()
    count_model_parameters(model, logger)
    count_model_parameters(model_D1, logger)
    count_model_parameters(model_D2, logger)

    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    targetloader_iter = Iterator(targetloader)

    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)


    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    source_label = 0
    target_label = 1

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        G_lr = adjust_learning_rate(optimizer, i_iter, cfg)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        D_lr = adjust_learning_rate_D(optimizer_D1, i_iter, cfg)
        adjust_learning_rate_D(optimizer_D2, i_iter, cfg)

        for sub_i in range(cfg.ITER_SIZE):
            # train G
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False
            # train with source

            batch = trainloader_iter.next()
            images, labels = batch[0]
            src_images = Variable(images).cuda()
            batch = targetloader_iter.next()
            images, _ = batch[0]
            tar_images = Variable(images).cuda()
            bs = src_images.shape[0]

            pred_blob1, pred_blob2 = model(torch.cat([src_images, tar_images], dim=0))
            pred1, pred2 = pred_blob1[:bs], pred_blob2[:bs]
            pred_target1, pred_target2 = pred_blob1[bs:], pred_blob2[bs:]
            # pred1, pred2 = model(images)


            loss_seg1 = loss_calc(pred1, labels['cls'].cuda())
            loss_seg2 = loss_calc(pred2, labels['cls'].cuda())
            loss = loss_seg2 + cfg.LAMBDA_SEG * loss_seg1

            # proper normalization
            loss = loss / cfg.ITER_SIZE
            loss.backward(retain_graph=True)
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / cfg.ITER_SIZE
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / cfg.ITER_SIZE

            # train with target


            # pred_target1, pred_target2 = model(images)

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            loss_adv_target1 = bce_loss(D_out1,
                                        Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

            loss = cfg.LAMBDA_ADV_TARGET1  * loss_adv_target1 + cfg.LAMBDA_ADV_TARGET2 * loss_adv_target2
            loss = loss / cfg.ITER_SIZE
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / cfg.ITER_SIZE
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / cfg.ITER_SIZE

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            D_out1 = model_D1(F.softmax(pred1))
            D_out2 = model_D2(F.softmax(pred2))

            loss_D1 = bce_loss(D_out1,
                               Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

            loss_D1 = loss_D1 / cfg.ITER_SIZE / 2
            loss_D2 = loss_D2 / cfg.ITER_SIZE / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            loss_D1 = bce_loss(D_out1,
                               Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())

            loss_D1 = loss_D1 / cfg.ITER_SIZE / 2
            loss_D2 = loss_D2 / cfg.ITER_SIZE / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D1.parameters()), max_norm=35, norm_type=2)
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D2.parameters()), max_norm=35, norm_type=2)
        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info(
                'i_iter = %d, loss_seg1 = %.3f loss_seg2 = %.3f loss_adv1 = %.3f, loss_adv2 = %.3f loss_D1 = %.3f loss_D2 = %.3f G_lr = %.3f D_lr = %.5f' % (
                    i_iter, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, G_lr, D_lr)
            )

        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model_D1.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '_D2.pth'))
            evaluate(model, cfg, True, ckpt_path=ckpt_path, logger=logger)
            break

        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model_D1.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '_D2.pth'))
            evaluate(model, cfg, True, ckpt_path=ckpt_path, logger=logger)
            model.train()


if __name__ == '__main__':
    seed_torch(2333)
    main()
