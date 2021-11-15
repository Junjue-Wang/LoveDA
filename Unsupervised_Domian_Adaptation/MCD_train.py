import argparse
import torch
import torch.optim as optim
import os.path as osp
from module.Deeplabv2_MMD import Deeplabv2_MMD
from data.loveda import LoveDALoader
from utils.tools import *
from ever.core.iterator import Iterator
from eval import evaluate
from tqdm import tqdm
from torch.nn.utils import clip_grad
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser(description='Run MCD methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

def main():
    """Create the model and start the training."""
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='MMD', logdir=cfg.SNAPSHOT_DIR)
    cudnn.enabled = True
    # Create Network
    model = Deeplabv2_MMD(dict(num_classes=7)).cuda()
    count_model_parameters(model, logger)

    model.train()
    model.cuda()

    cudnn.benchmark = True

    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    targetloader_iter = Iterator(targetloader)
    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    optimizer.zero_grad()
    logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        # Train with Source
        batch = trainloader_iter.next()
        images_s, labels_s = batch[0]
        batch = targetloader_iter.next()
        images_t, _ = batch[0]

        pred_source, feat_source =  model(images_s.cuda())
        _, feat_target = model(images_t.cuda())

        mmd_loss = mmd_linear(feat_source, feat_target) * cfg.MMD_SCALER

        #Segmentation Loss
        loss_seg = loss_calc(pred_source, labels_s['cls'].cuda())
        loss = (loss_seg + mmd_loss)
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        loss.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        optimizer.step()
        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            text = 'i_iter = %d, loss_seg = %.4f, mmd_loss = %.4f, lr = %.3f' % (
                i_iter,loss_seg, mmd_loss, lr)
            logger.info(text)

        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            break

        if i_iter % cfg.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            model.train()

if __name__ == '__main__':
    seed_torch(2333)
    main()
