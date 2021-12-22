import argparse
import os.path as osp
import torch.optim as optim
from eval import evaluate
from utils.tools import *
from module.Encoder import Deeplabv2
from module.Discriminator import FCDiscriminator
from data.loveda import LoveDALoader
from utils.tools import COLOR_MAP
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()



parser = argparse.ArgumentParser(description='Run ISAT methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)

def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='ISAT', logdir=cfg.SNAPSHOT_DIR)
    save_pseudo_label_dir = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    os.makedirs(save_pseudo_label_dir, exist_ok=True)
    '''
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
            multi_layer=True,
            uselayer6=False,
        )
    )).cuda()'''
    model = Deeplabv2(dict(
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
            
            
        ),
        inchannels=2048,
        num_classes=7
    )).cuda()
    model_D = FCDiscriminator(7).cuda()


    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    evalloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    epochs = cfg.NUM_STEPS / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()
    optimizer_D = optim.Adam(model_D.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()


    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        if i_iter < cfg.WARMUP_STEP:
            # Train with Source
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_iter, cfg)
            batch = trainloader_iter.next()
            images_s, labels_s = batch[0]
            pred_source = model(images_s.cuda())
            pred_source = pred_source[0] if isinstance(pred_source, tuple) else pred_source
            # Segmentation Loss
            loss = loss_calc(pred_source, labels_s['cls'].cuda())
            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            optimizer.step()
            if i_iter % 50 == 0:
                logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                text = 'Warm-up iter = %d, loss_seg = %.3f, lr = %.3f'% (
                    i_iter, loss, lr)
                logger.info(text)
            if i_iter >= cfg.NUM_STEPS_STOP - 1:
                print('save model ...')
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                break
            if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                model.train()
        else:
            # PSEUDO learning
            if i_iter % cfg.GENERATE_PSEDO_EVERY == 0 or i_iter == cfg.WARMUP_STEP:
                pseudo_dir = os.path.join(save_pseudo_label_dir, str(i_iter))
                pseudo_pred_dir = generate_pseudo(model, evalloader, pseudo_dir, pseudo_dict=cfg.PSEIDO_DICT, logger=logger)
                target_config = cfg.TARGET_DATA_CONFIG
                target_config['mask_dir'] = [pseudo_pred_dir]
                logger.info(target_config)
                targetloader = LoveDALoader(target_config)
                targetloader_iter = Iterator(targetloader)
            model.train()
            model_D.train()
            lr = adjust_learning_rate(optimizer, i_iter, cfg)
            lr_D = adjust_learning_rate_D(optimizer_D, i_iter, cfg)

            batch = trainloader_iter.next()

            images_s, labels_s = batch[0]
            pred_source = model(images_s.cuda())
            pred_source = pred_source[0] if isinstance(pred_source, tuple) else pred_source

            batch = targetloader_iter.next()
            images_t, labels_t = batch[0]
            pred_target = model(images_t.cuda())
            pred_target = pred_target[0] if isinstance(pred_target, tuple) else pred_target


            # defaut reg_weight
            if cfg.DISCRIMINATOR['lambda_entropy_weight'] or cfg.DISCRIMINATOR['lambda_kldreg_weight']:
                reg_val_matrix = torch.ones_like(labels_t['cls']).type_as(pred_target)
                reg_val_matrix[labels_t['cls']==-1]=0
                reg_val_matrix = reg_val_matrix.unsqueeze(dim=1)
                reg_ignore_matrix = 1 - reg_val_matrix
                reg_weight = torch.ones_like(pred_target)
                reg_weight_val = reg_weight * reg_val_matrix
                reg_weight_ignore = reg_weight * reg_ignore_matrix
                del reg_ignore_matrix, reg_weight, reg_val_matrix

            loss_dict = dict()

            # forward discriminators
            s_D_logits = model_D(pred_source.softmax(dim=1).detach())
            t_D_logits = model_D(pred_target.softmax(dim=1).detach())


            is_source = torch.zeros_like(s_D_logits).cuda()
            is_target = torch.ones_like(t_D_logits).cuda()
            discriminator_loss = (bce_loss(s_D_logits, is_source) +
                                  bce_loss(t_D_logits, is_target))/2

            # adv_losses
            t_D_logits = model_D(pred_target.softmax(dim=1).detach())
            is_source = torch.zeros_like(t_D_logits).cuda()
            adv_loss = cfg.DISCRIMINATOR['weight'] * bce_loss(t_D_logits, is_source)
            loss_dict['adv_loss'] = adv_loss

            # update seg loss
            seg_loss = cfg.SOURCE_LOSS_WEIGHT * loss_calc(pred_source, labels_s['cls'].cuda())
            loss_dict['seg_loss'] = seg_loss

            # pseudo label target seg loss
            target_seg_loss = cfg.PSEUDO_LOSS_WEIGHT * loss_calc(pred_target, labels_t['cls'].cuda())
            loss_dict['target_seg_loss'] = target_seg_loss

            # entropy reg
            if cfg.DISCRIMINATOR['lambda_entropy_weight'] > 0:
                entropy_reg_loss = entropyloss(pred_target, reg_weight_ignore)
                entropy_reg_loss =  entropy_reg_loss * cfg.DISCRIMINATOR['lambda_entropy_weight']
                loss_dict['entropy_reg_loss'] = entropy_reg_loss
            # kld reg
            if cfg.DISCRIMINATOR['lambda_kldreg_weight'] > 0:
                kld_reg_loss = kldloss(pred_target, reg_weight_val)
                kld_reg_loss =  kld_reg_loss * cfg.DISCRIMINATOR['lambda_kldreg_weight']
                loss_dict['kld_reg_loss'] = kld_reg_loss

            # backward model
            optimizer.zero_grad()
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            optimizer.step()

            # backward model_D
            optimizer_D.zero_grad()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D.parameters()), max_norm=35, norm_type=2)
            discriminator_loss.backward()
            optimizer_D.step()

            if i_iter % 50 == 0:
                logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                text = 'UDA iter = %d ' % i_iter
                for k, v in loss_dict.items():
                    text += '%s = %.3f ' % (k, v)
                text += 'd_loss = %.3f ' % discriminator_loss
                text += 'lr = %.3f ' % lr
                text += 'd_lr = %.3f ' % lr_D
                logger.info(text)
            if i_iter >= cfg.NUM_STEPS_STOP - 1:
                print('save model ...')
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                break
            if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                model.train()

if __name__ == '__main__':
    seed_torch(2333)
    main()
