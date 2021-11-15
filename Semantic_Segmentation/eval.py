import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import os
from data.loveda import COLOR_MAP
from tqdm import tqdm
from module.tta import tta, Scale
from module.viz import VisualizeSegmm
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import numpy as np
from train import seed_torch

import argparse

parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline.deeplabv3p')
parser.add_argument('--tta',  type=bool,
                    help='use tta', default=False)
args = parser.parse_args()

seed_torch(2333)
logger = logging.getLogger(__name__)

er.registry.register_all()


def evaluate(ckpt_path, config_path='base.hrnetw32', use_tta=False):
    cfg = import_config(config_path)
    model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)

    log_dir = os.path.dirname(ckpt_path)
    test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    metric_op = er.metric.PixelMetric(7, logdir=log_dir, logger=logger)
    vis_dir = os.path.join(log_dir, 'vis-{}'.format(os.path.basename(ckpt_path)))
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.cuda()
            y_true = gt['cls']
            y_true = y_true.cpu()
            if use_tta:
                pred = tta(model, img, tta_config=[
                    Scale(scale_factor=0.5),
                    Scale(scale_factor=0.75),
                    Scale(scale_factor=1.0),
                    Scale(scale_factor=1.25),
                    Scale(scale_factor=1.5),
                    Scale(scale_factor=1.75),
                ])
            else:
                pred = model(img)
            pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], pred[valid_inds])

            for clsmap, imname in zip(pred, gt['fname']):
                viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
    metric_op.summary_all()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # ckpt_path = './log/deeplabv3p.pth'
    # config_path = 'baseline.deeplabv3p'
    evaluate(args.ckpt_path, args.config_path, args.tta)
