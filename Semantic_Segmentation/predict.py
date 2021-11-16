import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import numpy as np
from tqdm import tqdm
import logging
from ever.core.checkpoint import remove_module_prefix
from ever.core.config import import_config
from train import seed_torch
import argparse
from skimage.io import imsave
import os

parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/hrnetw32.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline.hrnetw32')
parser.add_argument('--out_dir',  type=str,
                    help='out dir', default='./out')
args = parser.parse_args()

seed_torch(2333)
logger = logging.getLogger(__name__)

er.registry.register_all()


def predict_test(ckpt_path, config_path='base.hrnetw32', save_dir=''):
    os.makedirs(save_dir, exist_ok=True)
    cfg = import_config(config_path)
    statedict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model_state_dict = remove_module_prefix(statedict)
    print('Load model!')
    test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.cuda()
            pred = model(img)
            pred = pred.argmax(dim=1).cpu()
            for clsmap, imname in zip(pred, gt['fname']):
                imsave(os.path.join(save_dir, imname), clsmap.cpu().numpy().astype(np.uint8))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    predict_test(args.ckpt_path, args.config_path, args.out_dir)
