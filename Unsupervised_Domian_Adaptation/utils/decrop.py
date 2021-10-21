import os
import glob
import random
from skimage.io import imread,imsave
import numpy as np
# root_dir = r'J:\2021LoveDA\Test\Urban'
#
#
# imagep_list = glob.glob(os.path.join(root_dir, 'images_png', '*png'))
#
#
# maskp_list = []
# for imagep in imagep_list:
#     maskp = os.path.join(root_dir, 'masks_png', os.path.basename(imagep))
#     maskp_list.append(maskp)
#
#
# count_idx = 5167
# for imagep, maskp in zip(imagep_list, maskp_list):
#     image_dir = os.path.dirname(imagep)
#     mask_dir = os.path.dirname(maskp)
#     name = int(os.path.basename(imagep).split('.')[0])
#     print(os.path.join(image_dir, str(count_idx + name)+'.png'))
#     os.rename(imagep, os.path.join(image_dir, str(count_idx + name)+'.png'))
#     os.rename(maskp, os.path.join(mask_dir, str(count_idx + name)+'.png'))


root_dir = r'J:\2021LoveDA\LoveDA_Test'
maskp_list = glob.glob(os.path.join(root_dir, '*png'))
for maskp in maskp_list:
    mask = imread(maskp).astype(np.float) - 1
    mask[mask==-1] = 0
    imsave(os.path.join(r'J:\2021LoveDA\LoveDA_Sub', os.path.basename(maskp)), mask.astype(np.uint8))

