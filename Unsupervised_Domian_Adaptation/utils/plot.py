from ever.core.checkpoint import load_model_state_dict_from_ckpt
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
sns.set()
save_dir = r'C:\Users\zouda\Desktop\Conference\NIPS\figs\appendix\transnorm'


# os.makedirs(save_dir, exist_ok=True)
# tn_model = torch.load(r'C:\Users\zouda\Desktop\Conference\TN2Rural.pth', map_location=lambda storage, loc: storage)
# urban_deeplabv3 = torch.load(r'C:\Users\zouda\Desktop\Conference\URBAN15000.pth', map_location=lambda storage, loc: storage)
# rural_deeplabv3 = torch.load(r'C:\Users\zouda\Desktop\Conference\RURAL15000.pth', map_location=lambda storage, loc: storage)
#
# extractor = 'running_mean'
# for k, v in tn_model.items():
#     # if 'resnet' in k and 'running_mean' in k:
#     #     if 'target' in k:
#     #         sns.distplot(v.reshape(-1), hist=False, kde=True, color='r', label='TN_Rural')
#     #         rural_weights = rural_deeplabv3[k.replace('_target', '')]
#     #         urban_weights = urban_deeplabv3[k.replace('_target', '')]
#     #         sns.distplot(rural_weights.reshape(-1), hist=False, kde=True, color='g',  label='Orcale_Rural')
#     #         sns.distplot(urban_weights.reshape(-1), hist=False, kde=True, color='b', label='Orcale_Urban')
#
#     if 'resnet' in k and 'running_mean' in k:
#         if 'target' in k:
#             sns.distplot(v.reshape(-1), hist=False, kde=True, color='r', label='TransNorm_Target')
#             rural_weights = rural_deeplabv3[k.replace('_target', '')]
#             urban_weights = urban_deeplabv3[k.replace('_target', '')]
#             sns.distplot(rural_weights.reshape(-1), hist=False, kde=True, color='g',  label='Orcale_Target')
#             sns.distplot(urban_weights.reshape(-1), hist=False, kde=True, color='b', label='Orcale_Source')
#
#             plt.savefig(os.path.join(save_dir, k+'.png'))
#
#             plt.clf()

import glob
pngp_list = glob.glob(os.path.join(save_dir, '*.png'))
for pngp in pngp_list:
    new_pngp = pngp.replace('.', '_').replace('_png', '.png')
    print(new_pngp)
    os.rename(pngp, new_pngp)