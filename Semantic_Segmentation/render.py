from PIL import Image
import numpy as np

COLOR_MAP = dict(
    IGNORE=(0, 0, 0),
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)


def render(mask_path, vis_path):
    new_mask = np.array(Image.open(mask_path)).astype(np.uint8)
    cm = np.array(list(COLOR_MAP.values())).astype(np.uint8)
    color_img = cm[new_mask]
    color_img = Image.fromarray(np.uint8(color_img))
    color_img.save(vis_path)


if __name__ == '__main__':
    mask_path = r'C:\Users\86158\Desktop\Wujin_9_6.png'
    vis_path = r'C:\Users\86158\Desktop\Wujin_9_6_vis.png'
    render(mask_path, vis_path)