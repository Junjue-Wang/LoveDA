from data.loveda import LoveDALoader
from utils.tools import *
from skimage.io import imsave
import os


def predict_test(model, cfg, ckpt_path=None, save_dir='./submit_test'):
    os.makedirs(save_dir, exist_ok=True)
    seed_torch(2333)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict,  strict=True)

    count_model_parameters(model)
    model.eval()
    print(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()
            for fname, pred in zip(ret_gt['fname'], cls):
                imsave(os.path.join(save_dir, fname), pred.astype(np.uint8))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    ckpt_path = './log/CBST_2Urban.pth'
    from module.Encoder import Deeplabv2
    cfg = import_config('st.cbst.2urban')
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
            num_classes=cfg.NUM_CLASSES,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=cfg.NUM_CLASSES
    )).cuda()
    predict_test(model, cfg, ckpt_path)