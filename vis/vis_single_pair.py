import torch
from torchvision.transforms import ToTensor, Compose
import numpy as np
import argparse
import os
import sys

sys.path.append("../")  # Add the project directory
import cv2
import matplotlib
import matplotlib.pyplot as plt

from src.config.default import get_cfg_defaults
from src.lgfctr import LGFCTR

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=300, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(2, 1, figsize=(6, 9), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=2)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=2)

    # put txts
    txt_color = 'k' if img0[:100, :200, :3].mean() > 150 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='visualize LGFCTR single pair of images')
    parser.add_argument('--img_path0', type=str, default='sample1.png')
    parser.add_argument('--img_path1', type=str, default='sample2.png')
    parser.add_argument('--save_dir', type=str, default='fig_single_pair')
    parser.add_argument('--topk', type=int, default=1000)
    parser.add_argument('--img_resize', type=int, default=640)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--is_original', type=str, default=False,
                        help='whether output original image')
    parser.add_argument('--ckpt_path', type=str,
                        default="weights/model.ckpt")
    parser.add_argument('--main_cfg_path', type=str,
                        default='../configs/lgfctr/outdoor/lgfctr_ds_eval.py',
                        help='main config path')
    args = parser.parse_args()

    # config initialization
    img_path0, img_path1 = args.img_path0, args.img_path1
    save_dir = args.save_dir
    topk, img_resize, dpi = args.topk, args.img_resize, args.dpi

    # LGFCTR initialization
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    matcher = LGFCTR(config['LGFCTR']).to('cuda')
    matcher.eval()
    state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict, strict=True)
    transform = Compose([ToTensor()])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img0, img1 = cv2.imread(img_path0), cv2.imread(img_path1)
    f0 = f1 = 1
    if img_resize > 0:
        f0 = img_resize / max(img0.shape)
        f1 = img_resize / max(img1.shape)
        img0 = cv2.resize(img0, None, fx=f0, fy=f0)
        img1 = cv2.resize(img1, None, fx=f1, fy=f1)
    tensor0 = transform(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)).unsqueeze(0).to('cuda')
    tensor1 = transform(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)).unsqueeze(0).to('cuda')
    batch = {'image0': tensor0, 'image1': tensor1}
    matcher(batch)

    source_points = batch['mkpts0_f'].cpu().numpy()
    target_points = batch['mkpts1_f'].cpu().numpy()
    scores = batch['mconf'].cpu().numpy()
    if topk > 0:
        ids = np.argsort(scores)[-topk:]
        source_points = source_points[ids, :]
        target_points = target_points[ids, :]

    make_matching_figure(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
                         source_points, target_points, ['g']*len(source_points),
                         dpi=dpi, path=os.path.join(save_dir, f'{img_path0}-{img_path1}'))
    if args.is_original:
        cv2.imwrite(os.path.join(save_dir, img_path0), img0)
        cv2.imwrite(os.path.join(save_dir, img_path1), img1)
