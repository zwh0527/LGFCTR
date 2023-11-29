import torch
from torchvision.transforms import ToTensor, Compose
import numpy as np
import argparse
import os
import sys

sys.path.append("../")  # Add the project directory
import cv2
import matplotlib.pyplot as plt

from src.config.default import get_cfg_defaults
from src.lgfctr import LGFCTR_vis

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='visualize LGFCTR single pair of images')
    parser.add_argument('--img_path0', type=str, default='sample1.png',
                        help='the first image path')
    parser.add_argument('--img_path1', type=str, default='sample2.png',
                        help='the second image path')
    parser.add_argument('--save_dir', type=str, default='fig_attention',
                        help='which directory to save generating figures')
    parser.add_argument('--img_resize', type=int, default=480,
                        help='480 for 24GB RAM')
    parser.add_argument('--dpi', type=int, default=300,
                        help='the dpi of generating figures')
    parser.add_argument('--is_original', type=bool, default=False,
                        help='whether output original image')
    parser.add_argument('--ckpt_path', type=str,
                        default="weights/model.ckpt")
    parser.add_argument('--main_cfg_path', type=str,
                        default='../configs/lgfctr/outdoor/lgfctr_ds_eval.py',
                        help='main config path')
    parser.add_argument('--cmap', type=str, default='jet',
                        help='colormap of generating attention weights')

    args = parser.parse_args()

    # choose the index of resolution and CTRs to show attention weights
    resolution_idxs, layer_idxs = [0, 5], [2]

    # config initialization
    img_path0, img_path1 = args.img_path0, args.img_path1
    img_resize, dpi = args.img_resize, args.dpi
    flag = True
    for resolution_idx in resolution_idxs:
        for layer_idx in layer_idxs:
            save_dir = os.path.join(args.save_dir, f'{img_path0}-{img_path1}-{resolution_idx}-{layer_idx}')
            npy_path0 = os.path.join(save_dir, f'{img_path0}-{resolution_idx}-{layer_idx}.npy')
            npy_path1 = os.path.join(save_dir, f'{img_path1}-{resolution_idx}-{layer_idx}.npy')
            if os.path.exists(npy_path0) and os.path.exists(npy_path1):
                attention_vis0 = np.load(npy_path0, allow_pickle=True)
                attention_vis1 = np.load(npy_path1, allow_pickle=True)
            else:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if flag:
                    # LGFCTR initialization
                    config = get_cfg_defaults()
                    config.merge_from_file(args.main_cfg_path)
                    matcher = LGFCTR_vis(config['LGFCTR']).to('cuda')
                    matcher.eval()
                    state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
                    matcher.load_state_dict(state_dict, strict=True)
                    transform = Compose([ToTensor()])

                    img0, img1 = cv2.imread(img_path0), cv2.imread(img_path1)
                    if args.is_original:
                        cv2.imwrite(os.path.join(save_dir, img_path0), img0)
                        cv2.imwrite(os.path.join(save_dir, img_path1), img1)
                    if img_resize > 0:
                        fx0 = fy0 = img_resize / max(img0.shape)
                        fx1 = fy1 = img_resize / max(img1.shape)
                        img0 = cv2.resize(img0, None, fx=fx0, fy=fy0)
                        img1 = cv2.resize(img1, None, fx=fx1, fy=fy1)
                    tensor0 = transform(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)).unsqueeze(0).to('cuda')
                    tensor1 = transform(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)).unsqueeze(0).to('cuda')
                    batch = {'image0': tensor0, 'image1': tensor1}
                    matcher(batch)

                    attention_vis_lists = batch['attention']
                    flag = False
                attention_vis = attention_vis_lists[resolution_idx][layer_idx]
                attention_vis0, attention_vis1 = attention_vis[0][0], attention_vis[1][0]
                np.save(npy_path0, np.array(attention_vis0))
                np.save(npy_path1, np.array(attention_vis1))

            print(attention_vis0.shape)
            print(attention_vis1.shape)
            h0, w0 = attention_vis0.shape[1:3]
            h1, w1 = attention_vis1.shape[1:3]
            size_factor = 2 ** (5 - (abs(resolution_idx - 2.5) + 0.5))
            h0_i, w0_i = int(h0 * size_factor), int(w0 * size_factor)
            h1_i, w1_i = int(h1 * size_factor), int(w1 * size_factor)
            print(h0_i, w0_i)
            print(h1_i, w1_i)

            # generating attention weights of the first image
            for i in range(len(attention_vis0)):
                attention_map = attention_vis0[i, h0//2, w0//2]
                attention_map = (attention_map - np.mean(attention_map)) / (np.std(attention_map) + 1e-6)
                attention_map_original = cv2.resize(attention_map, (w0_i, h0_i))
                fig, axe = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
                axe.imshow(attention_map_original, cmap=args.cmap)
                axe.get_yaxis().set_ticks([])
                axe.get_xaxis().set_ticks([])
                for spine in axe.spines.values():
                    spine.set_visible(False)
                plt.tight_layout(pad=1)
                path = os.path.join(save_dir, f'{img_path0}-{img_path1}-{resolution_idx}-{layer_idx}-0-{i}.png')
                plt.savefig(path, bbox_inches='tight', pad_inches=0)
                plt.close()

            # generating attention weights of the second image
            for i in range(len(attention_vis1)):
                attention_map = attention_vis1[i, h1//2, w1//2]
                attention_map = (attention_map - np.mean(attention_map)) / (np.std(attention_map) + 1e-6)
                attention_map_original = cv2.resize(attention_map, (w1_i, h1_i))
                fig, axe = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
                axe.imshow(attention_map_original, cmap=args.cmap)
                axe.get_yaxis().set_ticks([])
                axe.get_xaxis().set_ticks([])
                for spine in axe.spines.values():
                    spine.set_visible(False)
                plt.tight_layout(pad=1)
                path = os.path.join(save_dir, f'{img_path0}-{img_path1}-{resolution_idx}-{layer_idx}-1-{i}.png')
                plt.savefig(path, bbox_inches='tight', pad_inches=0)
                plt.close()
