import os
from collections import OrderedDict as OD

from model.isnet import DISNet, GtEncoder

import torch
from torchvision.transforms import transforms as T

from PIL import Image
import numpy as np
import cv2

import argparse

def tensor2np(tensor_img, dst_size):
    img_np = np.array(tensor_img.cpu().detach().squeeze(0)*255, np.uint8)
    img_np = img_np.transpose(1,2,0).squeeze()
    img_np = cv2.resize(img_np, dsize=(dst_size))
    return img_np

def img2tensor(img_path):
    pil_image = Image.open(img_path)
    transform = T.Compose([
        T.Resize((1024,1024)),
        T.ToTensor()
    ])
    tn_img = transform(pil_image).unsqueeze(0)
    return tn_img, pil_image.size

def load_model(model_weight, net_type):
    state_dict = torch.load(model_weight, map_location='cpu')['state_dict']
    sd = OD()
    
    for key, value in state_dict.items():
        sd[key.replace('net.', '')] = value
    
    if net_type == 'disnet':
        net = DISNet(3,1)
    else:
        net = GtEncoder(1,1)
        
    net.load_state_dict(sd)
    print('----------------------------------------------------------------------------------------------------')
    print('net loaded succesfully!')
    print('----------------------------------------------------------------------------------------------------')
    
    return net

def inference(args):
    tn_img, init_size = img2tensor(args.img_path)
    net = load_model(args.model_weight, args.net_type)
    device = f'cuda:{str(args.device)}'
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        output, _ = net(tn_img.to(device))
    pred = output[0]
    
    return pred, init_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U2-Net Inference')
    parser.add_argument('--img_path',        type=str,      default='')
    parser.add_argument('--net_type',        type=str,      default='disnet', choices=['disnet', 'gt_encoder'])
    parser.add_argument('--device',          type=int,      default=0)
    parser.add_argument('--model_weight',    type=str,      default='')
    parser.add_argument('--save_path',       type=str,      default='output')
    args = parser.parse_args()
    
    pred, init_size = inference(args)
    pred_np = tensor2np(pred, init_size)
    
    save_path = os.path.join(os.getcwd(), args.save_path)
    dst_img_path = os.path.basename(args.img_path)
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, dst_img_path), pred_np)
    
    