import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

import pytorch_lightning as pl
from model.isnet import DISNet, GtEncoder

from collections import OrderedDict as OD

bce_loss = nn.BCELoss()
bce_w_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss(reduction = "mean")

def bce_loss_calc(gt_feature_maps, gt):
    sum_loss = 0
    for idx, output in enumerate(gt_feature_maps):
        component = F.interpolate(output, gt.shape[2:], mode='bilinear', align_corners=True)
        loss = bce_loss(component, gt)
        sum_loss += loss
    return sum_loss

def feature_sync(gt_outputs, net_outputs):
    loss_lst = []
    for idx, gt_output in enumerate(gt_outputs):
        loss = mse_loss(gt_output, net_outputs[idx])
        loss_lst.append(loss)

    loss = sum(loss_lst)
    return loss

    loss = nn.L1Loss()
    return loss(input_data, target_data)

class Net(pl.LightningModule):
    def __init__(self, model, pretrained: str = None, lr: float = 0.001, epsilon: float = 1e-08) -> object:
        super(Net, self).__init__()
        self.lr = lr
        self.epsilon = epsilon
        self.net = model
        self.gt_encoder = None
        
        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.net.load_state_dict(state_dict)
            print('----------------------------------------------------------------------------------------------------')
            print('pretrained loaded')
            print('----------------------------------------------------------------------------------------------------')
    
    def load_gt_encoder(self, gt_encoder, pretrained: str = None):
        self.gt_encoder = gt_encoder
        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
            sd = OD()
            for key, value in state_dict.items():
                sd[key.replace('net.', '')] = value
                
            self.gt_encoder.load_state_dict(sd)
            
        self.gt_encoder.eval()
        print('gt_encoder is loaded')
        print('----------------------------------------------------------------------------------------------------')
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=self.epsilon, weight_decay=0)
        return optimizer
    
    def forward(self, x):
        return self.net(x)
    
    def _common_step(self, batch, batch_idx, stage):
        image, gt = batch['image'], batch['gt']
        im_side_outputs, im_features = self.net(image)
        loss = bce_loss_calc(im_side_outputs, gt)
        
        if self.gt_encoder:
            gt_side_outputs, gt_features = self.gt_encoder(gt)
            fs_mse_loss = feature_sync(gt_features, im_features)
            loss += fs_mse_loss
        
        self.log(f"{stage}_loss", loss, on_epoch=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, 'train')
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, 'val')
        self.val_loss = loss
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.gt_encoder:
            state_dict = checkpoint['state_dict']
            for key, value in state_dict.items():
                if 'gt_encoder' in key:
                    state_dict.pop(key)
                
    def predict_step(self, batch, batch_idx):
        image, gt = batch['image'], batch['gt']
        return self.net(image)
        