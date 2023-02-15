import os
import os.path as osp
import torch
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import albumentations as A

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model.isnet import DISNet
from model.isnet_custom import GtEncoder
from model.segmentation import Net

import argparse

def load_dataloader(args):    
        
    mask_transform = A.Compose([
        # A.Resize(width=args.input_size, height=args.input_size),
        A.RandomCrop(width=1024, height=1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.8),
        A.RandomRotate90(p=0.8)
    ])

    image_transform = A.Compose([
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)]
    )
    
    vd_transform = A.Compose([
        A.Resize(width=1024, height=1024)
    ])
    
    if args.train_type == 'disnet':
        from utils.isnet_dataset import Dataset, RandomBlur
        tr_ds = Dataset(image_path=args.tr_im_path, gt_path=args.tr_gt_path,
                        image_transform=image_transform,
                        gt_transform=mask_transform,
                        random_blur=None,
                        load_on_mem=args.load_data_on_mem)
        vd_ds = Dataset(image_path=args.vd_im_path, gt_path=args.vd_gt_path,
                        image_transform=vd_transform,
                        gt_transform=vd_transform,
                        load_on_mem=args.load_data_on_mem)
    else:
        from utils.gt_dataset import Dataset
        tr_ds = Dataset(image_path=args.tr_gt_path, transform=mask_transform)
        vd_ds = Dataset(image_path=args.vd_gt_path, transform=vd_transform)
    
    
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=8)
    vd_dl = DataLoader(vd_ds, args.batch_size, shuffle=False, num_workers=4)
    
    return tr_dl, vd_dl

def load_model(args):
    os.makedirs(args.save_weight_path, exist_ok=True)
    if args.train_type == 'disnet':
        net = Net(DISNet(3,1), args.dis_weight)
        net.load_gt_encoder(GtEncoder(1,1), args.gt_weight)
    elif args.train_type == 'gt_encoder':
        net = Net(GtEncoder(1,1), args.gt_weight)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=osp.join(args.save_weight_path, args.train_type),
        filename="{epoch:02d}-{val_loss:.2f}-" + f"batch_size={str(args.batch_size)}",
        save_top_k=3,
        mode="min"
    )
    
    # wandb_logger = WandbLogger(name=f'{args.train_type}',project='DISNet')
    trainer = pl.Trainer(# logger=wandb_logger,
             callbacks=[checkpoint_callback, early_stop_callback],
             devices=torch.cuda.device_count(), strategy='ddp',
             accelerator='gpu',
             min_epochs=args.min_epoch,
             max_epochs=args.max_epoch,
             profiler='simple')
    
    return trainer, net
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DISNet Training')
    parser.add_argument('--input_size',        type=int,      default=1280)
    parser.add_argument('--min_epoch',         type=int,      default=300)
    parser.add_argument('--max_epoch',         type=int,      default=400)
    parser.add_argument('--load_data_on_mem',  type=bool,     default=False)
    parser.add_argument('--batch_size',        type=int,      default=4)
    parser.add_argument('--lr',                type=float,    default=0.001)
    parser.add_argument('--epsilon',           type=float,    default=1e-08)
    parser.add_argument('--train_type',        type=str,      default='disnet', choices=['disnet', 'gt_encoder'])
    parser.add_argument('--tr_im_path',        type=str,      default='')
    parser.add_argument('--tr_gt_path',        type=str,      default='')
    parser.add_argument('--vd_im_path',        type=str,      default='')
    parser.add_argument('--vd_gt_path',        type=str,      default='')
    parser.add_argument('--save_weight_path',  type=str,      default='saved_model')
    parser.add_argument('--dis_weight',        type=str)
    parser.add_argument('--gt_weight',         type=str)
    
    args = parser.parse_args()
    
    # dataloader
    tr_dl, vd_dl = load_dataloader(args)
    
    # model
    trainer, model = load_model(args)
    
    # run
    trainer.fit(model, tr_dl, vd_dl)

    
    
