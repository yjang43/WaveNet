import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import VCTKAudio
from model import WaveNet


def set_option():
    parser = ArgumentParser()

    parser.add_argument('--DEVICE', default='cuda', type=str)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_sz', default=128, type=int)
    parser.add_argument('--num_class', default=256, type=int)
    parser.add_argument('--clip', default=1.0, type=float)
    parser.add_argument('--max_itr', default=10_000, type=int)

    parser.add_argument('--src_len', default=1024 + 64, type=int)
    parser.add_argument('--tgt_len', default=64, type=int)

    parser.add_argument('--num_block', default=4, type=int)
    parser.add_argument('--num_layer', default=10, type=int)
    parser.add_argument('--residual_dim', default=32, type=int)
    parser.add_argument('--dilation_dim', default=128, type=int)
    parser.add_argument('--skip_dim', default=256, type=int)
    parser.add_argument('--kernel_size', default=2, type=int)
    parser.add_argument('--bias', default=False, type=bool)

    parser.add_argument('--loss_update_itr', default=100, type=int)
    parser.add_argument('--ckpt_dir', default='ckpt', type=str)
    parser.add_argument('--ckpt_name', default='', type=str)
    parser.add_argument('--dataset_path', default='dataset.npz', type=str)

    return parser.parse_args()


def save_ckpt(ckpt_path, model, optimizer, misc=None):
    is_train = model.training
    device = next(model.parameters()).device

    # eval mode and cpu
    model.eval()
    model.cpu()

    # save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'last_epoch': misc['epoch'],
        'losses': misc['losses'],
    }, ckpt_path)

    # recover mode and device
    if is_train: 
        model.train()
    model.to(device)


if __name__ == '__main__':
    opt = set_option()
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # prepare dataset and dataloader
    dataset = VCTKAudio(opt.dataset_path, opt.src_len, opt.tgt_len, opt.num_class)    # TODO: give parameter accordingly
    dataloader = DataLoader(dataset, 
                            batch_size=opt.batch_sz, 
                            shuffle=True, 
                            num_workers=2)
    pbar = tqdm(range(opt.epoch * min(opt.max_itr, len(dataloader))))

    # prepare model
    model = WaveNet(    
        num_block = opt.num_block,
        num_layer = opt.num_layer,   # 10,
        class_dim = opt.num_class,
        residual_dim = opt.residual_dim,
        dilation_dim = opt.dilation_dim,
        skip_dim = opt.skip_dim,
        kernel_size = opt.kernel_size,
        bias=opt.bias
    )

    # prepare optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    losses = []
    last_epoch = 0

    # load from checkpoint
    if opt.ckpt_name:
        ckpt = torch.load(os.path.join(opt.ckpt_dir, opt.ckpt_name))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        last_epoch = ckpt['last_epoch']
        losses = ckpt['losses']

    # load model to device
    model.train()
    model.to(opt.DEVICE)

    # train
    for e in range(last_epoch, opt.epoch):
        accum_loss = 0
        
        for idx, batch in enumerate(dataloader):
            src, tgt = batch['src'].to(opt.DEVICE), batch['tgt'].to(opt.DEVICE)
            pred = model(src)[:, :, -opt.tgt_len: ]
            loss = loss_fn(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()

            accum_loss += loss.item()
            pbar.update()
            
            if (idx + 1) % opt.loss_update_itr == 0:
                avg_loss = accum_loss / opt.loss_update_itr
                pbar.set_description(f"Epoch {round(e + idx / min(opt.max_itr, len(dataloader)), 2)} | Loss: {round(avg_loss, 5)}")
                losses.append(avg_loss)
                accum_loss = 0

            if idx + 1 == opt.max_itr:
                break

        # save checkpoint
        save_ckpt(os.path.join(opt.ckpt_dir, str(e) + '.pt'), 
                  model, 
                  optimizer, 
                  {"epoch": e + 1, 
                   "losses": losses})
