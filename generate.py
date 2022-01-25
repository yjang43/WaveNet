import os
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from tqdm import tqdm
from preprocess import inv_quantize
    

def get_latest_ckpt():
    _, _, ckpts = next(os.walk('ckpt'))
    latest = -1
    for ckpt in ckpts:
        if ckpt.split('.')[-1] == 'pt':
            latest = (max(int(ckpt.split('.')[0]), latest))
    if latest == -1:
        raise ValueError('There is no checkpoint')
    return str(latest) + '.pt'

def set_option():
    parser = ArgumentParser()
    
    parser.add_argument('--ckpt_path', default=get_latest_ckpt(), type=str)
    parser.add_argument('--audio_num', default=32, type=int)
    parser.add_argument('--audio_len', default=44100, type=int)
    parser.add_argument('--num_class', default=256, type=int)
    parser.add_argument('--receptive_field', default=1024, type=int)
    return parser.parse_args()


def generate_audio(model, audio_num, audio_len, num_class, receptive_field=1024):
    model.eval()
    n, c, l = audio_num, num_class, audio_len
    device = next(model.parameters()).device
    input_ = torch.zeros(n, c, 1, device=device)   # [N, C, L + 1]
    
    for i in tqdm(range(l)):
        pred = model(input_[:, :, -receptive_field: ])
        dist = F.softmax(pred[:, :, -1], dim=1)
        sample = torch.multinomial(dist, num_samples=1)
        one_hot = F.one_hot(sample.view(n, 1), num_class).type(torch.float).permute(0, 2, 1)
        input_ = torch.cat((input_, one_hot), dim=-1)
#         input_ = torch.cat((input_, torch.zeros(n, c, 1, device=device).scatter_(1, sample.view(n, 1, 1), 1.0)), -1)
        
    audio = input_.argmax(1).to('cpu').numpy()
    audio = inv_quantize(audio, 8)
    return audio
    

if __name__ == '__main__':
    opt = set_option()
    print(opt)
    # TODO: incomplete data

