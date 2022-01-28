import os
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from scipy.io.wavfile import write
from tqdm import tqdm
from preprocess import inv_quantize
from model import WaveNet
    

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
    
    parser.add_argument('--DEVICE', default='cpu', type=str)
    parser.add_argument('--ckpt_path', default=get_latest_ckpt(), type=str)
    # parser.add_argument('--audio_num', default=32, type=int)
    parser.add_argument('--audio_num', default=1, type=int)
    # parser.add_argument('--audio_len', default=44100, type=int)
    parser.add_argument('--audio_len', default=1025, type=int)
    parser.add_argument('--num_class', default=256, type=int)
    parser.add_argument('--receptive_field', default=1024, type=int)
    parser.add_argument('--sample_rate', default=11025, type=int)

    parser.add_argument('--num_block', default=4, type=int)
    parser.add_argument('--num_layer', default=10, type=int)
    parser.add_argument('--residual_dim', default=32, type=int)
    parser.add_argument('--dilation_dim', default=128, type=int)
    parser.add_argument('--skip_dim', default=256, type=int)
    parser.add_argument('--kernel_size', default=2, type=int)
    parser.add_argument('--bias', default=False, type=bool)

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
    ckpt = torch.load(os.path.join('ckpt', opt.ckpt_path), map_location=torch.device('cpu'))

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(opt.DEVICE)
    model.eval()

    audio = generate_audio(model, opt.audio_num, opt.audio_len, opt.num_class, opt.receptive_field)
    os.makedirs('audio', exist_ok=True)

    for idx in range(opt.audio_num):
        # write to file
        fn = f"ckpt{opt.ckpt_path.split('.')[0]}_{round(opt.audio_len / opt.sample_rate)}s_{idx}.wav"
        write(os.path.join('audio', fn), opt.sample_rate, audio[idx])



