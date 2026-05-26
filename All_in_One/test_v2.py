
import argparse
import os
import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from net.promptrestormerv3_arch import DualPromptRestormerv3_only_task_prompt


def tensor2img(tensor):

    img = tensor.squeeze(0).clamp(0, 1).cpu().numpy()   # [3,H,W]
    img = (img * 255).round().astype(np.uint8)
    img = img.transpose(1, 2, 0)                         # [H,W,3]
    return img


def save_img(img_np, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(img_np).save(save_path)


def load_model(ckpt_path, device):

    net = DualPromptRestormerv3_only_task_prompt()

    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                new_state_dict[k[4:]] = v

        net.load_state_dict(new_state_dict, strict=True)
        print(f"[✓] 从 Lightning checkpoint 加载权重: {ckpt_path}")
    else:

        net.load_state_dict(ckpt, strict=True)
        print(f"[✓] 从 PyTorch checkpoint 加载权重: {ckpt_path}")

    net = net.to(device).eval()
    return net



def test_denoise(net, dataset, sigma, output_dir, device, save_images):
    
    dataset.set_sigma(sigma)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True,
                        shuffle=False, num_workers=0)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    save_subdir = os.path.join(output_dir, f'denoise_sigma{sigma}')

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(loader,
                desc=f'Denoise σ={sigma}', leave=False):
            degrad_patch = degrad_patch.to(device)
            clean_patch  = clean_patch.to(device)

            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr_meter.update(temp_psnr, N)
            ssim_meter.update(temp_ssim, N)

            if save_images:
                img_np = tensor2img(restored)
                fname = os.path.splitext(os.path.basename(clean_name[0]))[0] + '.png'
                save_img(img_np, os.path.join(save_subdir, fname))

    return psnr_meter.avg, ssim_meter.avg


def test_derain_dehaze(net, dataset, task, output_dir, device, save_images):
    
    dataset.set_dataset(task)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True,
                        shuffle=False, num_workers=0)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    factor = 32

    save_subdir = os.path.join(output_dir, task)

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(loader,
                desc=task, leave=False):
            degrad_patch = degrad_patch.to(device)
            clean_patch  = clean_patch.to(device)

            b, c, h, w = degrad_patch.shape
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            degrad_patch_pad = F.pad(degrad_patch, (0, w_n, 0, h_n), mode='reflect')

            restored = net(degrad_patch_pad)[:, :, :h, :w]

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr_meter.update(temp_psnr, N)
            ssim_meter.update(temp_ssim, N)

            if save_images:
                img_np = tensor2img(restored)
                fname = os.path.splitext(os.path.basename(degraded_name[0]))[0] + '.png'
                save_img(img_np, os.path.join(save_subdir, fname))

    return psnr_meter.avg, ssim_meter.avg


def print_summary(results, log_path):
    order   = ['denoise_15', 'denoise_25', 'denoise_50',
                'derain', 'dehaze', 'deblur', 'enhance']
    avg_set = {'denoise_25', 'derain', 'dehaze', 'deblur', 'enhance'}

    psnr_vals = [results[k][0] for k in avg_keys if k in results]
    ssim_vals = [results[k][1] for k in avg_keys if k in results]
    avg_psnr  = sum(psnr_vals) / len(psnr_vals) if psnr_vals else 0.0
    avg_ssim  = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0.0

    lines = []
    lines.append("")
    lines.append("=" * 62)
    lines.append("  Evaluation Results")
    lines.append("-" * 62)
    lines.append(f"  {'Task':<16}{'PSNR':>10}{'SSIM':>12}   in_avg")
    lines.append("-" * 62)
    for task in order:
        if task in results:
            p, s = results[task]
            mark = 'Y' if task in avg_set else '-'
            lines.append(f"  {task:<16}{p:>10.2f}{s:>12.4f}   {mark}")
    lines.append("-" * 62)
    lines.append(f"  {'Avg (5-task)':<16}{avg_psnr:>10.2f}{avg_ssim:>12.4f}   "
                 f"(N={len(psnr_vals)})")
    lines.append("=" * 62)

    for line in lines:
        print(line)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write('\n'.join(lines) + '\n')

    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description='DualPromptRestormerv3_only_task_prompt_5Task_testing')

    TEST_BASE = 'datasets/ALL_IN_ONE_5T_TEST'

    parser.add_argument('--ckpt', type=str,
        default='',
        help='checkpoint 路径')
    parser.add_argument('--output_dir', type=str, default='test_results/',
        help='结果保存根目录')
    parser.add_argument('--save_images', action='store_true', default=True,
        help='是否保存输出图像（默认开启）')
    parser.add_argument('--no_save_images', dest='save_images', action='store_false',
        help='只计算指标，不保存图像')

    # 测试集路径（与 train.py 保持一致）
    parser.add_argument('--denoise_path', type=str, default=TEST_BASE)
    parser.add_argument('--derain_path',  type=str, default=TEST_BASE)
    parser.add_argument('--dehaze_path',  type=str, default=os.path.join(TEST_BASE, 'SOTS/'))
    parser.add_argument('--gopro_path',   type=str, default=TEST_BASE)
    parser.add_argument('--enhance_path', type=str, default=TEST_BASE)

    # 任务列表
    parser.add_argument('--de_type', nargs='+',
        default=['denoise_15', 'denoise_25', 'denoise_50',
                 'derain', 'dehaze', 'deblur', 'enhance'])


    parser.add_argument('--patch_size',   type=int, default=192)
    parser.add_argument('--data_file_dir',type=str, default='data_dir/')

    parser.add_argument('--denoise_dir',  type=str, default='')
    parser.add_argument('--derain_dir',   type=str, default='')
    parser.add_argument('--dehaze_dir',   type=str, default='')
    parser.add_argument('--gopro_dir',    type=str, default='')
    parser.add_argument('--enhance_dir',  type=str, default='')

    parser.add_argument('--gpu', type=int, default=0, help='使用哪张 GPU，-1 表示 CPU')

    opt = parser.parse_args()

    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{opt.gpu}')
    else:
        device = torch.device('cpu')
    print(f"[设备] {device}")

    net = load_model(opt.ckpt, device)

    # 去噪
    denoise_path_base = opt.denoise_path
    opt.denoise_path  = os.path.join(denoise_path_base, 'CBSD68/')
    denoise_testset   = DenoiseTestDataset(opt)

    # 去雨
    derain_path_base  = opt.derain_path
    opt.derain_path   = os.path.join(derain_path_base, 'Rain100L/')
    derain_set        = DerainDehazeDataset(opt, addnoise=False, sigma=15)

    # 去雾
    dehaze_set        = DerainDehazeDataset(opt, addnoise=False, sigma=15)

    # 去模糊
    gopro_path_base   = opt.gopro_path
    opt.gopro_path    = os.path.join(gopro_path_base, 'GoPro/')
    deblur_set        = DerainDehazeDataset(opt, addnoise=False, sigma=15)

    # 低光增强
    enhance_path_base = opt.enhance_path
    opt.enhance_path  = os.path.join(enhance_path_base, 'LOL/')
    enhance_set       = DerainDehazeDataset(opt, addnoise=False, sigma=15)

    results = {}
    print("\n开始推理测试...\n")

    for task in opt.de_type:
        if task == 'denoise_15':
            p, s = test_denoise(net, denoise_testset, sigma=15,
                                output_dir=opt.output_dir, device=device,
                                save_images=opt.save_images)
            results['denoise_15'] = (p, s)
            print(f"  Denoise σ=15  |  PSNR: {p:.2f}  SSIM: {s:.4f}")

        elif task == 'denoise_25':
            p, s = test_denoise(net, denoise_testset, sigma=25,
                                output_dir=opt.output_dir, device=device,
                                save_images=opt.save_images)
            results['denoise_25'] = (p, s)
            print(f"  Denoise σ=25  |  PSNR: {p:.2f}  SSIM: {s:.4f}")

        elif task == 'denoise_50':
            p, s = test_denoise(net, denoise_testset, sigma=50,
                                output_dir=opt.output_dir, device=device,
                                save_images=opt.save_images)
            results['denoise_50'] = (p, s)
            print(f"  Denoise σ=50  |  PSNR: {p:.2f}  SSIM: {s:.4f}")

        elif task == 'derain':
            p, s = test_derain_dehaze(net, derain_set, task='derain',
                                      output_dir=opt.output_dir, device=device,
                                      save_images=opt.save_images)
            results['derain'] = (p, s)
            print(f"  Derain        |  PSNR: {p:.2f}  SSIM: {s:.4f}")

        elif task == 'dehaze':
            p, s = test_derain_dehaze(net, dehaze_set, task='dehaze',
                                      output_dir=opt.output_dir, device=device,
                                      save_images=opt.save_images)
            results['dehaze'] = (p, s)
            print(f"  Dehaze        |  PSNR: {p:.2f}  SSIM: {s:.4f}")

        elif task == 'deblur':
            p, s = test_derain_dehaze(net, deblur_set, task='deblur',
                                      output_dir=opt.output_dir, device=device,
                                      save_images=opt.save_images)
            results['deblur'] = (p, s)
            print(f"  Deblur        |  PSNR: {p:.2f}  SSIM: {s:.4f}")

        elif task == 'enhance':
            p, s = test_derain_dehaze(net, enhance_set, task='enhance',
                                      output_dir=opt.output_dir, device=device,
                                      save_images=opt.save_images)
            results['enhance'] = (p, s)
            print(f"  Enhance       |  PSNR: {p:.2f}  SSIM: {s:.4f}")

    log_path = os.path.join(opt.output_dir, 'test_results.log')
    avg_psnr, avg_ssim = print_summary(results, log_path)

    print(f"\n[✓] 日志已保存到: {log_path}")
    if opt.save_images:
        print(f"[✓] 图像已保存到: {opt.output_dir}")

avg_keys = ['denoise_25', 'derain', 'dehaze', 'deblur', 'enhance']

if __name__ == '__main__':
    main()
