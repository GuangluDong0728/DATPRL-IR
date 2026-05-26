import argparse
import subprocess
from tqdm import tqdm
import os

import torch
torch.set_float32_matmul_precision('high')
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import StarIRTrainDataset
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from net.model import StarIR
from net.promptrestormerv3_arch import DualPromptRestormerv3_only_task_prompt
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import random
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)

        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class StarIRModel(pl.LightningModule):
    def __init__(self, opt, eval_interval):
        super().__init__()
        self.opt = opt
        self.net = DualPromptRestormerv3_only_task_prompt()
        self.loss_fn = nn.L1Loss()
        self.eval_datasets()
        self.eval_interval = eval_interval
        self.loss_fft = FFTLoss()

        os.makedirs(opt.ckpt_dir, exist_ok=True)
        self.log_file = os.path.join(opt.ckpt_dir, 'train_metrics.log')
        self._init_log_file()

    @rank_zero_only
    def _init_log_file(self):
        import datetime
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"de_type     : {self.opt.de_type}\n")
            f.write(f"batch_size  : {self.opt.batch_size}\n")
            f.write(f"lr          : {self.opt.lr}\n")
            f.write(f"epochs      : {self.opt.epochs}\n")
            f.write(f"patch_size  : {self.opt.patch_size}\n")
            if self.opt.pretrain_ckpt:
                f.write(f"pretrain    : {self.opt.pretrain_ckpt}\n")
            f.write("=" * 70 + "\n")

    @rank_zero_only
    def _write_log(self, msg, also_print=True):
        if also_print:
            print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        fft_loss = self.loss_fft(restored, clean_patch)

        self.log("loss_l1", loss, prog_bar=True)
        self.log("loss_fft", fft_loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        loss += fft_loss
        return loss

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=0,
            max_epochs=self.opt.epochs,
            warmup_start_lr=1e-5
        )
        return [optimizer], [scheduler]

    def on_train_start(self):
        """训练开始前先跑一次验证，确保测试流程没问题"""
        if self.trainer.is_global_zero:
            self._write_log("[on_train_start] 初始指标验证...", also_print=True)
            self.test_all()
        if self.trainer.world_size > 1:
            self.trainer.strategy.barrier()

    def on_train_epoch_end(self, unused=None):
        metrics = self.trainer.callback_metrics
        loss_l1  = metrics.get('loss_l1',  None)
        loss_fft = metrics.get('loss_fft', None)
        lr       = metrics.get('lr',       None)

        parts = [f"[Epoch {self.current_epoch + 1:03d}] train"]
        if loss_l1  is not None: parts.append(f"loss_l1={float(loss_l1):.4f}")
        if loss_fft is not None: parts.append(f"loss_fft={float(loss_fft):.4f}")
        if lr       is not None: parts.append(f"lr={float(lr):.2e}")
        self._write_log(" ".join(parts), also_print=False)

        if (self.current_epoch + 1) % self.eval_interval == 0:
            if self.trainer.is_global_zero:
                self.test_all()
            if self.trainer.world_size > 1:
                self.trainer.strategy.barrier()

    def test_all(self):
        self.test_mode_3()

    def test_mode_3(self):
        results = {}

        for testset in self.denoise_tests:
            if 'denoise_15' in self.opt.de_type:
                results['denoise_15'] = self.test_Denoise(testset, sigma=15)
            if 'denoise_25' in self.opt.de_type:
                results['denoise_25'] = self.test_Denoise(testset, sigma=25)
            if 'denoise_50' in self.opt.de_type:
                results['denoise_50'] = self.test_Denoise(testset, sigma=50)
        if 'derain' in self.opt.de_type:
            results['derain'] = self.test_Derain_Dehaze(self.derain_set, task="derain")
        if 'dehaze' in self.opt.de_type:
            results['dehaze'] = self.test_Derain_Dehaze(self.dehaze_set, task="dehaze")
        if 'deblur' in self.opt.de_type:
            results['deblur'] = self.test_Derain_Dehaze(self.deblur_set, task="deblur")
        if 'enhance' in self.opt.de_type:
            results['enhance'] = self.test_Derain_Dehaze(self.enhance_set, task="enhance")

        avg_keys = ['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze']
        psnr_vals = [results[k][0] for k in avg_keys if k in results]
        ssim_vals = [results[k][1] for k in avg_keys if k in results]

        avg_psnr = sum(psnr_vals) / len(psnr_vals) if psnr_vals else 0.0
        avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0.0

        if psnr_vals:
            self.log("avg_psnr_3task", avg_psnr, rank_zero_only=True)
            self.log("avg_ssim_3task", avg_ssim, rank_zero_only=True)

        self._log_epoch_results(results, avg_psnr, avg_ssim,
                                len(psnr_vals), avg_keys)

    def _log_epoch_results(self, results, avg_psnr, avg_ssim, n_tasks, avg_keys):
        order   = ['denoise_15', 'denoise_25', 'denoise_50',
                   'derain', 'dehaze', 'deblur', 'enhance']
        avg_set = set(avg_keys)

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"Epoch {self.current_epoch + 1} - Evaluation Results")
        lines.append("-" * 60)
        lines.append(f"{'Task':<14}{'PSNR':>10}{'SSIM':>12}   {'in_avg'}")
        lines.append("-" * 60)
        for task in order:
            if task in results:
                p, s = results[task]
                mark = 'Y' if task in avg_set else '-'
                lines.append(f"{task:<14}{p:>10.2f}{s:>12.4f}   {mark}")
        lines.append("-" * 60)
        lines.append(f"{'Avg':<14}{avg_psnr:>10.2f}{avg_ssim:>12.4f}   "
                     f"(N={n_tasks})")
        lines.append("=" * 60)

        for line in lines:
            self._write_log(line)

    def eval_datasets(self):
        self.denoise_tests = []
        self.derain_tests  = []
        self.dehaze_tests  = []

        # 去噪
        denoise_splits    = ["CBSD68/"]
        denoise_base_path = self.opt.denoise_path
        for i in denoise_splits:
            self.opt.denoise_path = os.path.join(denoise_base_path, i)
            denoise_testset = DenoiseTestDataset(self.opt)
            self.denoise_tests.append(denoise_testset)

        # 去雨
        derain_splits    = ["Rain100L/"]
        derain_base_path = self.opt.derain_path
        for name in derain_splits:
            self.opt.derain_path = os.path.join(derain_base_path, name)
            self.derain_set = DerainDehazeDataset(self.opt, addnoise=False, sigma=15)

        # 去雾
        self.opt.dehaze_path = self.opt.dehaze_path
        self.dehaze_set = DerainDehazeDataset(self.opt, addnoise=False, sigma=15)

        # 去模糊
        deblur_splits    = ["GoPro/"]
        deblur_base_path = self.opt.gopro_path
        for name in deblur_splits:
            self.opt.gopro_path = os.path.join(deblur_base_path, name)
            self.deblur_set = DerainDehazeDataset(self.opt, addnoise=False, sigma=15)

        # 低光增强
        enhance_splits    = ["LOL/"]
        enhance_base_path = self.opt.enhance_path
        for name in enhance_splits:
            self.opt.enhance_path = os.path.join(enhance_base_path, name)
            self.enhance_set = DerainDehazeDataset(self.opt, addnoise=False, sigma=15)

    def test_Denoise(self, dataset, sigma=15):
        dataset.set_sigma(sigma)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True,
                                shuffle=False, num_workers=0)
        psnr = AverageMeter()
        ssim = AverageMeter()

        with torch.no_grad():
            for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
                restored = self.forward(degrad_patch)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

            print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
            self.log("psnr %d" % sigma, psnr.avg, rank_zero_only=True)
            self.log("SSIM %d" % sigma, ssim.avg, rank_zero_only=True)
        return psnr.avg, ssim.avg

    def test_Derain_Dehaze(self, dataset, task="derain"):
        dataset.set_dataset(task)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True,
                                shuffle=False, num_workers=0)
        psnr   = AverageMeter()
        ssim   = AverageMeter()
        factor = 32

        with torch.no_grad():
            for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

                b, c, h, w = degrad_patch.shape
                h_n = (factor - h % factor) % factor
                w_n = (factor - w % factor) % factor
                degrad_patch = torch.nn.functional.pad(
                    degrad_patch, (0, w_n, 0, h_n), mode='reflect')

                restored = self.forward(degrad_patch)[:, :, :h, :w]
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

            self.log("psnr %s" % task, psnr.avg, rank_zero_only=True)
            self.log("SSIM %s" % task, ssim.avg, rank_zero_only=True)
            print("PSNR_%s: %.2f, SSIM_%s: %.4f" % (task, psnr.avg, task, ssim.avg))
        return psnr.avg, ssim.avg


@rank_zero_only
def _copy_model_file(ckpt_dir):
    path = ckpt_dir + '_model'
    if not os.path.exists(path):
        os.makedirs(path)
    os.system('cp net/model.py ' + path)


def load_pretrain_weights(model, ckpt_path):

    print(f"[pretrain] 加载模型权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']

        net_state = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                net_state[k[4:]] = v
        missing, unexpected = model.net.load_state_dict(net_state, strict=True)
        if missing:
            print(f"  [警告] 缺失 key: {missing}")
        if unexpected:
            print(f"  [警告] 多余 key: {unexpected}")
        print(f"  [✓] 成功加载 {len(net_state)} 个参数块（只加载网络权重，训练状态从头开始）")
    else:
        # 普通 state_dict
        model.net.load_state_dict(ckpt, strict=True)
        print("  [✓] 加载普通 state_dict 成功")


def main():
    parser = argparse.ArgumentParser(description='DualPromptRestormerv3_only_task_prompt 三任务训练')

    TEST_BASE = 'datasets/ALL_IN_ONE_5T_TEST'

    # ── 测试集路径 ──
    parser.add_argument('--denoise_path', type=str, default=TEST_BASE)
    parser.add_argument('--derain_path',  type=str, default=TEST_BASE)
    parser.add_argument('--dehaze_path',  type=str,
                        default=os.path.join(TEST_BASE, 'SOTS/'))
    parser.add_argument('--gopro_path',   type=str, default=TEST_BASE)
    parser.add_argument('--enhance_path', type=str, default=TEST_BASE)

    # ── 训练超参 ──
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--batch_size', type=int,   default=24)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--patch_size', type=int,   default=128)
    parser.add_argument('--num_workers',type=int,   default=8)

    parser.add_argument('--de_type', nargs='+',
                        default=['denoise_15', 'denoise_25', 'denoise_50',
                                 'derain', 'dehaze'],
                        help='训练和测试的退化类型')

    parser.add_argument('--data_file_dir', type=str, default='data_dir/')
    parser.add_argument('--denoise_dir', type=str,
        default='datasets/WED/')
    parser.add_argument('--derain_dir', type=str,
        default='datasets/Rain100L_test_train/Rain100L/')
    parser.add_argument('--dehaze_dir', type=str,
        default='datasets/OST_BETA/')
    parser.add_argument('--gopro_dir', type=str,
        default='datasets/GoPro/train/')
    parser.add_argument('--enhance_dir', type=str,
        default='datasets/LOL/our485/')

    parser.add_argument('--output_path', type=str, default="output/")
    parser.add_argument('--ckpt_path',   type=str, default="ckpt/Denoise/")
    parser.add_argument("--ckpt_dir",    type=str, default="3Task",
                        help="checkpoint 保存目录")
    parser.add_argument("--wblogger",    type=str, default="3Task")
    parser.add_argument("--num_gpus",    type=int, default=2)

    parser.add_argument('--pretrain_ckpt', type=str,
        default='',
        help='只加载网络权重，优化器/调度器从头开始')

    opt = parser.parse_args()

    _copy_model_file(opt.ckpt_dir)

    logger = TensorBoardLogger(save_dir=opt.ckpt_dir + '_logs/')

    trainset = StarIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True
    )
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    model = StarIRModel(opt, eval_interval=1)

    if opt.pretrain_ckpt:
        load_pretrain_weights(model, opt.pretrain_ckpt)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp" if opt.num_gpus > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback],
        sync_batchnorm=(opt.num_gpus > 1),
    )

    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()

