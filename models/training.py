"""
Train/Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.train_utils import build_param_groups


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    """Decay the base LR while preserving per-group LR multipliers.

    When ``build_param_groups`` creates separate groups for attention params
    (with ``lr = base_lr * mult``), a naive ``param_group['lr'] = new_lr``
    loop would wipe out the multiplier.  We instead scale each group's LR
    by the same decay ratio so the multiplier relationship is preserved.
    """
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
        # Scheduler already set group LRs; preserve multipliers if present.
        if len(optimizer.param_groups) > 1:
            base_group = optimizer.param_groups[0]
            base_ratio = new_lr / base_group['lr'] if base_group['lr'] else 1.0
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * base_ratio
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        # Compute decay ratio relative to the *base* group's current LR so
        # that any per-group multiplier (e.g. attention_lr_mult) is preserved.
        base_group = optimizer.param_groups[0]
        old_base = base_group.get('lr', lr)
        if len(optimizer.param_groups) > 1 and old_base:
            ratio = new_lr / old_base if old_base else 1.0
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * ratio
        else:
            for group in optimizer.param_groups:
                group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            # Build param groups so attention params can have a differential
            # LR / weight decay. When attention is 'none' or multipliers are
            # both 1.0, build_param_groups returns a plain list (single group).
            param_groups = build_param_groups(model, args)
            self.optimizer = optimizer_f(param_groups)
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            try:
                checkpoint = torch.load(filename, map_location=self.args.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(filename, map_location=self.args.device)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, clip=100):
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        key_break = False
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    score = data[-2].amin(dim=-1)
                    label = data[-1]
                    if self.args.model_confidence:
                        samp = data[0]
                    else:
                        samp = data[0][:, :2]
                    z, nll = self.model(samp.float(), label=label, score=score)
                    if nll is None:
                        continue
                    if self.args.model_confidence:
                        nll = nll * score
                    losses = compute_loss(nll, reduction="mean")["total_loss"]
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_description("Loss: {}".format(losses.item()))
                    log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            self.save_checkpoint(epoch, filename=checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval")
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :2]
            with torch.no_grad():
                z, nll = self.model(samp.float(), label=torch.ones(data[0].shape[0], device=self.args.device), score=score)
            if self.args.model_confidence:
                nll = nll * score
            probs = torch.cat((probs, -1 * nll), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
