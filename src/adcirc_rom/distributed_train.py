import os
import subprocess
import sys
import builtins
import argparse
import torch
import numpy as np
import random
import torch.distributed as dist
from mpi4py import MPI
from torch import nn, optim
from torch.utils import data
from adcirc_rom.torch_models import FeedForwardNet
from adcirc_rom.torch_datasets import SyntheticTCDataset, tc_collate_fn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='feedforward', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--datadir', default="/scratch1/06307/clos21/shared/prateek-updated/NA")
    parser.add_argument('--workers', default=8, type=int, help="Num workers for dataloader")
    parser.add_argument('--save_dir', default='./trained_model', type=str, help='directory to save checkpoints and final model')
    parser.add_argument('--test', action='store_true', help='run test dataset evaluation')

    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    args = parser.parse_args()
    return args

def main(args):
    start_time = time.time()
    print("starting main function")
    print(f"args.test = {args.test}")

    # DDP setting
    comm = MPI.COMM_WORLD
    args.world_size = comm.size
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    os.environ['MASTER_PORT'] = "55667"
    nodelist = os.environ['SLURM_JOB_NODELIST']
    master_addr = subprocess.check_output(f'scontrol show hostnames "{nodelist}" | head -n 1', shell=True)
    master_addr = master_addr.decode().strip()
    print("Setting master_addr to ", master_addr, "on rank", comm.rank)
    os.environ['MASTER_ADDR'] = master_addr

    if args.distributed:
        args.rank = comm.rank
        args.gpu = args.rank % ngpus_per_node
        dist.init_process_group("nccl", world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # make checkpoint directory if not exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    ### model ###
    model = FeedForwardNet(156)
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # print(f"rank {args.rank} is using GPU {args.gpu}") # sanity check
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    ### optimizer ###
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) # change to AdamW to prevent overfitting issue
    scheduler = CosineAnnealingLR(optimizer, T_max=10) # add this to adjustable learning rate
    
    ### data loading ###
    train_dataset = SyntheticTCDataset(args.datadir)
    val_dataset = SyntheticTCDataset(args.datadir, val=True)
    test_dataset = SyntheticTCDataset(args.datadir, test=True)


    train_sampler = data.distributed.DistributedSampler(train_dataset, shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_dataset, shuffle=False) 

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=tc_collate_fn, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=tc_collate_fn, drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=tc_collate_fn, drop_last=True)


    torch.backends.cudnn.benchmark = True
    criterion = nn.MSELoss()

    # log file
    log_file = open(os.path.join(args.save_dir, 'training_log.txt'), 'w')

    # Training and validation loop
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        if args.distributed and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if args.distributed and isinstance(val_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            val_loader.sampler.set_epoch(epoch)


        epoch_start_time = time.time()
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        epoch_duration = (time.time() - epoch_start_time) / 60  # convert to min

        if args.rank == 0:
            val_loss = validate(val_loader, model, criterion, epoch, args)
            scheduler.step(val_loss)
            save_checkpoint(model_without_ddp, optimizer, epoch, args.save_dir)
            log_file.write(f"Epoch {epoch} - Training Loss: {train_loss}, Validation Loss: {val_loss}, Epoch Time: {epoch_duration:.2f} minutes\n")
    
    # save the final model
    if args.rank == 0:  
        save_final_model(model_without_ddp, args.save_dir)

    if args.test:
        print("Running test evaluation")
        test_loss = test(test_loader, model, criterion, args)
        print(f'Test Loss: {test_loss}')
        log_file.write(f"Test Loss: {test_loss}\n")
    else:
        print("Test flag not set, skipping test evaluation")
    log_file.close()
    end_time = time.time()
    training_duration = (end_time - start_time) / 60 
    print(f"Training completed in {training_duration:.2f} minutes")

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """Train model for one epoch
    """
    model.train()
    epoch_loss = 0.0
    for i, (target, features) in enumerate(train_loader):
        target = target.unsqueeze(1)  
        features, target = features.cuda(args.gpu), target.cuda(args.gpu)
        model.zero_grad()
        preds = model(features)
        err = criterion(preds, target)
        err.backward()
        # grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad(): 
            epoch_loss += err.item()  
        print(err.item())

        if i % 100 == 0:
            torch.cuda.empty_cache()
        sys.stdout.flush()

    if len(train_loader) > 0:
        epoch_loss /= len(train_loader)
    print(f"Epoch {epoch} - Training Loss: {epoch_loss}")
    torch.cuda.empty_cache()
    return epoch_loss

def validate(val_loader, model, criterion, epoch, args):
    """Validate model
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for target, features in val_loader:
            target = target.unsqueeze(1)  
            features, target = features.cuda(args.gpu), target.cuda(args.gpu)
            preds = model(features)
            err = criterion(preds, target)
            val_loss += err.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch} - Validation Loss: {val_loss}")
    torch.cuda.empty_cache()  
    return val_loss  

def test(test_loader, model, criterion, args):
    """Test model
    """
    print("Starting test function")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for target, features in test_loader:
            target = target.unsqueeze(1)  
            features, target = features.cuda(args.gpu), target.cuda(args.gpu)
            preds = model(features)
            err = criterion(preds, target)
            test_loss += err.item()

    test_loss /= len(test_loader)
    print("Test function complete")
    return test_loss

def save_checkpoint(model, optimizer, epoch, save_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def save_final_model(model, save_dir):
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
