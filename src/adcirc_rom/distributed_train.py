import os
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='feedforward', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--datadir', default="/scratch/06307/clos21/shared/prateek/NA")
    parser.add_argument('--workers', default=1, help="Num workers for dataloader")
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args
                                         
def main(args):
    # DDP setting
    comm = MPI.COMM_WORLD
    args.world_size = comm.size
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    os.environ['MASTER_PORT'] = "55667"
    nodelist = os.environ['SLURM_JOB_NODELIST']
    master_addr = nodelist.strip().split()[0]
    print("Setting master_addr to ", master_addr, "on rank", comm.rank)
    os.environ['MASTER_ADDR'] = master_addr

    if args.distributed:
        args.rank = comm.rank
        args.gpu = args.rank % ngpus_per_node
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    model = FeedForwardNet(156)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    ### optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    ### resume training if necessary ###
    #if args.resume:
    #    pass
    
    ### data ###
    train_dataset = SyntheticTCDataset(args.datadir)
    train_sampler = data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
            collate_fn=tc_collate_fn
            )
    
    val_dataset = SyntheticTCDataset(args.datadir, val=True)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True,
            collate_fn=tc_collate_fn
            )
    
    torch.backends.cudnn.benchmark = True
    criterion = nn.MSELoss()
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #
        
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        if args.rank == 0: # only val and save on master node
            validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """Train model for one epoch
    """
    
    # only one gpu is visible here, so you can send cpu data to gpu by 
    # input_data = input_data.cuda() as normal
    epoch_loss = 0.0
    for target, features in train_loader:
        model.zero_grad()
        preds = model(features.cuda())
        err = criterion(preds, target.cuda())
        err.backwards()
        optimizer.step()
        epoch_loss += float(err)

    print(f"Epoch loss for {epoch}: {epoch_loss}")

    
def validate(val_loader, model, criterion, epoch, args):
    """Validate model
    """

    val_loss = 0.0
    for target, features in val_loader:
        with torch.no_grad():
            preds = model(features.cuda())
            err = criterion(preds, target.cuda())
            val_loss += float(err)

    print(f"Epoch val loss for {epoch}: {val_loss}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
