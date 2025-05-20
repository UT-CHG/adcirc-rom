import os
import subprocess
import sys
import builtins
import argparse
import torch
import numpy as np
import random
import torch.distributed as dist
try:
    from mpi4py import MPI
    have_mpi4py = True
except:
    have_mpi4py = False

from torch import nn, optim
from torch.utils import data
from torch_models import FeedForwardNet, SimpleFTTransformer
from torch_datasets import tc_collate_fn, SyntheticTCDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd

def set_seed(seed=36):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='feedforward', type=str)
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--datadir', default="/scratch/06307/clos21/public/prateek-updated/Texas")
    # parser.add_argument('--datadir', default="/scratch/06307/clos21/public/prateek-updated/v3-new/NA")
    parser.add_argument('--workers', default=8, type=int, help="Num workers for dataloader")
    parser.add_argument('--save_dir', default='/work/09631/maxzhao88/vista/dev_feedfwd/trained_model', type=str, help='directory to save checkpoints and final model')
    #parser.add_argument('--save_dir', default='/work/09631/maxzhao88/vista/FTT/trained_model', type=str, help='directory to save checkpoints and final model')
    parser.add_argument('--test', action='store_true', help='run test dataset evaluation')
    parser.add_argument('--seed', default=36, type=int, help='random seed for reproducibility')
    
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    args = parser.parse_args()
    return args

# A little helper function to add the prefix to the model when loading. 
def add_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("module."):
            new_state_dict["module." + k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main(args):
    set_seed(args.seed)
    
    start_time = time.time()
    print("starting main function")
    print(f"args.test = {args.test}")

    # DDP setting
    if have_mpi4py:
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["SLURM_NTASKS"])

    args.world_size = size
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    os.environ['MASTER_PORT'] = "55667"
    nodelist = os.environ['SLURM_JOB_NODELIST']
    master_addr = subprocess.check_output(f'scontrol show hostnames "{nodelist}" | head -n 1', shell=True)
    master_addr = master_addr.decode().strip()
    print("Setting master_addr to ", master_addr, "on rank", rank)
    os.environ['MASTER_ADDR'] = master_addr

    if args.distributed:
        args.rank = rank
        args.gpu = args.rank % ngpus_per_node
        dist.init_process_group("nccl", world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    ### model ###
    # ANN
    model = FeedForwardNet(80)
    
    # FTT
    #model = SimpleFTTransformer(
    #    n_features=161,   # same as your input_dim
    #    d_token=16,      # embedding dimension (try 16/32/64, etc.)
    #    n_blocks=1,      # number of transformer blocks
    #    n_heads=1,       # must divide d_token
    #    ff_factor=4.0,
    #    dropout=0.1
    #)
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_path = os.path.join(args.save_dir, 'final_model.pth')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.gpu}
            
            #Load weights
            state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
            state_dict = add_module_prefix(state_dict)
            model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")

            #Need to set to eval mode...
            model.eval()  

 
    else:
        if torch.cuda.is_available():
            args.gpu = 0
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model_path = os.path.join(args.save_dir, 'final_model.pth')
            model.load_state_dict(torch.load(model_path))
            print("Model weights loaded successfully on single GPU.")

            model.eval()  
        else:
            raise RuntimeError("CUDA is not available, and DistributedDataParallel is required for training.")


    test_dataset = SyntheticTCDataset(args.datadir, test=True, seed=args.seed)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=tc_collate_fn, drop_last=True)
    criterion = nn.MSELoss()
    test_loss = test(test_loader, model, criterion, args)
    print(f"Final Test Loss: {test_loss}")

def test(test_loader, model, criterion, args):
    """Test model and save scatter plot of true vs predicted values."""
    print("Starting test function")
    model.eval()
    test_loss = 0.0
    all_true_values = []
    all_predicted_values = []
    
    with torch.no_grad():
        for target, features in test_loader:
            target = target.unsqueeze(1)  
            features, target = features.cuda(args.gpu), target.cuda(args.gpu)
            preds = model(features)
            err = criterion(preds, target).sqrt()
            test_loss += err.item()
            
            all_true_values.extend(target.cpu().numpy().flatten())
            all_predicted_values.extend(preds.cpu().numpy().flatten())

    test_loss /= len(test_loader)
    print("Test function complete")
    
    # set output dir for result csv file
    output_dir = './'

    df_test_result = pd.DataFrame({
        'True Values': all_true_values,
        'Predicted Values': all_predicted_values
    })

    df_test_result.to_csv('test_result.csv', index=False)
    print(f"Saved to {output_dir}/test_result.csv")

    
    return test_loss




if __name__ == '__main__':
    args = parse_args()
    main(args)
