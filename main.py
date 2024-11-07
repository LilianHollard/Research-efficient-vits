from utils.dataloader import get_data, get_cifar
from utils.liltrainer import LilTrainer
from utils.losses import DistillationLoss

from models.models import custom_dino, ResNet, ViT_cls, ViT_nocls, leyolo_backbone, SwiftFormer, mini_former, mini_vit, mini_mlp_encoder, SwiftFormer_full, mini_vit_former, le_vit_former
from models.models import efficientMod_xxs, StarNet 
from utils import utils

import torch.multiprocessing as mp
import torch.distributed as dist

import torch.backends.cudnn as cudnn

import os
import argparse

import torch
import torch.nn as nn

import pandas as pd

#for warmup and cosine decay (pip install transformers)
from transformers import get_cosine_schedule_with_warmup




def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--data', default="./imagenette320/", help="path to dataset")
    parser.add_argument('--train', default="False", help="launch training (--val for validation only)", action="store_true")
    parser.add_argument('--val', default="False", help="launch validation (--train for training and validation)", action="store_true")

    parser.add_argument('--model_path', default="", help="path to model checkpoint (mostly for validation)")

    parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
    parser.add_argument('--seed', default=42, type=int, help="training seed")
    parser.add_argument('--device', default='cuda', help="device for training/testing")
    parser.add_argument('--dir_path', default='runs/', help="save path folder")
    parser.add_argument('--gpus', default="", help="gpus indexes")
    

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    
    
    
    #LeYOLO backbone specific args
    parser.add_argument('--leyolo_k', default=16, type=int, help="number of channels scaling")

    return parser


def train(args):
    if 'RANK' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        
        args.gpu = int(os.environ["LOCAL_RANK"])
        
        args.distributed = True

        #torch.cuda.set_device(args.gpu)
        args.dist_backend = "nccl"
        print('Distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

        torch.distributed.init_process_group(backend=args.dist_backend)#, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        torch.distributed.barrier()
        utils.setup_for_distributed(args.rank == 0)
    else:
        args.distributed = False
        torch.cuda.set_device(0)


    save_path = args.dir_path + "run"
    # get number of folder in dir_path
    if utils.is_main_process():
        count = 0
        for path in os.listdir(args.dir_path):
            if os.path.isdir(os.path.join(args.dir_path, path)):
                count += 1

        # create path for saving the new training
        save_path += str(count) + "/"
        os.mkdir(save_path)

    print("saving model to ", save_path)
    device = torch.device(args.device)

    # reproducibility seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    # np.random.seed(seed)

    cudnn.benchmark = True
    print("Loading dataset...")
    # data loading
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    #train_loader, test_loader = get_cifar(args.data, args.batch_size, args.distributed, num_tasks, global_rank)
    train_loader, test_loader = get_data(args.data, args.batch_size, args.distributed, num_tasks, global_rank)
    # model dist
    #
    model = StarNet(24, [2,2,8,3], drop_path_rate=0., num_classes=1000)
    #model = SwiftFormer(10, 0.5, 0.5)
    #model = SwiftFormer_full(10)
    #model = mini_former()
    #model = mini_vit_former(10, 1.0, 1.0)
    #model = le_vit_former()
    #model = ResNet(18)
    #model = mini_vit()
    #model = mini_mlp_encoder()
    #d_dim = 96
    #ff_dim = 4 * d_dim
    #model = ViT_cls(d_dim, ff_dim, 8, 6, 10, 4)

    model.to(device)

    print('Loading model...')
    print('=============================================')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)

    """with torch.profiler.profile(with_flops=True) as p, torch.autocast('cuda'):
        _ = model(torch.randn(1,3,32,32).to(device))
    print(p.key_averages().table(sort_by="flops", row_limit=5))
    print('{:.2f} GMAC (torch profile)'.format(sum(k.flops for k in p.key_averages()) / 1e9))
    print('{:.2f} GFLOP (torch profile)'.format(sum(k.flops for k in p.key_averages()) / 1e9 * 2.0))"""



    
    df = pd.DataFrame(columns=["epoch","param", "best", "cos"]) 

    teacher_model = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    trainer = LilTrainer(model)

    # learning
    #linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    #args.lr = linear_scaled_lr
    #print(args.lr)

    opt_func = torch.optim.AdamW
    #opt_func = torch.optim.SGD
    max_lr = 0.0001
    grad_clip = 0.01
    weight_decay = 0.025

    # history = []
    # Fitting the first 1/4 epochs

    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), args.lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
    # steps_per_epoch=len(train_loader))

    """sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30000,
    )"""
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = 5 * len(train_loader)
    sched = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
    #                                        steps_per_epoch=len(train_loader))

    # grad_scaler = torch.cuda.amp.GradScaler()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # LeViT code is not very clear on which loss is used

    if teacher_model is not None:
        criterion = DistillationLoss(criterion, teacher_model, "soft", 0.5, 1.0)

    best = 0.0
    
    save_csv = pd.DataFrame([])
    
    
    print("Start training...")
    for epoch in range(args.epochs):


        if epoch%10 == 0:
            if utils.is_main_process():
                df.to_csv("csv/rewrite_the_square.csv", index=False)


        print("Epoch {}/{}".format(epoch, args.epochs))
        # Training Phase
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_losses = []
        lrs = []
        for i, batch in enumerate(train_loader):
            #print(i)
            torch.cuda.empty_cache()
            loss = trainer.training_step(batch, device, criterion)
            #train_losses.append(loss)

            loss.backward()

            #if (i+1) % (256 // args.batch_size) == 0:
                #print("accumulating gradient ?")
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss)

            torch.cuda.synchronize()
            # grad_scaler.scale(loss).backward()
            # grad_scaler.step(optimizer)
            # grad_scaler.update()

            # Record & update learning rate
            lrs.append(utils.get_lr(optimizer))

            sched.step()

        # Validation phase
        print("Evaluating...")
        result = trainer.evaluate(test_loader, device)
    
        if args.distributed and utils.is_main_process():
            df = pd.concat([df,pd.DataFrame([[epoch,result['val_loss'],n_parameters]],columns=["epoch", "param", "best"])],ignore_index=True)
            #df = pd.concat([df,pd.DataFrame([[epoch,result['val_loss'],n_parameters, result['cos_similarity'].numpy().tolist()]],columns=["epoch", "param", "best", "cos"])],ignore_index=True)
   

        if best < result['val_acc']:
            print("========= NEW best accuracy !! =========")
            best = result['val_acc']
            if args.distributed:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': sched.state_dict(),
                    'epoch': epoch,
                    # 'scaler': loss.state_dict(),
                    'args': args,

                }, save_path + "best.pt")
            else:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': sched.state_dict(),
                    'epoch': epoch,
                    # 'scaler': loss.state_dict(),
                    'args': args,

                }, save_path + "best.pt")

        result['train_loss'] = torch.stack(train_losses).mean().item()
        # result['lrs'] = lrs
        trainer.epoch_end(epoch, result)
        history.append(result)


    #End of training
    #Loading best saved model (from best accuracy)
    #Evaluating once again, printing results and corresponding epoch.
    if args.distributed and utils.is_main_process():
        print("End of training... loading and printing best model evaluation")
        best_model = torch.load(save_path + "best.pt")
        model_without_ddp.load_state_dict(best_model["model"])
        result = trainer.evaluate(test_loader, device)

        print("Epoch [{}], val_acc: {:.4f}".format(
            best_model["epoch"], result['val_acc']))
        #todo...
        #trainer.epoch_end(best_model["epoch"], result) #need to save all training and val loss btw to make it work


def val(args):

    print("Validating model...")
    device = torch.device(args.device)
    # model dist
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = custom_dino(dinov2_vits14)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)
    trainer = LilTrainer(model)

    _, test_loader = get_data(args.data, args.batch_size)


    val_model = torch.load(args.model_path)
    model.load_state_dict(val_model["model"])
    result = trainer.evaluate(test_loader, device)

    print("Epoch [{}], val_acc: {:.4f}".format(
        val_model["epoch"], result['val_acc']))



def main(args):

    ## I have no idea why "if args.train:" only doesn't work (Maybe it is stored as a String with argparser).
    if args.train == True:
        train(args)
    elif args.val == True:
        val(args)


if __name__ == "__main__":

    #Launch
    #python main.py --val --data ".\imagenette2-320\" --model_path ".\runs\run10\best.pt"

    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
