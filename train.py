import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os, sys
import time
import tabulate
import data
import training_utils
import nets as models
import numpy as np
from parser_train import parser

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def cross_entropy(model, x, target, reduction="mean"):
    """standard cross-entropy loss function"""
    if model is not None:
        output = model(x)
    else:
        output = x

    loss = F.cross_entropy(output, target, reduction=reduction)

    if reduction is None or reduction == "none":
        loss = loss
    if reduction == 'mean':
        loss = torch.mean(loss)
    if reduction == 'sum':
        loss = torch.sum(loss)

    if model is not None:
        return loss, output

    return loss



def check_si_name(n, model_name='ResNet18'):
    return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n

def main():
    args = parser()
    args.device = None
    
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.cuda = True
    else:
        args.device = torch.device("cpu")
        args.cuda = False
        
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed)

    # n_trials = 1
    
    print("Preparing base directory %s" % args.dir)
    os.makedirs(args.dir, exist_ok=True)

    # for trial in range(n_trials):
    trial = args.trial
    output_dir = args.dir + f"/trial_{trial}"
    
    ### resuming is modified!!!
    if args.resume_epoch > -1:
        assert False
        
    ### resuming is modified!!!
    print("Preparing directory %s" % output_dir)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    print("Using model %s" % args.model)
    model_cfg = getattr(models, args.model)

    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    transform_train = model_cfg.transform_test if args.no_aug else model_cfg.transform_train
    
    loaders, num_classes = data.loaders(
        'CIFAR10',
        args.data_path,
        128,
        4,
        model_cfg.transform_test,
        model_cfg.transform_test,
        use_validation=False,
        use_data_size=50000,
        split_classes=None,
        corrupt_train=0.0
    )
    assert num_classes == 10

    print("Preparing model")
    print(*model_cfg.args)

  
    extra_args = {'init_channels':args.num_channels, 'max_depth':args.depth,'init_scale':args.init_scale}

    
    
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs,
                           **extra_args)
    
    set_random_seed(228)
    fin = nn.Linear(model.linear_layers[-1].in_features,model.linear_layers[-1].out_features) 
    alpha = args.init_scale
    W = fin.weight.data
    model.linear_layers[-1].weight.data = alpha * W / W.norm()
    model.linear_layers[-1].bias.data = fin.bias.data
    set_random_seed(args.seed)

    model.to(args.device)

       
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if check_si_name(n, args.model)]},  # SI params are convolutions
        {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args.model)]},  # other params
            ]

    with torch.no_grad():
        si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for p in param_groups[0]["params"]))
        lr = args.elr * si_pnorm_0 ** 2
        
   
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if check_si_name(n, args.model)]},  
        {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args.model)],'lr':args.noninvlr}, 
    ]

    optimizer = torch.optim.SGD(param_groups, 
                                lr=lr, 
                                momentum=args.momentum, 
                                weight_decay=args.wd)

    epoch_from = args.resume_epoch + 1
    epoch_to = epoch_from + args.epochs
    print(f"Training from {epoch_from} to {epoch_to - 1} epochs")

    train_res = {"loss": None, "accuracy": None}
    test_res = {"loss": None, "accuracy": None}

    epoch_from += 1
    
    for epoch in range(epoch_from, epoch_to+1):
        train_epoch(model, loaders, cross_entropy, optimizer,
                    epoch=epoch, 
                    end_epoch=epoch_to+1, 
                    fix_elr = args.fix_elr,
                    si_pnorm_0=si_pnorm_0,
                    lr=lr,
                    noninvlr=args.noninvlr,
                    fbgd=args.fbgd,
                    cosan_schedule = args.cosan_schedule,
                    model_name = args.model)
        if args.cosan_schedule:
            assert False

    print("model ", trial, " done")


def train_epoch(model, loaders, criterion, optimizer, epoch, end_epoch,
                eval_freq=1, save_freq=10, save_freq_int=0, fix_elr=False, fix_all_elr = False,
                si_pnorm_0=None,output_dir='./',
                lr=0.01, lr_schedule=True, noninvlr = -1, c_schedule=None, d_schedule=None,
                fbgd=False, cosan_schedule = False, model_name = 'ResNet18'):

    time_ep = time.time()


    train_res = training_utils.train_epoch(loaders["train"], model, criterion, optimizer, 
                                            epoch = epoch,fix_elr = True,
                                           si_pnorm_0=si_pnorm_0, model_name = model_name)
    
    test_res = {"loss": None, "accuracy": None}
        

        
    time_ep = time.time() - time_ep
    values = [
        epoch,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
    ]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

if __name__ == '__main__':
    main()
