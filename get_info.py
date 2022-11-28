"""
    script to get info from trained models
"""
import torch
import os, sys
import re
import numpy as np
from collections import defaultdict

import data
import nets as models
from get_info_funcs import eval_model, eval_trace, eval_eigs, calc_grads, get_batch_outliers_gnorms, calc_probs_stats, calc_grads_norms_small_memory
from parser_get_info import parser
from train import cross_entropy, squared_loss
from training_utils import get_resnet_prebn_groups

def check_si_name(n, model_name='ResNet18'):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNetSICI3WN' in model_name:
        return 'weight_v' in n
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    return False

def calc_cos(params_1, params_2):
    pnorm_1 = pnorm_2 = dp = 0.0

    for p1, p2 in zip(params_1, params_2):
        pnorm_1 += (p1 ** 2).sum().item()
        pnorm_2 += (p2 ** 2).sum().item()
        dp += (p1 * p2).sum().item()
    
    return dp / (pnorm_1 * pnorm_2) ** 0.5


def calc_l2(params_1, params_2):
    dp = 0.0

    for p1, p2 in zip(params_1, params_2):
        dp += ((p1 - p2)**2).sum().item()

    return dp ** 0.5

def main():
    args = parser()

    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Using model %s" % args.model)
    print("Train mode:", args.train_mode)
    model_cfg = getattr(models, args.model)

    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    print("Augmentation:", args.use_aug)
    transform = model_cfg.transform_train if args.use_aug else model_cfg.transform_test
    data_part = "test" if args.use_test else "train"
    print("Data part:", data_part)
    return_train_subsets = args.eval_on_train_subsets and args.corrupt_train is not None and data_part == "train"
    print("Evaluate model on train subsets:", return_train_subsets)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        transform,
        transform,
        use_validation=False,
        use_data_size = args.use_data_size,
        corrupt_train=args.corrupt_train,
        shuffle_train=False,
        return_train_subsets=return_train_subsets
    )
    loader = loaders[data_part]

    if args.eval_on_random_subset:
        # Currently is not compatible with eval_on_train_subsets flag
        dataset = loader.dataset
        bs = args.batch_size
        rs_size = min(10 * bs, len(dataset))
        inds = np.random.choice(len(dataset), rs_size, False)
        subset = torch.utils.data.Subset(dataset, inds)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # add extra args for varying names
    if 'ResNet18' in args.model:
        extra_args = {'init_channels':args.num_channels}
        if "SI" in args.model:
            extra_args.update({'linear_norm':args.init_scale})
    elif 'ConvNet' in args.model:
        extra_args = {'init_channels':args.num_channels, 'max_depth':args.depth,'init_scale':args.init_scale}
    elif args.model == 'LeNet':
        extra_args = {'scale':args.scale}
    else:
        extra_args = {}

    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, 
                        **model_cfg.kwargs, **extra_args)
    model.cuda()

    if args.prebn_only:
        print("Using pre-BN parameters only!")

        if args.model == 'ResNet18':
            pnames = [n for g in range(1, 4) for group in get_resnet_prebn_groups(g) for n in group]
        if args.model == 'ResNet18SI' or args.model == 'ResNet18SIAf':
            pnames = [n for n, _ in model.named_parameters() if check_si_name(n, args.model)]  # SI params are all but linear
        elif 'ConvNetSI' in args.model:
            pnames = [n for n, _ in model.named_parameters() if check_si_name(n, args.model)]  # SI params are convolutions
        else:
            raise ValueError("Using pre-BN parameters currently is not allowed for this model!")
    else:
        pnames = [n for n, _ in model.named_parameters()]

    results = defaultdict(list)
    chkpts = [f for f in os.listdir(args.models_dir) if 'checkpoint-' in f]
    results['report_epochs'] = sorted([int(re.findall(r'\d+', s)[0]) for s in chkpts])
    if args.save_freq_int > 0:
        results['report_epochs'] = sorted([int(re.findall(r'\d+', s)[0])+(int(re.findall(r'\d+', s)[1])/args.save_freq_int if len(re.findall(r'\d+', s))>1 else 1) for s in chkpts])
        epoch_indexes = {int(re.findall(r'\d+', s)[0])+(int(re.findall(r'\d+', s)[1])/args.save_freq_int if len(re.findall(r'\d+', s))>1 else 1):'-'.join(re.findall(r'\d+', s)) for s in chkpts}
    print()
    
    if 0 in results['report_epochs'] and args.calc_cos_init:
        model_path = os.path.join(args.models_dir, f"checkpoint-0.pt")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        params_init_si = [p.clone() for n, p in model.named_parameters() if check_si_name(n, args.model)]

    if 0 in results['report_epochs'] and args.calc_l2_init:
        model_path = os.path.join(args.models_dir, f"checkpoint-0.pt")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        params_init = [p.clone() for n, p in model.named_parameters()]

    criterion = squared_loss if args.use_squared_loss else cross_entropy

    for epoch in results['report_epochs']:
        if args.save_freq_int > 0:
            model_path = os.path.join(args.models_dir, f"checkpoint-{epoch_indexes[epoch]}.pt")
        else:
            model_path = os.path.join(args.models_dir, f"checkpoint-{epoch}.pt")
        print("Loading model: ", model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])

        #with torch.no_grad():
        #    results["params_norm"].append(np.sqrt(sum((p ** 2).sum().item() for p in model.parameters())))
        #    if args.all_pnorm:
        #        for n, p in model.named_parameters():
        #            results[n + '_norm'].append(p.norm().item())
                    
        with torch.no_grad():
            allowed_pnames = pnames if pnames is not None else [n for n, _ in model.named_parameters()]
            results["params_norm"].append(np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if n in allowed_pnames)))
            results["params_numel"].append(sum(p.numel() for n, p in model.named_parameters() if n in allowed_pnames))
            if args.all_pnorm:
                for n, p in model.named_parameters():
                    if n in allowed_pnames:
                        results[n + '_norm'].append(p.norm().item())
                        results[n + '_numel'].append(p.numel())

        if args.eval_model:
            print("Evaluating model...")   

            if return_train_subsets:         
                loss_corrupt, acc_corrupt = eval_model(model, loaders["train_corrupt"],criterion,
                                                       args.train_mode, False)
                corrupt_size = len(loaders["train_corrupt"].dataset)
                results['loss_corrupt'].append(loss_corrupt / corrupt_size)
                results['acc_corrupt'].append(acc_corrupt / corrupt_size)

                loss_normal, acc_normal = eval_model(model, loaders["train_normal"],criterion,
                                                     args.train_mode, False)
                normal_size = len(loaders["train_normal"].dataset)
                results['loss_normal'].append(loss_normal / normal_size)
                results['acc_normal'].append(acc_normal / normal_size)

                loss = (loss_corrupt + loss_normal) / len(loader.dataset)
                acc = (acc_corrupt + acc_normal) / len(loader.dataset)
            else:         
                loss, acc = eval_model(model, loader, criterion, args.train_mode)

            results['loss'].append(loss)
            results['acc'].append(acc)

        if 0 in results['report_epochs'] and args.calc_cos_init:
            print("Calculating cos with init...")
            with torch.no_grad():
                results["cos_init"].append(calc_cos(params_init_si, 
                                                    [p for n, p in model.named_parameters() if check_si_name(n, args.model)]))

        if 0 in results['report_epochs'] and args.calc_l2_init:
            print("Calculating l2 with init...")
            with torch.no_grad():
                results["l2_init"].append(calc_l2(params_init,
                                                    [p for n, p in model.named_parameters()]))

        if args.calc_grads:
            print("Calculating gradients...")
            grads_list = calc_grads(model, loader, criterion, args.train_mode, pnames=pnames)
            with torch.no_grad():
                gm, gs = [], []
                gnorms = []
                
                for p_grads in zip(*grads_list):  # taking all the gradients w.r.t. a particular parameter
                    p_grads = torch.stack(p_grads)
                    gm.append(p_grads.mean(0))
                    gs.append(p_grads.std(0))
                
                results["gm_norm"].append(np.sqrt(sum((g ** 2).sum().item() for g in gm)))
                results["gs_norm"].append(np.sqrt(sum((g ** 2).sum().item() for g in gs)))
                #print(results["gm_norm"][-1]**2+results["gs_norm"][-1]**2)
        torch.cuda.empty_cache()

        if args.calc_batch_outliers and args.calc_grad_norms_all:
            print("Calculating outliers and gradients norms...")
            outliers_list, gnorms_list = get_batch_outliers_gnorms(model, loader, criterion,
                                                                   args.train_mode, pnames=pnames)
            results["outliers_list"].append(outliers_list)
            results["gnorm_all"].append(gnorms_list)

        if args.calc_probs_stats:
            print("Calculating probs statistics per each correct class...")
            p_stats = calc_probs_stats(model, loader, args.train_mode)
            results["p_stats"].append(p_stats)

        if args.calc_grad_norms:
            print("Calculating gradients norms...")
            grads_list = calc_grads(model, loader, criterion, args.train_mode, pnames=pnames)
            with torch.no_grad():
                gm, gs = [], []
                gnorms = []
                
                for p_grads in grads_list:  
                    gnorm = 0
                    for t in p_grads:
                        gnorm += (t**2).sum().item()
                    gnorms.append(np.sqrt(gnorm))
                results["gnorm_m"].append(np.array(gnorms).mean())
        torch.cuda.empty_cache()

        if args.calc_grad_norms_small_memory:
            print("Calculating gradients norms...")
            gnorm = calc_grads_norms_small_memory(model, loader, args.train_mode, pnames=pnames)
            results["gnorm_m"].append(np.array(gnorm).mean())
            # print(results["gnorm_m"][-1])
        torch.cuda.empty_cache()

        if args.calc_grad_norms_param_groups:
            print("Calculating gradients norms for all param groups...")
            grads_list = calc_grads(model, loader, criterion, args.train_mode, pnames=pnames)
            with torch.no_grad():
                gnorms = {}
                gnorms[0] = np.zeros((32,))
                gnorms[1] = np.zeros((64,))
                gnorms[2] = np.zeros((128,))
                gnorms[3] = np.zeros((256,))

                for p_grads in grads_list:
                    for k in [0, 1, 2, 3]:
                        gnorms[k] += np.sqrt((p_grads[k] ** 2).reshape(gnorms[k].shape[0], -1).sum(-1).cpu().detach().numpy())
                for k in [0, 1, 2, 3]:
                    gnorms[k] / len(grads_list)
                results["gnorm_m_param_groups"].append(gnorms)
        torch.cuda.empty_cache()

        def custom_eval_trace(fisher):
            prefix = 'F' if fisher else 'H'
            print(f"Calculating tr({prefix}) / N...")
            prefix = 'fisher_' if fisher else 'hess_'
            
            if return_train_subsets:   
                trace_corrupt = eval_trace(model, loaders["train_corrupt"], fisher, args.train_mode, pnames=pnames,
                                           criterion=criterion)
                results[prefix + 'trace_corrupt'].append(trace_corrupt)
                w_corrupt = len(loaders["train_corrupt"].dataset) / len(loader.dataset)

                trace_normal = eval_trace(model, loaders["train_normal"], fisher, args.train_mode, pnames=pnames,
                                          criterion=criterion)
                results[prefix + 'trace_normal'].append(trace_normal)
                w_normal = len(loaders["train_normal"].dataset) / len(loader.dataset)

                trace = w_corrupt * trace_corrupt + w_normal * trace_normal
            else:
                trace = eval_trace(model, loader, fisher, args.train_mode, pnames=pnames, criterion=criterion)

            results[prefix + 'trace'].append(trace)

        if args.fisher_trace:
            custom_eval_trace(True)
        elif args.hess_trace:
            custom_eval_trace(False)

        def custom_eval_eigs(fisher):
            prefix = 'F' if fisher else 'H'
            print(f"Calculating eig({prefix})...")
            prefix = 'fisher_' if fisher else 'hess_'

            eigvals, eigvecs = eval_eigs(model, loader, fisher, args.train_mode, pnames=pnames, criterion=criterion)
            eigvals = eigvals.cpu().numpy()
            results[prefix + 'evals'].append(eigvals)

        if args.fisher_evals:
            custom_eval_eigs(True)
        elif args.hess_evals:
            custom_eval_eigs(False)

        print()

    for k, v in results.items():
        results[k] = np.array(v)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.update and os.path.isfile(args.save_path):
        print("Updating the old file")
        results_old = dict(np.load(args.save_path))
        results_old.update(results)
        results = results_old

    print("Saving all results to ", args.save_path)
    np.savez(args.save_path, **results)

    print()
    print(100 * '=')
    print()


if __name__ == '__main__':
    main()
