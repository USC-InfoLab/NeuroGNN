import numpy as np
import os
from datetime import datetime
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import math
import utils
from data.data_utils import *
from data.dataloader_ssl import load_dataset_ssl
from constants import *
from args import get_args
from collections import OrderedDict
from json import dumps
from model.model import DCRNNModel_nextTimePred
from model.model import NeuroGNN_nextTimePred
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import WandbLogger, get_extended_adjacency_matrix


def main(args):

    # Get device
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.device == 'cuda' else "cpu"

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)

    # Get save directories
    args.save_dir = utils.get_save_dir(f'{args.save_dir}/{args.model_name}', training=True)

    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logging
    run_name = f'{args.model_name}-window:{args.max_seq_len}-horizon:{args.output_seq_len}-{str(datetime.now().strftime("%Y-%m-%d %H:%M"))}'
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    wandb_logger = WandbLogger("EEG_Pretrain_Forecast", args.use_wandb, run_name)
    wandb_logger.log_hyperparams(args)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    
    augment_metaseries = True if args.model_name == 'neurognn' else False

    # Build dataset
    log.info('Building dataset...')
    dataloaders, _, scaler = load_dataset_ssl(
        input_dir=args.input_dir,
        raw_data_dir=args.raw_data_dir,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        time_step_size=args.time_step_size,
        input_len=args.max_seq_len,
        output_len=args.output_seq_len,
        standardize=True,
        num_workers=args.num_workers,
        augmentation=args.data_augment,
        adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
        graph_type=args.graph_type,
        top_k=args.top_k,
        filter_type=args.filter_type,
        use_fft=args.use_fft,
        preproc_dir=args.preproc_dir,
        augment_metaseries=augment_metaseries)

    # Build model
    log.info('Building model...')
    if args.model_name == "dcrnn":
        model = DCRNNModel_nextTimePred(device=device, args=args)
    elif args.model_name == "neurognn":
        distances_df = pd.read_csv('./data/electrode_graph/distances_3d.csv')
        dist_adj, _, _ = get_extended_adjacency_matrix(distances_df, INCLUDED_CHANNELS, ELECTRODES_REGIONS)
        initial_sem_embs = utils.get_semantic_embeds()
        model = NeuroGNN_nextTimePred(args, device, dist_adj, initial_sem_embs)
    wandb_logger.watch_model(model)

    num_params = utils.count_parameters(model)
    log.info('Total number of trainable parameters: {}'.format(num_params))

    if args.load_model_path is not None:
        model = utils.load_model_checkpoint(
            args.load_model_path, model)
    model = model.to(device)

    if args.do_train:
        train(model, dataloaders, args, device, args.save_dir, log, tbx,
            scaler=scaler, wandb_logger=wandb_logger)
        # Load best model after training finished
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)
    # Evaluate on test set
    log.info('Training DONE. Evaluating model...')
    test_loss = evaluate(model,
                         dataloaders['test'],
                         args,
                         args.save_dir,
                         device,
                         is_test=True,
                         nll_meter=None,
                         scaler=scaler)

    # Log to console
    log.info('Test set prediction MAE loss: {:.3f}'.format(test_loss))


def train(
        model,
        dataloaders,
        args,
        device,
        save_dir,
        log,
        tbx,
        scaler=None,
        wandb_logger=None):
    """
    Perform training and evaluate on dev set
    """
    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            for x, y, _, supports, _, _ in train_loader:
                batch_size = x.shape[0]

                # input seqs
                # (batch_size, input_seq_len, num_nodes, input_dim)
                x = x.to(device)
                # (batch_size, output_seq_len, num_nodes, output_dim)
                y = y.to(device)
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                # (batch_size, seq_len, num_nodes, output_dim)
                if args.model_name == "dcrnn":
                    seq_preds = model(x, y, supports, batches_seen=step)
                elif args.model_name == "neurognn":
                    seq_preds = model(x, y, batches_seen=step)
                    
                # TODO: Should I use MAE or MSE loss?
                loss = utils.compute_regression_loss(
                    y_true=y,
                    y_predicted=seq_preds,
                    loss_fn="mse",
                    standard_scaler=scaler,
                    device=device)
                
                # TODO start: Should I use hierarchical consistency loss?
                # Calculate the hierarchical consistency loss
                consistency_loss = hierarchical_loss(seq_preds, META_NODE_INDICES)
                total_loss = 0.9 * loss + 0.1 * consistency_loss
                # TODO end
                loss_val = loss.item()
                
                

                # TODO start: Backward
                # loss.backward()
                total_loss.backward()
                # TODO end
                # TODO: clip norm?
                # nn.utils.clip_grad_norm_(
                #     model.parameters(), args.max_grad_norm)
                optimizer.step()
                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         total_loss=total_loss.item(),
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/MAE Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                wandb_logger.log('train/MAE Loss', loss_val, step)
                wandb_logger.log('train/LR', optimizer.param_groups[0]['lr'], step)

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))
                eval_loss = evaluate(model,
                                     dev_loader,
                                     args,
                                     save_dir,
                                     device,
                                     is_test=False,
                                     nll_meter=nll_meter,
                                     scaler=scaler)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_loss)

                # Accumulate patience for early stopping
                if eval_loss < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_loss

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                log.info('Dev MAE loss: {:.3f}'.format(eval_loss))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                tbx.add_scalar(
                    'eval/{}'.format('MAE Loss'), eval_loss, step)
                wandb_logger.log('eval/{}'.format('MAE Loss'), eval_loss, step)

        # step lr scheduler
        scheduler.step()

def evaluate(model, dataloader, args, save_dir, device, is_test=False,
             nll_meter=None, scaler=None):
    # To evaluate mode
    model.eval()

    file_name_all = []
    y_truths = []
    y_preds = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for x, y, _, supports, _, file_name in dataloader:
            batch_size = x.shape[0]

            # input seqs
            # (batch_size, max_seq_len-1, num_nodes, input_dim)
            x = x.to(device)
            y = y.to(device)  # (batch_size, horizon, num_nodes, output_dim)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            # Forward
            # (batch_size, output_seq_len, num_nodes, output_dim)
            seq_preds = model(x, y, supports)
            if args.model_name == "dcrnn":
                seq_preds = model(x, y, supports)
            elif args.model_name == "neurognn":
                seq_preds = model(x, y)
                seq_preds = seq_preds[:, :, :args.num_nodes, :]
                y = y[:, :, :args.num_nodes, :]

            loss = utils.compute_regression_loss(
                y_true=y,
                y_predicted=seq_preds,
                loss_fn="mae",
                standard_scaler=scaler,
                device=device)

            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            file_name_all.extend(file_name)
            y_truths.append(y.cpu().numpy())
            y_preds.append(seq_preds.cpu().numpy())

            # Log info
            progress_bar.update(batch_size)

    # (all_samples, output_len, num_nodes, output_dim)
    y_truths = np.concatenate(y_truths, axis=0)
    # (all_samples, output_len, num_nodes, output_dim)
    y_preds = np.concatenate(y_preds, axis=0)

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()

    return eval_loss


def hierarchical_loss(y_hat, meta_node_indices):
    mse_loss = nn.MSELoss()

    total_loss = 0
    for i, indices in enumerate(meta_node_indices):
        meta_node_prediction = y_hat[:, :, 19+i, :]  # Get the prediction for the current meta node
        original_node_predictions = y_hat[:, :, indices, :]  # Get the predictions for the original nodes
        predicted_meta_value = original_node_predictions.mean(dim=2)  # Calculate the mean of the original node predictions

        total_loss += mse_loss(meta_node_prediction, predicted_meta_value)  # Add the MSE loss to the total loss

    return total_loss


if __name__ == '__main__':
    main(get_args())
