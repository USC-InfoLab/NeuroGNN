"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from contextlib import contextmanager
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import OrderedDict, defaultdict
from itertools import repeat
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from scipy.sparse import linalg
import sklearn
import matplotlib.cm as cm
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import math
import tqdm
import shutil
import queue
import random
import time
import json
import torch
import h5py
import logging
import numpy as np
import os
import sys
import pickle
import scipy.sparse as sp
import wandb
from sentence_transformers import SentenceTransformer

MASK = 0.
LARGE_NUM = 1e9


class WandbLogger():
    def __init__(self, project, is_used, name=None):

        self.is_used = is_used
        if is_used and not name:
            wandb.init(project=project)
        elif is_used and name:
            wandb.init(project=project, name=name)

    def watch_model(self,model):
        if self.is_used:
            wandb.watch(model)

    def log_hyperparams(self, params):
        if self.is_used:
            wandb.config.update(params)

    def log_metrics(self, metrics):
        if self.is_used:
            wandb.log(metrics)

    def log(self, key, value,  round_idx):
        if self.is_used:
            wandb.log({key: value, "Round": round_idx})

    def log_str(self, key, value):
        if self.is_used:
            wandb.log({key: value})


    def save_file(self, path):
        if path is not None and os.path.exists(path) and self.is_used:
            wandb.save(path)

@contextmanager
def timer(name="Main", logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0} s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def seed_torch(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_save_dir(base_dir, training, id_max=500):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(
            base_dir, subdir, '{}-{:02d}'.format(subdir, uid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, metric_name, maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print('Saver will {}imize {}...'
                    .format('max' if maximize_metric else 'min', metric_name))

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val <= metric_val)
                or (not self.maximize_metric and self.best_val >= metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, epoch, model, optimizer, metric_val):
        """Save model parameters to disk.
        Args:
            epoch (int): Current epoch.
            model (torch.nn.DataParallel): Model to save.
            optimizer: optimizer
            metric_val (float): Determines whether checkpoint is best so far.
        """
        ckpt_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        checkpoint_path = os.path.join(self.save_dir, 'last.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ''
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print('New best checkpoint at epoch {}...'.format(epoch))


def load_model_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer

    return model


def build_finetune_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to DCRNN model
    """
    # Load in pre-trained parameters
    for l in range(num_rnn_layers):
        model_new.encoder.encoding_cells[l].dconv_gate = model_pretrained.encoder.encoding_cells[l].dconv_gate
        model_new.encoder.encoding_cells[l].dconv_candidate = model_pretrained.encoder.encoding_cells[l].dconv_candidate

    return model_new

class AverageMeter:
    """Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """
    State transition matrix D_o^-1W in paper.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    """
    Reverse state transition matrix D_i^-1W^T in paper.
    """
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def count_parameters(model):
    """
    Counter total number of parameters, for Pytorch
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_dict(y_pred, y, y_prob=None, file_names=None, average='macro'):
    """
    Args:
        y_pred: Predicted labels of all samples
        y : True labels of all samples
        file_names: File names of all samples
        average: 'weighted', 'micro', 'macro' etc. to compute F1 score etc.
    Returns:
        scores_dict: Dictionary containing scores such as F1, acc etc.
        pred_dict: Dictionary containing predictions
        true_dict: Dictionary containing labels
    """

    scores_dict = {}
    pred_dict = defaultdict(list)
    true_dict = defaultdict(list)

    # write into output dictionary
    if file_names is not None:
        for idx, f_name in enumerate(file_names):
            pred_dict[f_name] = y_pred[idx]
            true_dict[f_name] = y[idx]

    if y is not None:
        scores_dict['acc'] = accuracy_score(y_true=y, y_pred=y_pred)
        scores_dict['F1'] = f1_score(y_true=y, y_pred=y_pred, average=average)
        scores_dict['precision'] = precision_score(
            y_true=y, y_pred=y_pred, average=average)
        scores_dict['recall'] = recall_score(
            y_true=y, y_pred=y_pred, average=average)
        if y_prob is not None:
            if len(set(y)) <= 2:  # binary case
                scores_dict['auroc'] = roc_auc_score(y_true=y, y_score=y_prob)

    return scores_dict, pred_dict, true_dict


def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / \
            (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh


def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def build_sparse_matrix(L):
    """
    Build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    """
    shape = L.shape
    i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
    v = torch.FloatTensor(L.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def compute_sampling_threshold(cl_decay_steps, global_step):
    """
    Compute scheduled sampling threshold
    """
    return cl_decay_steps / \
        (cl_decay_steps + np.exp(global_step / cl_decay_steps))


class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean  # (1,num_nodes,1)
        self.std = std  # (1,num_nodes,1)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, is_tensor=False, device=None, mask=None):
        """
        Masked inverse transform
        Args:
            data: data for inverse scaling
            is_tensor: whether data is a tensor
            device: device
            mask: shape (batch_size,) nodes where some signals are masked
        """
        mean = self.mean.copy()
        std = self.std.copy()
        if len(mean.shape) == 0:
            mean = [mean]
            std = [std]
        if is_tensor:
            mean = torch.FloatTensor(mean)
            std = torch.FloatTensor(std)
            if device is not None:
                mean = mean.to(device)
                std = std.to(device)
            #mean = torch.FloatTensor([mean])
            #std = torch.FloatTensor([std])

        return (data * std + mean)


def masked_mae_loss(y_pred, y_true, mask_val=0.):
    """
    Only compute loss on unmasked part
    """
    masks = (y_true != mask_val).float()
    masks /= masks.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_mse_loss(y_pred, y_true, mask_val=0.):
    """
    Only compute MSE loss on unmasked part
    """
    masks = (y_true != mask_val).float()
    masks /= masks.mean()
    loss = (y_pred - y_true).pow(2)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    loss = torch.sqrt(torch.mean(loss))
    return loss


def compute_regression_loss(
        y_true,
        y_predicted,
        standard_scaler=None,
        device=None,
        loss_fn='mae',
        mask_val=0.,
        is_tensor=True):
    """
    Compute masked MAE loss with inverse scaled y_true and y_predict
    Args:
        y_true: ground truth signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        y_predicted: predicted signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        standard_scaler: class StandardScaler object
        device: device
        mask: int, masked node ID
        loss_fn: 'mae' or 'mse'
        is_tensor: whether y_true and y_predicted are PyTorch tensor
    """
    if device is not None:
        y_true = y_true.to(device)
        y_predicted = y_predicted.to(device)

    if standard_scaler is not None:
        y_true = standard_scaler.inverse_transform(y_true,
                                                   is_tensor=is_tensor,
                                                   device=device)

        y_predicted = standard_scaler.inverse_transform(y_predicted,
                                                        is_tensor=is_tensor,
                                                        device=device)

    if loss_fn == 'mae':
        return masked_mae_loss(y_predicted, y_true, mask_val=mask_val)
    else:
        return masked_mse_loss(y_predicted, y_true, mask_val=mask_val)


def get_semantic_embeds():
    llm = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Regions of the brain
    cortex_regions = [
        'Frontal Lobe',
        'Parietal Lobe',
        'Left Temporal Lobe',
        'Right Temporal Lobe',
        'Occipital Lobe',
        'Central Region'
    ]

    # Dictionary describing eeg electrodes
    electrodes_descriptions = {
        'EEG FP1': 'Fp1 is an EEG electrode positioned at the frontopolar region of the scalp on the left side. It is commonly used to record electrical brain activity in that specific area. The Fp1 electrode is important for capturing frontal lobe activity, including cognitive processes such as attention, decision-making, and emotional responses. It plays a crucial role in EEG monitoring and can provide valuable insights into brain function.',
        'EEG FP2': 'Fp2 is an EEG electrode placed at the frontopolar region of the scalp on the right side. It is similar to Fp1 in terms of function and purpose. By recording electrical signals from the right frontopolar area, the Fp2 electrode helps monitor activity in the frontal lobe. This electrode can be instrumental in detecting abnormalities or changes in cognitive processing, emotional regulation, and other functions associated with the frontal brain regions.',
        'EEG F3': 'F3 is an EEG electrode positioned on the left side of the scalp, over the frontal lobe. It captures electrical brain activity from the left frontal region and plays a crucial role in monitoring cognitive processes, attention, and motor planning associated with the left hemisphere. The F3 electrode is essential for evaluating frontal lobe abnormalities and can provide valuable insights into conditions such as epilepsy, ADHD, and executive function disorders.',
        'EEG F4': "F4 is an EEG electrode situated on the right side of the scalp, mirroring F3's position. It records electrical signals originating from the right frontal lobe. Similar to F3, the F4 electrode is vital for assessing cognitive functions, attention, and motor planning associated with the right hemisphere. Monitoring the electrical activity in this area is crucial for detecting abnormalities or changes in brain function and can aid in the diagnosis and management of various neurological disorders.",
        'EEG C3': "C3 is an EEG electrode placed on the left side of the scalp, over the central region. It captures electrical brain activity from the left central area, including the sensorimotor cortex. The C3 electrode is important for studying motor control, movement planning, and somatosensory processing associated with the left hemisphere. Monitoring this region can help identify abnormalities or disruptions in motor function and contribute to the evaluation of conditions such as stroke, movement disorders, and brain injuries.",
        'EEG C4': "C4 is an EEG electrode located on the right side of the scalp, corresponding to C3's position. It records electrical signals from the right central region, encompassing the sensorimotor cortex. Similar to C3, the C4 electrode is crucial for monitoring motor control, movement planning, and somatosensory processing associated with the right hemisphere. It plays a significant role in assessing motor function asymmetries and can aid in the diagnosis and treatment of conditions such as Parkinson's disease, motor cortex lesions, and focal seizures.",
        'EEG P3': "P3 is an EEG electrode positioned on the left side of the scalp, above the parietal lobe. It captures electrical brain activity from the left parietal region, which is involved in processes such as spatial awareness, attention, and sensory integration. The P3 electrode is essential for studying visuospatial processing, visual attention, and other functions associated with the left hemisphere's parietal areas. Monitoring this region can provide valuable insights into conditions like neglect syndrome, spatial processing disorders, and attentional deficits.",
        'EEG P4': "P4 is an EEG electrode situated on the right side of the scalp, mirroring P3's position. It records electrical signals from the right parietal lobe. The P4 electrode is instrumental in monitoring visuospatial processing, attention, and sensory integration associated with the right hemisphere's parietal regions. By assessing electrical activity in this area, it can help identify abnormalities or changes in brain function and contribute to the evaluation and management of conditions such as spatial neglect, visual attention disorders, and parietal lobe epilepsy.",
        'EEG O1': "O1 is an EEG electrode positioned on the left side of the scalp, over the occipital lobe. It captures electrical brain activity from the left occipital region, which is primarily responsible for visual processing and perception. The O1 electrode is essential for studying visual evoked potentials, visual attention, and other functions associated with the left hemisphere's occipital areas. Monitoring this region can provide valuable insights into conditions such as visual processing disorders, occipital lobe epilepsy, and visual hallucinations.",
        'EEG O2': "O2 is an EEG electrode situated on the right side of the scalp, corresponding to O1's position. It records electrical signals from the right occipital lobe. Similar to O1, the O2 electrode is vital for monitoring visual processing, visual attention, and perception associated with the right hemisphere's occipital areas. It plays a significant role in assessing visual function asymmetries and can aid in the diagnosis and management of conditions such as visual field defects, occipital seizures, and visual processing impairments.",
        'EEG F7': "F7 is an EEG electrode situated at the left frontotemporal region of the scalp. It captures electrical activity from the left side of the brain, specifically the frontal and temporal lobes. The F7 electrode plays a significant role in assessing brain functions related to language processing, memory, and emotion. It is particularly useful for investigating disorders like epilepsy and monitoring the presence of abnormal electrical patterns in these areas.",
        'EEG F8': "F8 is an EEG electrode positioned at the right frontotemporal region of the scalp. It complements F7 by capturing electrical brain activity from the right frontal and temporal lobes. The F8 electrode helps monitor cognitive functions associated with the right hemisphere, including language processing, memory retrieval, and emotional regulation. It is essential for identifying any asymmetries or abnormalities in these brain regions and can contribute to the diagnosis and treatment of various neurological conditions.",
        'EEG T3': "T3 is an EEG electrode located on the left side of the scalp, above the temporal lobe. It records electrical signals originating from the left temporal region of the brain. The T3 electrode is significant for monitoring auditory processing, language comprehension, and memory functions associated with the left hemisphere. It is commonly used in diagnosing and studying conditions such as temporal lobe epilepsy and language-related disorders.",
        'EEG T4': "T4 is an EEG electrode placed on the right side of the scalp, above the temporal lobe. It complements T3 by recording electrical brain activity from the right temporal region. By monitoring the right hemisphere's functions related to auditory processing, language comprehension, and memory, the T4 electrode assists in assessing brain activity asymmetries and identifying abnormalities in these areas. It is particularly useful in the evaluation of temporal lobe epilepsy and language disorders.",
        'EEG T5': "T5 is an EEG electrode placed on the left side of the scalp, above the temporal lobe, but slightly posterior to T3. It captures electrical brain activity from the left temporal region, particularly the superior and posterior aspects. The T5 electrode is important for studying auditory processing, language comprehension, and memory functions associated with the left hemisphere's superior temporal gyrus. Monitoring this region can provide valuable insights into conditions such as temporal lobe epilepsy, auditory processing disorders, and language impairments.",
        'EEG T6': "T6 is an EEG electrode located on the right side of the scalp, corresponding to T5's position. It records electrical signals from the right temporal region, particularly the superior and posterior aspects of the superior temporal gyrus. Similar to T5, the T6 electrode is crucial for monitoring auditory processing, language comprehension, and memory functions associated with the right hemisphere. It plays a significant role in assessing auditory function asymmetries and can aid in the diagnosis and treatment of conditions such as temporal lobe epilepsy, auditory hallucinations, and language disorders.",
        'EEG FZ': "FZ is an EEG electrode positioned at the midline of the scalp, between F3 and F4. It captures electrical brain activity from the frontal-central region known as the midline prefrontal cortex. The FZ electrode is important for studying cognitive processes, working memory, and attention regulation. Monitoring this midline region can provide valuable insights into executive functions, emotional regulation, and frontal lobe abnormalities. It is particularly useful in the evaluation of conditions such as attention deficit hyperactivity disorder (ADHD), frontal lobe epilepsy, and mood disorders.",
        'EEG CZ': "CZ is an EEG electrode placed at the midline of the scalp, between C3 and C4. It records electrical signals from the central-parietal region, encompassing the sensorimotor and somatosensory cortices. The CZ electrode is crucial for monitoring motor control, sensory integration, and somatosensory processing. It plays a significant role in assessing abnormalities or changes in sensorimotor functions and can contribute to the evaluation and treatment of conditions such as movement disorders, sensory processing disorders, and central seizures.",
        'EEG PZ': "PZ is an EEG electrode positioned at the midline of the scalp, between P3 and P4. It captures electrical brain activity from the parietal-occipital region, which encompasses the parietal lobes and the posterior aspects of the occipital lobes. The PZ electrode plays a crucial role in studying visuospatial processing, attention, and sensory integration in the parietal and occipital regions. It is particularly useful for monitoring visual-spatial cognition, visual attention, and multisensory integration processes. The PZ electrode can provide valuable insights into conditions such as spatial neglect, visuospatial processing disorders, and parietal lobe epilepsy. By monitoring electrical signals from this region, it contributes to the evaluation and understanding of brain activity patterns related to visuospatial perception and attentional processes."
    }

    # Dictionary describing brain regions
    cortex_regions_descriptions = {
        'Frontal Lobe': 'The frontal lobe, located at the front of the brain, is involved in a wide range of higher cognitive functions. It plays a crucial role in executive functions such as decision-making, problem-solving, planning, and reasoning. Additionally, the frontal lobe contributes to motor control, including the initiation and coordination of voluntary movements. It also influences personality, social behavior, emotional regulation, and attentional processes. EEG electrodes associated with the frontal lobe include Fp1, Fp2, F3, F4, F7, F8, and FZ.',
        'Parietal Lobe': 'The parietal lobe, situated near the top and back of the brain, is responsible for various functions related to sensory perception and spatial awareness. It integrates sensory information from different modalities, such as touch, temperature, and proprioception, to create our perception of the surrounding world. The parietal lobe is also involved in spatial cognition, attentional processes, and the coordination of movements. It helps us navigate our environment and manipulate objects in space. EEG electrodes associated with the parietal lobe include P3, P4, PZ, and sometimes T5 and T6.',
        'Left Temporal Lobe': "The left temporal lobe, located on the left side of the brain, is involved in several crucial functions. It plays a critical role in auditory processing, allowing us to perceive and interpret sounds. In particular, the left temporal lobe is involved in language comprehension and production, including the understanding and generation of spoken and written language. It houses important language-related structures, such as Wernicke's area, which contributes to language processing and semantic understanding. Additionally, the left temporal lobe is involved in memory formation, including the encoding and retrieval of verbal and auditory information. EEG electrodes associated with the left temporal lobe include T3, T5, and sometimes P3.",
        'Right Temporal Lobe': "The right temporal lobe, positioned on the right side of the brain, shares several functions with the left temporal lobe. It is involved in auditory processing, including the perception and interpretation of sounds. While the left temporal lobe primarily handles language functions, the right temporal lobe is crucial for the processing of non-verbal auditory information, such as music and environmental sounds. It also contributes to aspects of visual-spatial processing and facial recognition. Additionally, the right temporal lobe plays a role in memory formation, particularly in the retrieval of non-verbal and visuospatial memories. EEG electrodes associated with the right temporal lobe include T4, T6, and sometimes P4.",
        'Occipital Lobe': "The occipital lobe, located at the back of the brain, is primarily dedicated to visual processing. It receives and processes visual information from the eyes, allowing us to perceive and interpret the world around us. The occipital lobe contains specialized regions that process different aspects of vision, such as object recognition, color perception, and motion detection. It helps us form visual representations of the environment and allows us to recognize and identify objects, faces, and visual patterns. EEG electrodes associated with the occipital lobe include O1 and O2.",
        'Central Region': "The central region encompasses the sensorimotor cortex, which is responsible for motor control and somatosensory processing. It plays a critical role in planning and executing voluntary movements. The central region receives sensory information related to touch, pressure, pain, and temperature, providing us with a sense of our body's position, movement, and interaction with the environment. This region is involved in the coordination and modulation of movements, integrating sensory feedback with motor commands to ensure smooth and precise execution of actions. The central region helps us manipulate objects, perform complex motor tasks, and engage in activities requiring fine motor control. EEG electrodes associated with the central region include C3, C4, CZ, and sometimes F3 and F4."
    }

    # Dictionary mapping brain regions to eeg electrodes
    electrodes_regions = {
        'EEG FP1': 'Frontal Lobe',
        'EEG FP2': 'Frontal Lobe',
        'EEG F3': 'Frontal Lobe',
        'EEG F4': 'Frontal Lobe',
        'EEG C3': 'Central Region',
        'EEG C4': 'Central Region',
        'EEG P3': 'Parietal Lobe',
        'EEG P4': 'Parietal Lobe',
        'EEG O1': 'Occipital Lobe',
        'EEG O2': 'Occipital Lobe',
        'EEG F7': 'Frontal Lobe',
        'EEG F8': 'Frontal Lobe',
        'EEG T3': 'Left Temporal Lobe',
        'EEG T4': 'Right Temporal Lobe',
        'EEG T5': 'Left Temporal Lobe',
        'EEG T6': 'Right Temporal Lobe',
        'EEG FZ': 'Frontal Lobe',
        'EEG CZ': 'Central Region',
        'EEG PZ': 'Parietal Lobe'
    }
    
    descriptions = []
    for node, descp in electrodes_descriptions.items():
        descp = f'This node represents electrode {node.split()[1]} recordings. {descp}'
        descriptions.append(descp)
    # for node, descp in cortex_regions_descriptions.items():
    #     descp = f'This is a meta-node that represents the recordings for {node} region of the cortext. {descp}'
    #     descriptions.append(descp)

    embeddings = llm.encode(descriptions)
    return embeddings

    