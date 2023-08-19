"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data.data_utils import computeFFT
from model.cell import DCGRUCell
from torch.autograd import Variable
import utils
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor
    """
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x)
                     for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor
    """
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(
                xs,
                dim) if isinstance(
                xs[0],
                torch.Tensor) else xs[0]) for xs in zip(
                *
                tups))
    else:
        return torch.cat(tups, dim)


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='laplacian',
                 device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)  # (seq_len, batch_size, num_nodes * rnn_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device)  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, dcgru_activation=None,
                 filter_type='laplacian', device=None, dropout=0.0):
        super(DCGRUDecoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device
        self.dropout = dropout

        cell = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                         max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, nonlinearity=dcgru_activation,
                         filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            decoding_cells.append(cell)

        self.decoding_cells = nn.ModuleList(decoding_cells)
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)  # dropout before projection layer

    def forward(
            self,
            inputs,
            initial_hidden_state,
            supports,
            teacher_forcing_ratio=None):
        """
        Args:
            inputs: shape (seq_len, batch_size, num_nodes, output_dim)
            initial_hidden_state: the last hidden state of the encoder, shape (num_layers, batch, num_nodes * rnn_units)
            supports: list of supports from laplacian or dual_random_walk filters
            teacher_forcing_ratio: ratio for teacher forcing
        Returns:
            outputs: shape (seq_len, batch_size, num_nodes * output_dim)
        """
        seq_length, batch_size, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        go_symbol = torch.zeros(
            (batch_size,
             self.num_nodes *
             self.output_dim)).to(
            self._device)

        # tensor to store decoder outputs
        outputs = torch.zeros(
            seq_length,
            batch_size,
            self.num_nodes *
            self.output_dim).to(
            self._device)

        current_input = go_symbol  # (batch_size, num_nodes * input_dim)
        for t in range(seq_length):
            next_input_hidden_state = []
            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    supports, current_input, hidden_state)
                current_input = output
                next_input_hidden_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)

            projected = self.projection_layer(self.dropout(
                output.reshape(batch_size, self.num_nodes, -1)))
            projected = projected.reshape(
                batch_size, self.num_nodes * self.output_dim)
            outputs[t] = projected

            if teacher_forcing_ratio is not None:
                teacher_force = random.random() < teacher_forcing_ratio  # a bool value
                current_input = (inputs[t] if teacher_force else projected)
            else:
                current_input = projected

        return outputs


########## Model for seizure classification/detection ##########
class DCRNNModel_classification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(DCRNNModel_classification, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type)

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(
            batch_size).to(self._device)

        # last hidden state of the encoder is the context
        # (max_seq_len, batch, rnn_units*num_nodes)
        _, final_hidden = self.encoder(input_seq, init_hidden_state, supports)
        # (batch_size, max_seq_len, rnn_units*num_nodes)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)

        # extract last relevant output
        last_out = utils.last_relevant_pytorch(
            output, seq_lengths, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        last_out = last_out.to(self._device)

        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
########## Model for seizure classification/detection ##########


########## Model for next time prediction ##########
class DCRNNModel_nextTimePred(nn.Module):
    def __init__(self, args, device=None):
        super(DCRNNModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = args.num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type)
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, hid_dim=rnn_units,
                                    output_dim=output_dim,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=device,
                                    dropout=args.dropout)

    def forward(
            self,
            encoder_inputs,
            decoder_inputs,
            supports,
            batches_seen=None):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            encoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            supports: list of supports from laplacian or dual_random_walk filters
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).cuda()

        # encoder
        # (num_layers, batch, rnn_units*num_nodes)
        encoder_hidden_state, _ = self.encoder(
            encoder_inputs, init_hidden_state, supports)

        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio)  # (seq_len, batch_size, num_nodes * output_dim)
        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs
########## Model for next time prediction ##########




########## NeuroGNN Classes ##########  
class NeuroGNN_GraphConstructor(nn.Module):
    def __init__(self, input_dim, seq_length, nodes_num=19, meta_nodes_num=6,
                 semantic_embs=None, semantic_embs_dim=256,
                 dropout_rate=0.0, leaky_rate=0.2,
                 device='cpu', gru_dim=256, num_heads=8,
                 dist_adj=None, temporal_embs_dim=256, meta_node_indices=None):
        super(NeuroGNN_GraphConstructor, self).__init__()
        self.gru_dim = gru_dim
        # self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.nodes_num = nodes_num
        self.meta_nodes_num = meta_nodes_num

        self.node_cluster_mapping = meta_node_indices + [list(range(nodes_num, nodes_num+meta_nodes_num))]
        self.seq1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(self.gru_dim*2)
        self.layer_norm2 = nn.LayerNorm(512)
        self.time_attention = Attention(self.gru_dim, self.gru_dim)
        self.mhead_attention = nn.MultiheadAttention(self.gru_dim*2, num_heads, dropout_rate, device=device, batch_first=True)
        # self.GRU_cells = nn.ModuleList(
        #     nn.GRU(512, gru_dim, batch_first=True, bidirectional=True) for _ in range(self.nodes_num+self.meta_nodes_num)
        # )
        # self.GRU_cells = nn.ModuleList(
        #     BiGRUWithMultiHeadAttention(256, gru_dim, 4) for _ in range(self.nodes_num+self.meta_nodes_num)
        # )
        self.bigru = nn.GRU(512, gru_dim, batch_first=True, bidirectional=True)
        # self.bigru_layernorm = nn.LayerNorm(gru_dim * 2)
        # self.bigru = Attentional_BiGRU(512, gru_dim, 4)
        # self.biGRU_cells = nn.ModuleList(
        #     nn.GRU(512, gru_dim, batch_first=True, bidirectional=True) for _ in range(len(self.node_cluster_mapping))
        # )
            
        
        # self.fc_ta = nn.Linear(gru_dim, self.time_step) #TODO remove this
        self.fc_ta = nn.Linear(gru_dim*2, temporal_embs_dim)
        self.layer_norm3 = nn.LayerNorm(temporal_embs_dim)
        self.layer_norm_sem = nn.LayerNorm(semantic_embs_dim)
        
        # for i, cell in enumerate(self.GRU_cells):
        #     cell.flatten_parameters()
            

        self.semantic_embs = torch.from_numpy(semantic_embs).to(device).float()
        
        # self.linear_semantic_embs = nn.Sequential(
        #     nn.Linear(semantic_embs.shape[1], semantic_embs_dim),
        #     nn.ReLU()
        # )
        self.linear_semantic_embs = nn.Linear(self.semantic_embs.shape[1], semantic_embs_dim) 
        # self.semantic_embs_layer_norm = nn.LayerNorm(semantic_embs_dim)
                
       
        # self.node_feature_dim = time_step + semantic_embs_dim
        self.node_feature_dim = temporal_embs_dim + semantic_embs_dim
        
    

        if dist_adj is not None:
            self.dist_adj = dist_adj
            self.dist_adj = torch.from_numpy(self.dist_adj).to(device).float()
        
        self.att_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Dropout):
                m.p = dropout_rate
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                m.in_proj_bias.data.zero_()
                m.out_proj.bias.data.zero_()  

        self.to(device) 


    # def latent_correlation_layer(self, x):
    #     # batch_size, _, node_cnt = x.shape
    #     batch_size, seq_len, node_cnt, input_dim = x.shape
    #     # (node_cnt, batch_size, seq_len, input_dim)
    #     new_x = x.permute(2, 0, 1, 3)
    #     weighted_res = torch.empty(batch_size, node_cnt, self.gru_dim*2).to(x.get_device())
    #     for i, cell in enumerate(self.GRU_cells):
    #         # cell.flatten_parameters()
    #         cell.flatten_parameters()
    #         x_sup = self.seq1(new_x[i])
    #         gru_outputs, hid = cell(x_sup)
    #         out, hid = cell(x_sup)
    #         # TODO: multi-layer GRU?
    #         # hid = hid[-1, :, :]
    #         # hid = hid.squeeze(0)
    #         # gru_outputs = gru_outputs.permute(1, 0, 2).contiguous()
    #         # TODO: to or not to use self-attention?
    #         # weights = self.time_attention(hid, gru_outputs)
    #         # updated_weights = weights.unsqueeze(1)
    #         # gru_outputs = gru_outputs.permute(1, 0, 2)
    #         # weighted = torch.bmm(updated_weights, gru_outputs)
    #         # weighted = weighted.squeeze(1)
    #         # weighted_res[:, i, :] = self.layer_norm(weighted + hid)
    #         h_n = hid.permute(1, 0, 2)
    #         weighted_res[:, i, :] = h_n.reshape(batch_size, -1)
    #     _, attention = self.mhead_attention(weighted_res, weighted_res, weighted_res)

    #     attention = torch.mean(attention, dim=0) #[2000, 2000]
    #     # TODO: Should I put drop_out for attention?
    #     # attention = self.drop_out(attention)

    #     return attention, weighted_res
    
    # def latent_correlation_layer(self, x):
    #     # batch_size, _, node_cnt = x.shape
    #     batch_size, seq_len, node_cnt, input_dim = x.shape
    #     # (batch_size, node_cnt, seq_len, input_dim)
    #     new_x = x.permute(0, 2, 1, 3)
    #     weighted_res = torch.empty(batch_size, node_cnt, self.gru_dim*2).to(x.get_device())
    #     # get temporal contexts for nodes and meta-nodes
    #     for i, node_indices in enumerate(self.node_cluster_mapping):
    #         group_x = new_x[:, node_indices, :, :]
    #         group_x_reshaped = group_x.reshape(batch_size*len(node_indices), seq_len, input_dim)
    #         x_sup = self.seq1(group_x_reshaped)
    #         bigru_cell = self.biGRU_cells[i]
    #         bigru_cell.flatten_parameters()
    #         gru_outputs, hid = bigru_cell(x_sup)
    #         h_n = hid.permute(1, 0, 2)
    #         h_n_reshaped = h_n.reshape(batch_size, len(node_indices), -1)
    #         weighted_res[:, node_indices, :] = h_n_reshaped
    #     weighted_res = self.layer_norm(weighted_res)
    #     _, attention = self.mhead_attention(weighted_res, weighted_res, weighted_res)
    #     attention = torch.mean(attention, dim=0) #[2000, 2000]
    #     # TODO: Should I put drop_out for attention?
    #     # attention = self.drop_out(attention)

    #     return attention, weighted_res
    
    
    def latent_correlation_layer(self, x):
        batch_size, seq_len, node_cnt, input_dim = x.shape
        
        # Reshape x to combine the batch and node dimensions
        # New shape: (batch_size * node_cnt, seq_len, input_dim)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * node_cnt, seq_len, input_dim)

        # Pass x_reshaped through the desired layers (e.g., self.seq1 and the bigru)
        x_sup = self.seq1(x_reshaped)
        x_sup = self.layer_norm2(x_sup)
        self.bigru.flatten_parameters()
        gru_outputs, hid = self.bigru(x_sup)
        h_n = hid.permute(1, 0, 2)
        h_n_reshaped = h_n.reshape(batch_size, node_cnt, -1)
        
        # Apply Layer Normalization
        h_n_normalized = self.layer_norm(h_n_reshaped)

        _, attention = self.mhead_attention(h_n_normalized, h_n_normalized, h_n_normalized)

        attention = torch.mean(attention, dim=0) #[2000, 2000]

        return attention, h_n_normalized
    
    # def latent_correlation_layer(self, x):
    #     batch_size, seq_len, node_cnt, input_dim = x.shape
        
    #     # Reshape x to combine the batch and node dimensions
    #     # New shape: (batch_size * node_cnt, seq_len, input_dim)
    #     x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * node_cnt, seq_len, input_dim)

    #     # Pass x_reshaped through the desired layers (e.g., self.seq1 and the bigru)
    #     x_sup = self.seq1(x_reshaped)
    #     hid = self.bigru(x_sup)
    #     hid_reshaped = hid.reshape(batch_size, node_cnt, -1)

    #     _, attention = self.mhead_attention(hid_reshaped, hid_reshaped, hid_reshaped)

    #     attention = torch.mean(attention, dim=0) #[2000, 2000]

    #     return attention, hid_reshaped




    def forward(self, x):
        attention, weighted_res = self.latent_correlation_layer(x) 
        mhead_att_mat = attention.detach().clone()        
        
        weighted_res = self.fc_ta(weighted_res)
        weighted_res = F.relu(weighted_res)
        
        X = weighted_res.permute(0, 1, 2).contiguous()
        X = self.layer_norm3(X)
        if self.semantic_embs is not None:
            init_sem_embs = self.semantic_embs.to(x.get_device())
            transformed_embeds = self.linear_semantic_embs(init_sem_embs)
            # transformed_embeds = self.semantic_embs_layer_norm(transformed_embeds + init_sem_embs)
            # transformed_embeds = self.semantic_embs.to(x.get_device())
            transformed_embeds = self.layer_norm_sem(transformed_embeds)
            transformed_embeds = transformed_embeds.unsqueeze(0).repeat(X.shape[0], 1, 1)
            X = torch.cat((X, transformed_embeds), dim=2)
            
        embed_att = self.get_embed_att_mat_cosine(transformed_embeds)
        self.dist_adj = self.dist_adj.to(x.get_device())
        attention = ((self.att_alpha*self.dist_adj) + (1-self.att_alpha)*embed_att) * attention
        adj_mat_unthresholded = attention.detach().clone()
        
        
        attention_mask = self.case_amplf_mask(attention)
        
        attention[attention_mask==0] = 0
        adj_mat_thresholded = attention.detach().clone()
        

        # TODO: add softmax for attention?
        # attention = attention.softmax(dim=1)
        
        # X: Node features, attention: fused adjacency matrix
        return X, attention, (adj_mat_thresholded, adj_mat_unthresholded, embed_att, self.dist_adj, mhead_att_mat) 
        
    
    def _create_embedding_layers(self, embedding_size_dict, embedding_dim_dict, device):
        """construct the embedding layer, 1 per each categorical variable"""
        total_embedding_dim = 0
        cat_cols = list(embedding_size_dict.keys())
        embeddings = {}
        for col in cat_cols:
            embedding_size = embedding_size_dict[col]
            embedding_dim = embedding_dim_dict[col]
            total_embedding_dim += embedding_dim
            embeddings[col] = nn.Embedding(embedding_size, embedding_dim, device=device)
            
        return nn.ModuleDict(embeddings), total_embedding_dim
    
    
    def _normalize_attention(self, attention):
        # Normalize each row of the attention matrix
        max_scores, _ = torch.max(attention, dim=1, keepdim=True)
        norm_scores = attention / max_scores
        return norm_scores
    
    
    def case_amplf_mask(self, attention, p=2.5, threshold=0.08):
        '''
        This function computes the case amplification mask for a 2D attention tensor 
        with the given amplification factor p.

        Parameters:
            - attention (torch.Tensor): A 2D attention tensor of shape [n, n].
            - p (float): The case amplification factor (default: 2.5).
            - threshold (float): The threshold for the mask (default: 0.05).

        Returns:
            - mask (torch.Tensor): A 2D binary mask of the same size as `attention`,
              where 0s denote noisy elements and 1s denote clean elements.
        '''
        # Compute the maximum value in the attention tensor
        max_val, _ = torch.max(attention.detach(), dim=1, keepdim=True)

        # Compute the mask as per the case amplification formula
        mask = (attention.detach() / max_val) ** p

        # Turn the mask into a binary matrix, where anything below threshold will be considered as zero
        mask = torch.where(mask > threshold, torch.tensor(1).to(attention.device), torch.tensor(0).to(attention.device))
        return mask
        
        
    
    
    def get_embed_att_mat_cosine(self, embed_tensor):
        # embe_vecs: the tensor with shape (batch, POI_NUM, embed_dim)
        # Compute the dot product between all pairs of embeddings
        similarity_matrix = torch.bmm(embed_tensor, embed_tensor.transpose(1, 2))

        # Compute the magnitudes of each embedding vector
        magnitude = torch.norm(embed_tensor, p=2, dim=2, keepdim=True)

        # Normalize the dot product by the magnitudes
        normalized_similarity_matrix = similarity_matrix / (magnitude * magnitude.transpose(1, 2))

        # Apply a softmax function to obtain a probability distribution
        # similarity_matrix_prob = F.softmax(normalized_similarity_matrix, dim=2).mean(dim=0)
        
        return normalized_similarity_matrix.mean(dim=0).abs()
        # return similarity_matrix_prob
    
    
    def get_embed_att_mat_euc(self, embed_tensor):
        # Compute the Euclidean distance between all pairs of embeddings
        similarity_matrix = torch.cdist(embed_tensor[0], embed_tensor[0])

        # Convert the distances to similarities using a Gaussian kernel
        sigma = 1.0  # adjust this parameter to control the width of the kernel
        similarity_matrix = torch.exp(-similarity_matrix.pow(2) / (2 * sigma**2))

        # Normalize the similarity matrix by row
        # row_sum = similarity_matrix.sum(dim=1, keepdim=True)
        # similarity_matrix_prob = similarity_matrix / row_sum
        
        # return similarity_matrix
        return similarity_matrix
    
    
    
    
class NeuroGNN_StemGNN_Block(nn.Module):
    def __init__(self, node_feature_dim,
                 device='cpu', output_dim=128, stack_cnt=2, nodes_num=24, multi_layer=5, conv_hidden_dim=32):
        super(NeuroGNN_StemGNN_Block, self).__init__()
        self.node_feature_dim = node_feature_dim      
        self.output_dim = output_dim
        self.stack_cnt = stack_cnt
        self.nodes_num = nodes_num
        self.multi_layer = multi_layer
        self.conv_hidden_dim = conv_hidden_dim
        
        self.fc1 = nn.Linear(self.node_feature_dim, self.conv_hidden_dim)

        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.conv_hidden_dim, self.nodes_num, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        
        self.fc2 = nn.Sequential(
            nn.Linear(int(self.conv_hidden_dim), int(self.conv_hidden_dim)),
            nn.LeakyReLU(),
            nn.Linear(int(self.conv_hidden_dim), self.output_dim),
            # nn.Tanh(), #TODO: Delete this line
        )
        
        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        self.to(device)


    def forward(self, x, adj_mat):
        degree = torch.sum(adj_mat, dim=1)
        # laplacian is symmetric
        new_adj = 0.5 * (adj_mat + adj_mat.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                            torch.matmul(degree_l - new_adj, diagonal_degree_hat))
        mul_L = cheb_polynomial(laplacian) # [k, N, N] k=4, N=node_num
        
        X = x.unsqueeze(1).permute(0, 1, 2, 3).contiguous() # [batch_size, 1, node_num, feature_dim]
        X = self.fc1(X)
        
        result = []
        for stack_i in range(self.stack_cnt):
            output, X = self.stock_block[stack_i](X, mul_L)
            result.append(output)
        output = result[0] + result[1]
        output = self.fc2(output)

        return output
    
    

# class NeuroGNN_GNN_GCN(nn.Module):
#     def __init__(self, node_feature_dim,
#                  device='cpu', conv_hidden_dim=64, conv_num_layers=3):
#         super(NeuroGNN_GNN_GCN, self).__init__()
#         self.node_feature_dim = node_feature_dim      
#         self.conv_hidden_dim = conv_hidden_dim
#         self.conv_layers_num = conv_num_layers

#         self.convs = nn.ModuleList()
#         self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim)) 
#         for l in range(self.conv_layers_num-1):
#             self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))
        
#         # initialize weights using xavier
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.GRU):
#                 for name, param in m.named_parameters():
#                     if 'weight' in name:
#                         nn.init.xavier_normal_(param)
#                     elif 'bias' in name:
#                         param.data.zero_()
#             elif isinstance(m, nn.LayerNorm):
#                 m.bias.data.zero_()
#                 m.weight.data.fill_(1.0)
#         self.to(device)


#     def forward(self, X, adj_mat):
#         edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)
#         X_gnn = self.convs[0](X, edge_indices, edge_attrs)
#         X_gnn = F.relu(X_gnn)
#         for stack_i in range(1, self.conv_layers_num):
#             X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
#             X_gnn = F.relu(X_gnn)
#         return X_gnn


# class NeuroGNN_GNN_GCN(nn.Module):
#     def __init__(self, node_feature_dim,
#                  device='cpu', conv_hidden_dim=64, conv_num_layers=3):
#         super(NeuroGNN_GNN_GCN, self).__init__()
#         self.node_feature_dim = node_feature_dim      
#         self.conv_hidden_dim = conv_hidden_dim
#         self.conv_layers_num = conv_num_layers

#         self.convs = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()
#         self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim))
#         self.batch_norms.append(nn.BatchNorm1d(self.conv_hidden_dim))
#         for l in range(self.conv_layers_num-1):
#             self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))
#             self.batch_norms.append(nn.BatchNorm1d(self.conv_hidden_dim))
        
#         # initialize weights using xavier
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.GRU):
#                 for name, param in m.named_parameters():
#                     if 'weight' in name:
#                         nn.init.xavier_normal_(param)
#                     elif 'bias' in name:
#                         param.data.zero_()
#             elif isinstance(m, nn.LayerNorm):
#                 m.bias.data.zero_()
#                 m.weight.data.fill_(1.0)
#         self.to(device)


#     def forward(self, X, adj_mat):
#         edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)
#         X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        
#         # Reshape to combine batch and node dimensions
#         X_gnn = X_gnn.view(-1, self.conv_hidden_dim)
        
#         X_gnn = self.batch_norms[0](X_gnn)
#         X_gnn = F.relu(X_gnn)
        
#         # Reshape back to original shape
#         X_gnn = X_gnn.view(-1, X.size(1), self.conv_hidden_dim)
        
#         for stack_i in range(1, self.conv_layers_num):
#             X_res = X_gnn # Store the current state for the residual connection
#             X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
            
#             # Reshape, apply batch normalization, and reshape back
#             X_gnn = X_gnn.view(-1, self.conv_hidden_dim)
#             X_gnn = self.batch_norms[stack_i](X_gnn)
#             X_gnn = X_gnn.view(-1, X.size(1), self.conv_hidden_dim)
            
#             X_gnn = F.relu(X_gnn + X_res) # Add the residual connection
#         return X_gnn

class NeuroGNN_GNN_GCN(nn.Module):
    def __init__(self, node_feature_dim,
                 device='cpu', conv_hidden_dim=64, conv_num_layers=3):
        super(NeuroGNN_GNN_GCN, self).__init__()
        self.node_feature_dim = node_feature_dim      
        self.conv_hidden_dim = conv_hidden_dim
        self.conv_layers_num = conv_num_layers

        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False)) 
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.conv_hidden_dim) for _ in range(self.conv_layers_num-1)
        )
        for l in range(self.conv_layers_num-1):
            self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False))
        
        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        self.to(device)

    def forward(self, X, adj_mat):
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)
        X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        X_gnn = F.relu(X_gnn)
        for stack_i in range(1, self.conv_layers_num):
            X_res = X_gnn # Store the current state for the residual connection
            X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
            X_gnn = F.relu(self.layer_norms[stack_i-1](X_gnn + X_res)) # Add the residual connection
        return X_gnn

    
    
    
class NeuroGNN_GNN_GraphConv(nn.Module):
    def __init__(self, node_feature_dim,
                 device='cpu', conv_hidden_dim=64, conv_num_layers=3):
        super(NeuroGNN_GNN_GraphConv, self).__init__()
        self.node_feature_dim = node_feature_dim      
        self.conv_hidden_dim = conv_hidden_dim
        self.conv_layers_num = conv_num_layers

        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GraphConv(self.node_feature_dim, self.conv_hidden_dim)) 
        for l in range(self.conv_layers_num-1):
            self.convs.append(pyg_nn.GraphConv(self.conv_hidden_dim, self.conv_hidden_dim))
            
        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

        # (Other initialization code unchanged)

    def forward(self, X, adj_mat):
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)
        X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        X_gnn = F.relu(X_gnn)
        for stack_i in range(1, self.conv_layers_num):
            X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
            X_gnn = F.relu(X_gnn)
        return X_gnn

    

class NeuroGNN_Encoder(nn.Module):
    def __init__(self, input_dim, seq_length, nodes_num=19, meta_nodes_num=6,
                 semantic_embs=None, semantic_embs_dim=512,
                 dropout_rate=0.2, leaky_rate=0.2,
                 device='cpu', gru_dim=512, num_heads=8,
                 conv_hidden_dim=256, conv_num_layers=3,
                 output_dim=512,
                 dist_adj=None,
                 temporal_embs_dim=512,
                 gnn_block_type='gcn',
                 meta_node_indices=None):
        super(NeuroGNN_Encoder, self).__init__()
        self.graph_constructor = NeuroGNN_GraphConstructor(input_dim=input_dim, 
                                                           seq_length=seq_length, 
                                                           nodes_num=nodes_num, 
                                                           meta_nodes_num=meta_nodes_num,
                                                           semantic_embs=semantic_embs, 
                                                           semantic_embs_dim=semantic_embs_dim,
                                                           dropout_rate=dropout_rate, 
                                                           leaky_rate=leaky_rate,
                                                           device=device, 
                                                           gru_dim=gru_dim, 
                                                           num_heads=num_heads,
                                                           dist_adj=dist_adj,
                                                           temporal_embs_dim=temporal_embs_dim,
                                                           meta_node_indices=meta_node_indices)
        
        self.node_features_dim = temporal_embs_dim+semantic_embs_dim
        self.conv_hidden_dim = conv_hidden_dim
        self.output_dim = output_dim
        
        if gnn_block_type.lower() == 'gcn':
            # TODO: update conv_hidden_dim
            self.gnn_block = NeuroGNN_GNN_GCN(node_feature_dim=self.node_features_dim,
                                            device=device,
                                            conv_hidden_dim=conv_hidden_dim,
                                            conv_num_layers=conv_num_layers)
        elif gnn_block_type.lower() == 'stemgnn':
            self.gnn_block = NeuroGNN_StemGNN_Block(node_feature_dim=self.node_features_dim,
                                                    device=device,
                                                    output_dim=conv_hidden_dim,
                                                    stack_cnt=2,
                                                    conv_hidden_dim=conv_hidden_dim)
        elif gnn_block_type.lower() == 'graphconv':
            self.gnn_block = NeuroGNN_GNN_GraphConv(node_feature_dim=self.node_features_dim,
                                                    device=device,
                                                    conv_hidden_dim=conv_hidden_dim,
                                                    conv_num_layers=conv_num_layers)
        self.layer_norm = nn.LayerNorm(self.conv_hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(self.node_features_dim, self.conv_hidden_dim),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(int(self.conv_hidden_dim + self.node_features_dim), int(self.output_dim)),
            nn.ReLU()
        )
        self.to(device)

        
    def forward(self, x):
        X, adj_mat, (adj_mat_thresholded, adj_mat_unthresholded, embed_att, dist_adj, mhead_att_mat) = self.graph_constructor(x)
        X_gnn = self.gnn_block(X, adj_mat)
        # TODO: best way to make X_hat?
        # X_hat = torch.cat((X, X_gnn), dim=2)
        # X_hat = self.fc(X_hat)
        X = self.fc1(X)
        X_hat = self.layer_norm(X_gnn + X)
        return X_hat, adj_mat, (adj_mat_thresholded, adj_mat_unthresholded, embed_att, dist_adj, mhead_att_mat)
        




class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2), dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
    
    
    
    
    
########## Model for seizure classification/detection ##########
# class NeuroGNN_Classification(nn.Module):
#     def __init__(self, args, num_classes, device=None, dist_adj=None, initial_sem_embeds=None, self.metanodes_num=6):
#         super(NeuroGNN_Classification, self).__init__()

#         num_nodes = args.num_nodes
#         num_rnn_layers = args.num_rnn_layers
#         rnn_units = args.rnn_units
#         enc_input_dim = args.input_dim

#         self.num_nodes = num_nodes
#         self.num_rnn_layers = num_rnn_layers
#         self.rnn_units = rnn_units
#         self._device = device
#         self.num_classes = num_classes
        
#         self.gnn_type = args.gnn_type

#         self.encoder = NeuroGNN_Encoder(input_dim=enc_input_dim,
#                                         seq_length=args.max_seq_len,
#                                         output_dim=self.rnn_units,
#                                         dist_adj=dist_adj,
#                                         semantic_embs=initial_sem_embeds,
#                                         gnn_block_type=self.gnn_type
#                                         )

#         # TODO: Update encoder dim input to linear layer
#         self.fc = nn.Linear(self.encoder.conv_hidden_dim, num_classes)
#         self.dropout = nn.Dropout(args.dropout)
#         self.relu = nn.ReLU()

#     def forward(self, input_seq, seq_lengths=None):
#         """
#         Args:
#             input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
#             seq_lengths: actual seq lengths w/o padding, shape (batch,)
#             supports: list of supports from laplacian or dual_random_walk filters
#         Returns:
#             pool_logits: logits from last FC layer (before sigmoid/softmax)
#         """
#         node_embeds, _, _ = self.encoder(input_seq)
#         logits = self.fc(self.relu(self.dropout(node_embeds)))
        
#         # max-pooling over nodes
#         # TODO: Mean pool or max pool
#         pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)
#         # pool_logits = torch.mean(logits, dim=1)

#         return pool_logits


class NeuroGNN_Classification(nn.Module):
    def __init__(self, args, num_classes, device=None, dist_adj=None, initial_sem_embeds=None, metanodes_num=6, meta_node_indices=None):
        super(NeuroGNN_Classification, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes
        self.metanodes_num = metanodes_num
        
        self.meta_node_indices = meta_node_indices
        
        self.gnn_type = args.gnn_type

        self.encoder = NeuroGNN_Encoder(input_dim=enc_input_dim,
                                        seq_length=args.max_seq_len,
                                        output_dim=self.rnn_units,
                                        dist_adj=dist_adj,
                                        semantic_embs=initial_sem_embeds,
                                        gnn_block_type=self.gnn_type,
                                        meta_node_indices=self.meta_node_indices
                                        )

        self.fc = nn.Linear(self.encoder.conv_hidden_dim, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()


    def forward(self, input_seq, seq_lengths=None, meta_node_indices=None):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            meta_node_indices: list of lists containing indices for each region
        Returns:
            logits: logits from last FC layer (before sigmoid/softmax)
        """
        node_embeds, _, _ = self.encoder(input_seq)
        pooled_embeddings = self.hierarchical_pooling(node_embeds, meta_node_indices=self.meta_node_indices)
        logits = self.fc(self.relu(self.dropout(pooled_embeddings)))

        return logits
    
    
    def hierarchical_pooling(self, node_embeddings, meta_node_indices):
        # Step 1: Pool Within Regions
        region_pooled_embeddings = [torch.mean(node_embeddings[:, indices, :], dim=1) for indices in meta_node_indices]
        region_pooled_embeddings = torch.stack(region_pooled_embeddings, dim=1) # Shape: (batch_size, num_regions, conv_dimension)

        # Step 2: Pool Across Meta Nodes
        meta_node_pooled_embeddings = torch.mean(node_embeddings[:, -self.metanodes_num:, :], dim=1) # Shape: (batch_size, conv_dimension)
        meta_node_pooled_embeddings = meta_node_pooled_embeddings.unsqueeze(1) # Add extra dimension, shape: (batch_size, 1, conv_dimension)
        
        # Step 3: Concatenate pooled embeddings
        all_pooled_embeddings = torch.cat([region_pooled_embeddings, meta_node_pooled_embeddings], dim=1) # Shape: (batch_size, num_regions + 1, conv_dimension)

        # Step 4: Max Pooling
        max_pooled_embeddings, _ = torch.max(all_pooled_embeddings, dim=1) # Shape: (batch_size, conv_dimension)

        return max_pooled_embeddings
    




        

########## Model for next time prediction ##########
class NeuroGNN_nextTimePred(nn.Module):
    def __init__(self, args, device=None, dist_adj=None, initial_sem_embeds=None, meta_nodes_num=6, meta_node_indices=None):
        super(NeuroGNN_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = 1
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        max_diffusion_step = args.max_diffusion_step
        dec_input_dim = args.output_dim

        self.num_nodes = args.num_nodes
        self.meta_nodes_num = meta_nodes_num
        self.num_rnn_layers = 1
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)
        
        self.gnn_type = args.gnn_type

        self.encoder = NeuroGNN_Encoder(input_dim=enc_input_dim,
                                        seq_length=args.max_seq_len,
                                        output_dim=self.rnn_units,
                                        dist_adj=dist_adj,
                                        semantic_embs=initial_sem_embeds,
                                        gnn_block_type=self.gnn_type,
                                        meta_node_indices=meta_node_indices
                                        )
        # TODO: update hid_dim
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes+meta_nodes_num, hid_dim=self.encoder.conv_hidden_dim,
                                    output_dim=output_dim,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=device,
                                    dropout=args.dropout)

    def forward(
            self,
            encoder_inputs,
            decoder_inputs,
            batches_seen=None):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            decoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # encoder
        # (num_layers, batch, rnn_units*num_nodes)
        # (batch, num_nodes, node_embedding_dim)
        encoder_hidden_state, adj_mat, _ = self.encoder(encoder_inputs)
        # should I detach adj_mat or not?
        supports = [adj_mat.repeat(batch_size, 1, 1)]  # (batch, num_nodes, num_nodes)
        
        #(num_layers, batch, node_embedding_dim*num_nodes)
        encoder_hidden_state = encoder_hidden_state.reshape(self.num_rnn_layers, batch_size, -1)

        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio)  # (seq_len, batch_size, num_nodes * output_dim)
        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs
########## Model for next time prediction ##########




########## StemGNN GNN Block model modules ##########
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))



class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):

        ### input: 32, 4, 1, 2000, 24
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step) ## 32, 4, 2000, 24
        #print(f"input{input.shape}")

        # ffted = torch.rfft(input, 1, onesided=False) ### older VERSION 32, 4, 2000, 24, 2
        # real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)### 32, 2000, 96
        # img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)


        ffted  = torch.fft.fft(input)
        real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)


        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous() ##32, 4, 2000, 120
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()

        # print(f"real{real.shape}")
        time_step_as_inner = torch.complex(real, img)
        iffted = torch.fft.ifft(time_step_as_inner)
        # print(f"iffted{iffted[0]}")

        # time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1) ### ordler 32, 4, 2000, 120, 2
        # iffted = torch.irfft(time_step_as_inner, 1, onesided=False) ###iffted =  32, 4, 2000, 120


        return iffted.real

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x) ## 32, 4, 1, 2000, 24
        #print(f"gfted {gfted.shape}")
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2) ### [32, 4, 1, 2000, 120]
        #print(f"gconv {gconv_input.shape}")
        igfted = torch.matmul(gconv_input, self.weight)
        #print(f"igfted  {igfted .shape}")
        igfted = torch.sum(igfted, dim=1) #### [32, 4, 1, 2000, 120]
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source
    
    
    
def cheb_polynomial(laplacian):
    """
    Compute the Chebyshev Polynomial, according to the graph laplacian.
    :param laplacian: the graph laplacian, [N, N].
    :return: the multi order Chebyshev laplacian, [K, N, N].
    """
    N = laplacian.size(0)  # [N, N]
    laplacian = laplacian.unsqueeze(0)
    # first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
    first_laplacian = torch.eye(N, device=laplacian.device, dtype=torch.float).unsqueeze(0) #TODO remove this
    second_laplacian = laplacian
    # third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
    third_laplacian = (2 * laplacian * second_laplacian) - first_laplacian #TODO remove this
    # forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
    forth_laplacian = 2 * laplacian * third_laplacian - second_laplacian #TODO remove this
    multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
    return multi_order_laplacian
    
    
    
class Attentional_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(Attentional_BiGRU, self).__init__()
        
        # Bidirectional GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        # Pass through GRU
        out, _ = self.gru(x) # out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Transpose for multi-head attention
        out_transposed = out.transpose(0, 1) # shape: (seq_len, batch_size, hidden_dim * 2)
        
        # Using the last hidden state as the query and the entire output as keys and values
        queries = out_transposed[-1:, :, :] # shape: (1, batch_size, hidden_dim * 2)
        keys = values = out_transposed # shape: (seq_len, batch_size, hidden_dim * 2)
        
        # Apply attention
        attention_out, _ = self.attention(queries, keys, values) # shape: (1, batch_size, hidden_dim * 2)
        
        # Squeeze to remove the sequence length dimension
        attention_out = attention_out.squeeze(0) # shape: (batch_size, hidden_dim * 2)
        
        # Add the last hidden state (from the original GRU output)
        final_out = attention_out + out[:, -1, :] # shape: (batch_size, hidden_dim * 2)
        
        # Apply layer normalization
        final_out = self.layer_norm(final_out)
        
        return final_out
