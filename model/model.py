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

class NeuroGNN_Encoder(nn.Module):
    def __init__(self, input_dim, seq_length, nodes_num=19,
                 semantic_embs=None, semantic_embs_dim=168,
                 dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu', gru_dim=128, num_heads=8,
                 conv_hidden_dim=128, conv_num_layers=3,
                 output_dim=128,
                 dist_adj=None):
        super(NeuroGNN_Encoder, self).__init__()
        self.gru_dim = gru_dim
        # self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.nodes_num = nodes_num
        self.seq1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(self.gru_dim)
        self.time_attention = Attention(self.gru_dim, self.gru_dim)
        self.mhead_attention = nn.MultiheadAttention(self.gru_dim, num_heads, dropout_rate, device=device, batch_first=True)
        
        self.GRU_cells = nn.ModuleList(
            nn.GRU(128, gru_dim, batch_first=True) for _ in range(self.nodes_num)
        )
        
        # self.fc_ta = nn.Linear(gru_dim, self.time_step) #TODO remove this
        self.fc_ta = nn.Linear(gru_dim, 128)
        
        for i, cell in enumerate(self.GRU_cells):
            cell.flatten_parameters()

        # TODO: Remove semantic embeddings condition
        self.semantic_embs = torch.from_numpy(semantic_embs).to(device).float()
        
        self.linear_semantic_embs = nn.Linear(self.semantic_embs.shape[1], semantic_embs_dim) 
                
       
        # self.node_feature_dim = time_step + semantic_embs_dim
        self.node_feature_dim = 128 + semantic_embs_dim
        
    


        self.convs = nn.ModuleList()

        self.conv_hidden_dim = conv_hidden_dim
        self.conv_layers_num = conv_num_layers
        
        self.output_dim = output_dim

        self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim)) 

        for l in range(self.conv_layers_num-1):
            self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))


        self.fc = nn.Sequential(
            nn.Linear(int(self.conv_hidden_dim + self.node_feature_dim), int(self.output_dim)),
            nn.ReLU()
        )

        if dist_adj is None:
            self.dist_adj = torch.ones((self.nodes_num, self.nodes_num)).to(device).float()
        else:
            self.dist_adj = torch.from_numpy(dist_adj).to(device).float()
        
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

        


    def latent_correlation_layer(self, x):
        # batch_size, _, node_cnt = x.shape
        batch_size, seq_len, node_cnt, input_dim = x.shape
        # (node_cnt, batch_size, seq_len, input_dim)
        new_x = x.permute(2, 0, 1, 3)
        weighted_res = torch.empty(batch_size, node_cnt, self.gru_dim).to(x.get_device())
        for i, cell in enumerate(self.GRU_cells):
            cell.flatten_parameters()
            x_sup = self.seq1(new_x[i])
            gru_outputs, hid = cell(x_sup)
            hid = hid.squeeze(0)
            gru_outputs = gru_outputs.permute(1, 0, 2).contiguous()
            weights = self.time_attention(hid, gru_outputs)
            updated_weights = weights.unsqueeze(1)
            gru_outputs = gru_outputs.permute(1, 0, 2)
            weighted = torch.bmm(updated_weights, gru_outputs)
            weighted = weighted.squeeze(1)
            weighted_res[:, i, :] = self.layer_norm(weighted + hid)
        _, attention = self.mhead_attention(weighted_res, weighted_res, weighted_res)

        attention = torch.mean(attention, dim=0) #[2000, 2000]
        attention = self.drop_out(attention)

        return attention, weighted_res



    def forward(self, x, static_features=None):
        attention, weighted_res = self.latent_correlation_layer(x) 
        mhead_att_mat = attention.detach().clone()

        # attention_mask = self.attention_thres(attention)
        # attention[~attention_mask] = 0


        
        
        weighted_res = self.fc_ta(weighted_res)
        weighted_res = F.relu(weighted_res)
        # weighted_res = F.relu(weighted_res)
        
        # X = weighted_res.unsqueeze(1).permute(0, 1, 2, 3).contiguous() 
        X = weighted_res.permute(0, 1, 2).contiguous()
        #TODO: should I add the static features (e.g., POI category vec) here??
        #TODO: appending static features vec to X
        if self.semantic_embs is not None:
            transformed_embeds = self.linear_semantic_embs(self.semantic_embs.to(x.get_device()))
            # transformed_embeds = self.semantic_embs.to(x.get_device())
            transformed_embeds = transformed_embeds.unsqueeze(0).repeat(X.shape[0], 1, 1)
            X = torch.cat((X, transformed_embeds), dim=2)
            
        embed_att = self.get_embed_att_mat_cosine(transformed_embeds)
        self.dist_adj = self.dist_adj.to(x.get_device())
        # attention = (((self.dist_adj + embed_att)/2) * attention)
        attention = ((self.att_alpha*self.dist_adj) + (1-self.att_alpha)*embed_att) * attention
        adj_mat_unthresholded = attention.detach().clone()
        # attention = ((self.dist_adj + embed_att) * attention)
        
        attention_mask = self.case_amplf_mask(attention)
        
        # attention_mask = self.attention_thres(attention)
        attention[~attention_mask] = 0
        adj_mat_thresholded = attention.detach().clone()
        
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(attention)
        
        # X = self.GLUs(X)
        

        
        X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        X_gnn = F.relu(X_gnn)
        for stack_i in range(1, self.conv_layers_num):
            X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
            X_gnn = F.relu(X_gnn)
        X_hat = torch.cat((X, X_gnn), dim=2)
        # TODO: Fix this part
        return self.fc(X_hat)
        # if forecast.size()[-1] == 1:
        #     return forecast.unsqueeze(1).squeeze(-1), attention.unsqueeze(0), (adj_mat_thresholded, adj_mat_unthresholded, embed_att, self.dist_adj, mhead_att_mat)
        # else:
        #     return forecast.squeeze(1).permute(0, 2, 1).contiguous(), attention.unsqueeze(0), (adj_mat_thresholded, adj_mat_unthresholded, embed_att, self.dist_adj, mhead_att_mat)
        
    
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
    
    
    def case_amplf_mask(self, attention, p=2.5, threshold=0.15):
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
class NeuroGNN_Classification(nn.Module):
    def __init__(self, args, num_classes, device=None, dist_adj=None, initial_sem_embeds=None):
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

        self.encoder = NeuroGNN_Encoder(input_dim=enc_input_dim,
                                        seq_length=args.max_seq_len,
                                        output_dim=self.rnn_units,
                                        dist_adj=dist_adj,
                                        semantic_embs=initial_sem_embeds
                                        )

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths=None):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        node_embeds = self.encoder(input_seq)
        logits = self.fc(self.relu(self.dropout(node_embeds)))
        
        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
        
