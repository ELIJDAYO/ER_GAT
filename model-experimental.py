import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, GATv2Conv, RGATConv
import numpy as np, itertools, random, copy, math
from torch_geometric.utils import add_self_loops
from tensorflow.keras.layers import Dense, Dropout
from graph_context_dataset import GraphContextDataset
from torch.utils.data import Dataset, DataLoader

# For methods and models related to DialogueGCN jump to line 516
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        #score = F.softmax(score, dim=-1)
        score = F.softmax(score, dim=0)
        # print (score)
        # print (sum(score))
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha
    
# class MatchingAttention(nn.Module):
#     def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general2'):
#         super(MatchingAttention, self).__init__()
#         self.mem_dim = mem_dim
#         self.cand_dim = cand_dim
#         self.alpha_dim = alpha_dim
#         self.att_type = att_type

#         if self.att_type == 'general2':
#             self.transform = nn.Linear(self.mem_dim, self.cand_dim * self.alpha_dim)

#     def forward(self, M, x, mask):
#         M_ = M.permute(1, 2, 0)  # (batch, mem_dim, seq_len)
#         x_ = self.transform(x).unsqueeze(1)  # (batch, 1, cand_dim * alpha_dim)
#         mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)  # (batch, mem_dim, seq_len)
        
#         M_ = M_ * mask_
#         alpha = torch.bmm(x_, M_)  # (batch, 1, seq_len)
        
#         alpha = F.softmax(alpha, dim=-1)  # Apply softmax to get attention weights
#         attended = torch.bmm(alpha, M.permute(1, 0, 2))  # (batch, 1, mem_dim)
        
#         return attended.squeeze(1), alpha
    
def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer):
    max_len = max(seq_lengths)
    batch_size = len(seq_lengths)
    mem_dim = emotions.size(1)

    padded_emotions = []
    for i in range(batch_size):
        length = seq_lengths[i]
        # Assuming emotions is already a 2D tensor of shape (seq_len, mem_dim)
        padded_emotion = F.pad(emotions[:length], (0, 0, 0, max_len - length), "constant", 0)
        padded_emotions.append(padded_emotion)

    emotions_padded = torch.stack(padded_emotions, dim=1)  # (max_len, batch_size, mem_dim)
    
    att_emotions = []
    alpha_list = []
    for t in range(max_len):
        att_em, alpha = matchatt_layer(emotions_padded, emotions_padded[t], umask)
        att_emotions.append(att_em)
        alpha_list.append(alpha)

    att_emotions = torch.stack(att_emotions, dim=0)  # (max_len, batch_size, mem_dim)

    # Remove the singleton dimension for batch size 1
    att_emotions = att_emotions.squeeze(1)  # (seq_len, mem_dim)

    return att_emotions, alpha_list
    
class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # Adjust input_dim to match flattened shape
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.activation2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.matchatt = MatchingAttention(mem_dim=input_dim, cand_dim=input_dim, alpha_dim=1, att_type='general2')

    def forward(self, x, nodalAtt, seq_lengths, umask):
        if nodalAtt:
            att_emotions, _ = attentive_node_features(x, seq_lengths, umask, self.matchatt)

            # Reshape att_emotions to have a 2D shape (batch_size, input_dim)
            att_emotions = att_emotions.view(att_emotions.size(0), -1)
            x = att_emotions
            
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class FCClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(FCClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.matchatt = MatchingAttention(mem_dim=input_dim, cand_dim=input_dim, alpha_dim=1, att_type='general2')
    def forward(self, x=None, nodalAtt=None, seq_lengths=None, umask=None, no_cuda=True):
        if nodalAtt:
            att_emotions, _ = attentive_node_features(x, seq_lengths, umask, self.matchatt)
            att_emotions = att_emotions.view(att_emotions.size(0), -1)
            x = att_emotions

        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        #score = F.softmax(score, dim=-1)
        score = F.softmax(score, dim=0)
        # print (score)
        # print (sum(score))
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score



class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper. 
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()
        
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        """
#         print("M -> (seq_len, batch, vector): ", M.shape)

        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            
            # Initialize masks
            batch_size, seq_len, _ = alpha.size()
            
            if not self.no_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()
                mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()
            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()

            # Prepare edge indices
            edge_ind_ = []
            for i, edges in enumerate(edge_ind):
                for x in edges:
                    if x[0] < seq_len and x[1] < seq_len:
                        edge_ind_.append([i, x[0], x[1]])
            
            edge_ind_ = np.array(edge_ind_).transpose()
        
            # Apply mask
            mask[edge_ind_[0], edge_ind_[1], edge_ind_[2]] = 1
            mask_copy[edge_ind_[0], edge_ind_[1], edge_ind_[2]] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy
            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            # if torch.cuda.is_available():
            if not self.no_cuda:
                scores = scores.cuda()


            for j in range(M.size(1)):
            
                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            #if torch.cuda.is_available():
            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):

                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    """

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
    
        
def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification 
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method 
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from 
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces 
    from any other dialogue instances in that mini-batch.
    """
    
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
#     print(features.size())
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []
    
    for j in range(batch_size):
#         print("lengths: ", lengths, " j: ", j)
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))
    
    # scores are the edge weights
    scores = att_model(features, lengths, edge_ind)
#     print("score.shape: ", scores.shape)
#     print("qmask.shape: ", qmask.shape)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))
    
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
#             print("item1[0] < qmask.shape[1] and item1[1] < qmask.shape[1]: ", item1[0], qmask.shape[1], item1[1], qmask.shape[1])
            try:
                if item1[0] < qmask.shape[1] and item1[1] < qmask.shape[1]:
                    speaker0 = (qmask[j, item1[0], :] == 1).nonzero(as_tuple=False)[0][0].tolist()
                    speaker1 = (qmask[j, item1[1], :] == 1).nonzero(as_tuple=False)[0][0].tolist()
#                     print(f"speaker0: {speaker0}, speaker1: {speaker1}")
                else:
                    raise IndexError(f"Index {item1[0]} or {item1[1]} is out of bounds for qmask.shape[1] {qmask.shape[1]}")

                if item1[0] < item1[1]:
                    # edge_type.append(0) # ablation by removing speaker dependency: only 2 relation types
                    # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                    edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
                else:
                    # edge_type.append(1) # ablation by removing speaker dependency: only 2 relation types
                    # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                    edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])
            except IndexError as e:
                print(f"IndexError at item1: {item1}, j: {j}")
                print(f"qmask.shape: {qmask.shape}")
                print(f"qmask[item1[0], j, :]: {qmask[item1[0], j, :]}")
                print(f"qmask[item1[1], j, :]: {qmask[item1[1], j, :]}")
                raise e
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    #if torch.cuda.is_available():
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()
    
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths 

def pad(tensor, length, no_cuda):
    # Pad the tensor to the given length
    if not no_cuda:
        pad_tensor = torch.zeros((length - tensor.size(0),) + tensor.size()[1:]).cuda()
    else:
        pad_tensor = torch.zeros((length - tensor.size(0),) + tensor.size()[1:])
    return torch.cat((tensor, pad_tensor), dim=0)


def validate_seq_lengths(seq_lengths):
    if not isinstance(seq_lengths, torch.Tensor):
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    if seq_lengths.dim() == 2:
        seq_lengths = seq_lengths.squeeze()
    if seq_lengths.dim() != 1:
        raise ValueError(f"seq_lengths should be a 1D tensor, but got {seq_lengths.dim()}D tensor.")
    return seq_lengths


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):
    """
    Function for the final classification, as in Equation 7, 8, 9. in the paper.
    """

    if nodal_attn:

        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob


class GraphNetwork_RGCN(nn.Module):
    def __init__(self, num_features, num_relations, hidden_size=64, dropout=0.5, no_cuda=False):
        super(GraphNetwork_RGCN, self).__init__()
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)

        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_type):
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        
        edge_index_cat = []
        edge_type_cat = []
        batch_offset = 0
        for e_i, e_t in zip(edge_index, edge_type):
            edge_index_cat.append(e_i + batch_offset)
            edge_type_cat.append(e_t)
            batch_offset += e_i.max().item() + 1
        
        edge_index = torch.cat(edge_index_cat, dim=1)
        edge_type = torch.cat(edge_type_cat, dim=0)

        torch.manual_seed(42)
        out = self.conv1(x, edge_index, edge_type)
        torch.manual_seed(42)
        out = self.conv2(out, edge_index)

        return out

class GraphNetwork_GAT(nn.Module):
    def __init__(self, num_features, num_relations, hidden_size=64, num_heads=8, dropout=0.5, no_cuda=True):
        super(GraphNetwork_GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, dropout=dropout, concat=False)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_type=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=0)

        # Correctly batch edge_index and edge_type (if edge_type is used)
        edge_index_cat = []
        batch_offset = 0
        for e_i in edge_index:
            edge_index_cat.append(e_i + batch_offset)
            batch_offset += e_i.max().item() + 1

        edge_index = torch.cat(edge_index_cat, dim=1)

        # Print shapes for debugging
#         print(f"x shape: {x.shape}")
#         print(f"edge_index shape: {edge_index.shape}")

        torch.manual_seed(42)
        out = self.conv1(x, edge_index)
        torch.manual_seed(42)
        out = self.conv2(out, edge_index)
        
        return out
    
class GraphNetwork_GAT_EdgeFeat(nn.Module):
    def __init__(self, num_features, num_relations, hidden_size=64, num_heads=8, dropout=0.5, no_cuda=True):
        super(GraphNetwork_GAT_EdgeFeat, self).__init__()
        self.conv1 = GATConv(num_features, hidden_size, heads=num_heads, dropout=dropout, edge_dim=num_relations)
        self.conv2 = GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, dropout=dropout, concat=False, edge_dim=num_relations)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=0)

        # Correctly batch edge_index and edge_attr (if edge_attr is used)
        edge_index_cat = []
        edge_attr_cat = []
        batch_offset = 0
        for e_i, e_a in zip(edge_index, edge_attr):
            edge_index_cat.append(e_i + batch_offset)
            edge_attr_cat.append(e_a)
            batch_offset += e_i.max().item() + 1

        edge_index = torch.cat(edge_index_cat, dim=1)
        edge_attr = torch.cat(edge_attr_cat, dim=0)

        # Print shapes for debugging
#         print(f"x shape: {x.shape}")
#         print(f"edge_index shape: {edge_index.shape}")
#         print(f"edge_attr shape: {edge_attr.shape}")
        
        torch.manual_seed(42)
        out = self.conv1(x, edge_index, edge_attr=edge_attr)
        torch.manual_seed(42)
        out = self.conv2(out, edge_index, edge_attr=edge_attr)
        
        return out

class GraphNetwork_GATv2(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_heads=8, dropout=0.5, no_cuda=False):
        super(GraphNetwork_GATv2, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_size, heads=num_heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_size * num_heads, hidden_size, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=False):
        # First convolutional layer
        out = self.conv1(x, edge_index, edge_attr=edge_attr)
        # Second convolutional layer
        if return_attention_weights:
            out, (edge_index, attention_weights) = self.conv2(out, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            return out, (edge_index, attention_weights)
        else:
            out = self.conv2(out, edge_index, edge_attr=edge_attr)
        return out
    
class GraphNetwork_GATv2_EdgeFeat(nn.Module):
    def __init__(self, num_features, num_relations, hidden_size=64, num_heads=8, dropout=0.5, no_cuda=False):
        super(GraphNetwork_GATv2_EdgeFeat, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_size, heads=num_heads, dropout=dropout, edge_dim=num_relations)
        self.conv2 = GATv2Conv(hidden_size * num_heads, hidden_size, heads=num_heads, dropout=dropout, concat=False, edge_dim=num_relations)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=0)

        # Add self-loops to edge_index and adjust edge_attr accordingly
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0.5, num_nodes=x.size(0))

        # Print shapes for debugging
#         print(f"x shape: {x.shape}")
#         print(f"edge_index shape: {edge_index.shape}")
#         print(f"edge_attr shape: {edge_attr.shape if edge_attr is not None else 'None'}")
        
        torch.manual_seed(42)
        out = self.conv1(x, edge_index, edge_attr=edge_attr)
        torch.manual_seed(42)
        out = self.conv2(out, edge_index, edge_attr=edge_attr)
#         dont return the 2nd obj
        return out
    
class GraphNetwork_RGAT(nn.Module):
    def __init__(self, num_features, num_relations, hidden_size=64, num_heads=8, dropout=0.5, edge_dim=1, no_cuda=False):
        super(GraphNetwork_RGAT, self).__init__()
        self.conv1 = RGATConv(num_features, hidden_size, num_relations, heads=num_heads, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = RGATConv(hidden_size * num_heads, hidden_size, num_relations, heads=num_heads, edge_dim=edge_dim, concat=False, dropout=dropout)
        self.no_cuda = no_cuda
        self.num_relations = num_relations

    def forward(self, x, edge_index, edge_type=None, edge_attr=None):
        # Add self-loops to edge_index and adjust edge_attr accordingly
        num_nodes = x.size(0)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0.5, num_nodes=num_nodes)
        
        self_loop_edge_type = (edge_type.max().item() + 1) % self.num_relations
        self_loop_edge_types = torch.full((num_nodes,), self_loop_edge_type, dtype=torch.long)
        edge_type = torch.cat([edge_type, self_loop_edge_types], dim=0)


        torch.manual_seed(42)
        out = self.conv1(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        torch.manual_seed(42)
        out = self.conv2(out, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        
        return out
    
class DialogueGCN_MELDModel(nn.Module):
    def __init__(self, D_m, D_g, D_p, D_e, D_h, D_a, \
                 graph_hidden_size, n_speakers, max_seq_len, window_past, window_future,
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5,
                 nodal_attention=True, avec=False, no_cuda=False):

        super(DialogueGCN_MELDModel, self).__init__()
        self.avec = avec
        self.no_cuda = no_cuda

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net = GraphNetwork(2 * D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout, self.no_cuda)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, features, edge_index, edge_type, edge_index_lengths, umask):
        log_prob = self.graph_net(features, edge_index, edge_type, edge_index_lengths, umask, self.nodal_attention, self.avec)
        return log_prob, edge_index, edge_type, edge_index_lengths

class DialogueGCNModel(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len, window_past, window_future,
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5, nodal_attention=True, avec=False, no_cuda=False):
        
        super(DialogueGCNModel, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda

        # The base model is the sequential context encoder.
        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
            self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)

        elif self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net = GraphNetwork(2*D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout, self.no_cuda)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping


    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        if self.base_model == "DialogRNN":

            if self.avec:
                emotions, _ = self.dialog_rnn_f(U, qmask)

            else:
                emotions_f, alpha_f = self.dialog_rnn_f(U, qmask) # seq_len, batch, D_e
                rev_U = self._reverse_seq(U, umask)
                rev_qmask = self._reverse_seq(qmask, umask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, umask)
                emotions = torch.cat([emotions_f,emotions_b],dim=-1)

        elif self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)

        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)

        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.att_model, self.no_cuda)
        log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask, self.nodal_attention, self.avec)

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths


class CNNFeatureExtractor(nn.Module):
    """
    Module from DialogueRNN
    """
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False

    def forward(self, x, umask):
        num_utt, batch, num_words = x.size()

        x = x.type(LongTensor)  # (num_utt, batch, num_words)
        x = x.view(-1, num_words)  # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x)  # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300)
        emb = emb.transpose(-2,
                            -1).contiguous()  # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words)

        convoluted = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated)))  # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(num_utt, batch, -1)  # (num_utt * batch, 100) -> (num_utt, batch, 100)
        mask = umask.unsqueeze(-1).type(FloatTensor)  # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1)  # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim)  # (num_utt, batch, 1) -> (num_utt, batch, 100)
        features = (features * mask)  # (num_utt, batch, 100) -> (num_utt, batch, 100)

        return features

def getDataLoaderAndLabels(file_path, ranges):
    with open(file_path[0], 'rb') as file:
         all_umask, \
         all_seq_lengths,\
         all_features, \
         all_edge_index, \
         all_edge_norm, \
         all_edge_type, \
         all_edge_index_lengths = pickle.load(file)

    with open(file_path[1], 'rb') as file:
        labels = pickle.load(file)

    dataset = GraphContextDataset(ranges, labels,
                                       all_features, all_edge_index,
                                       all_edge_type,
                                       all_edge_index_lengths,
                                       all_umask, all_seq_lengths)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    return dataLoader, labels
    
DATASET_PATH = "dataset_original"
# DATASET_PATH = "dataset_drop_noise"
# DATASET_PATH = "dataset_smote"    