import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class AttLstm(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda=True):
        super(self.__class__, self).__init__()
        self.lstm_hidden_layer = hidden_size
        self.use_cuda = use_cuda
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        if use_cuda:
            self.lstm_cell = self.lstm_cell.cuda()

    def forward(self, _input):
        # initiate hidden states
        batch_size = _input.size(0)
        if self.use_cuda:
            h_state = torch.zeros((batch_size, self.lstm_hidden_layer)).cuda()
            c_state = torch.zeros((batch_size, self.lstm_hidden_layer)).cuda()
            h_states = torch.Tensor().cuda()
        else:
            h_state = torch.zeros((batch_size, self.lstm_hidden_layer))
            c_state = torch.zeros((batch_size, self.lstm_hidden_layer))
            h_states = torch.Tensor()

        # iterate time series
        for j in range(_input.size(1)):
            h_state, c_state = self.lstm_cell(_input[:, j, :], (h_state, c_state))
            h_states = torch.cat((h_states, h_state.view(1, -1, self.lstm_hidden_layer)))
        h_states = h_states.transpose(0, 1)

        # calculate attention value(context_vector)
        att_score = torch.matmul(h_states, h_state.view(-1, self.lstm_hidden_layer, 1))
        att_dist = att_score / torch.sum(att_score, dim=1).view(-1, 1, 1)
        context_vector = torch.matmul(att_dist.view(batch_size, 1, -1), h_states)

        return context_vector


class Dtml(nn.Module):
    """
    n_stock_input_vars: stock_input에 포함된 변수 개수(stock * window * variable 의 마지막 차원)
    n_macro_input_vars: macro_input에 포함된 변수 개수(window * variable 의 마지막 차원)
    n_stock: stock_input에 포함된 전체 종목 개수(stock * window * variable 의 첫번째 차원)
    n_time: stock_input 내 시퀀스의 길이(stock * window * variable 의 두번째 차원)
    n_heads: multi head attemtion 수행 시 사용할 param. lstm_hidden_layer // n_heads는 반드시 정수여야 한다.
    d_lstm_input: feature transformation layer의 output 차원이자 lstm의 input 차원. None일 경우 n_stock_input_vars 적용
    lstm_hidden_layer: att.lstm의 hidden layer 수(lstm output 차원 수)
    """
    def __init__(self, n_stock_input_vars, n_macro_input_vars, n_stock, n_time, n_heads, d_lstm_input=None, lstm_hidden_layer=64, use_cuda=True):
        super(self.__class__, self).__init__()
        if d_lstm_input is None:
            d_lstm_input = n_stock_input_vars
        self.lstm_hidden_layer = lstm_hidden_layer
        self.use_cuda = use_cuda
        self.att_weight = None

        # Feature Transformation layers
        self.stock_f_tr_layer = nn.Sequential(
            nn.Linear(n_stock_input_vars, d_lstm_input),
            nn.Tanh()
        )
        self.macro_f_tr_layer = nn.Sequential(
            nn.Linear(n_macro_input_vars, d_lstm_input),
            nn.Tanh()
        )
        if use_cuda:
            self.stock_f_tr_layer = self.stock_f_tr_layer.cuda()
            self.macro_f_tr_layer = self.macro_f_tr_layer.cuda()

        # LSTM cell
        self.stock_att_lstm = AttLstm(input_size=d_lstm_input, hidden_size=lstm_hidden_layer, use_cuda=use_cuda)
        self.macro_att_lstm = AttLstm(input_size=d_lstm_input, hidden_size=lstm_hidden_layer, use_cuda=use_cuda)

        # Context Normalization parameters
        if use_cuda:
            self.norm_weight = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer).cuda())
            self.norm_bias = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer).cuda())
        else:
            self.norm_weight = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer))
            self.norm_bias = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer))

        # Macro weight for Multi-Level Contexts
        if use_cuda:
            self.macro_weight = nn.Parameter(torch.randn(1).cuda())
        else:
            self.macro_weight = nn.Parameter(torch.randn(1))

        # Q, K, V weight for Self-Attention input
        self.multi_head_att = nn.MultiheadAttention(lstm_hidden_layer, n_heads, batch_first=True)
        if use_cuda:
            self.multi_head_att = self.multi_head_att.cuda()

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_layer, lstm_hidden_layer * 4),
            nn.ReLU(),
            nn.Linear(lstm_hidden_layer * 4, lstm_hidden_layer)
        )
        if use_cuda:
            self.mlp = self.mlp.cuda()

        # final single layer
        self.final_layer = nn.Linear(lstm_hidden_layer, 2)
        if use_cuda:
            self.final_layer = self.final_layer.cuda()

    def forward(self, stock_input, macro_input):
        """
        stock_input은 (t, s, w, v)차원으로 구성되어 있다.
        macro_input은 (t, w, v)차원으로 구성되어 있다.
         * t: 시간(batch size), s: 주식, w: 시퀀스(window), v: 변수
        """
        # Feature Transformation - (1)
        stock_input = self.stock_f_tr_layer(stock_input)
        macro_input = self.macro_f_tr_layer(macro_input)

        # Attention LSTM - (2)
        if self.use_cuda:
            c_matrix = torch.Tensor().cuda()
        else:
            c_matrix = torch.Tensor()

        # stock
        for i in range(stock_input.size(1)):
            context_vector = self.stock_att_lstm(stock_input[:, i, :, :])
            c_matrix = torch.cat((c_matrix, context_vector.view(1, -1, 1, self.lstm_hidden_layer)))
        c_matrix = c_matrix.view(stock_input.size(0), -1, self.lstm_hidden_layer)
        # macro
        macro_context = self.macro_att_lstm(macro_input)

        # Context Normalization - (3)
        c_matrix = self.norm_weight * ((c_matrix - torch.mean(c_matrix, dim=(1, 2)).view(-1, 1, 1)) / torch.std(c_matrix, dim=(1, 2)).view(-1, 1, 1)) + self.norm_bias

        # Multi-Level Contexts - (4)
        ml_c_matrix = c_matrix + self.macro_weight * macro_context

        # Multi-Head Self-Attention Q, K, V - (6), (7)
        att_value_matrix, self.att_weight = self.multi_head_att(ml_c_matrix, ml_c_matrix, ml_c_matrix)

        # Nonlinear Transformation - (8)
        if self.use_cuda:
            mlp_out_matrix = torch.Tensor().cuda()
        else:
            mlp_out_matrix = torch.Tensor()

        for i in range(att_value_matrix.size(1)):
            mlp_out = self.mlp(ml_c_matrix[:, i, :] + att_value_matrix[:, i, :])
            mlp_out_matrix = torch.cat((mlp_out_matrix, mlp_out.view(1, -1, self.lstm_hidden_layer)))
        mlp_out_matrix = mlp_out_matrix.transpose(0, 1)
        out_matrix = torch.tanh(ml_c_matrix + att_value_matrix + mlp_out_matrix)

        # Final Prediction - (9)
        if self.use_cuda:
            final_out_matrix = torch.Tensor().cuda()
        else:
            final_out_matrix = torch.Tensor()

        for i in range(out_matrix.size(1)):
            f_out = self.final_layer(out_matrix[:, i, :])
            final_out_matrix = torch.cat((final_out_matrix, f_out.view(1, -1, 2)))
        final_out_matrix = final_out_matrix.transpose(0, 1)

        return F.softmax(final_out_matrix, dim=2)
