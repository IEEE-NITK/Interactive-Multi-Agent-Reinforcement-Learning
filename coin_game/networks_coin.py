import torch
import torch.nn as nn
import torch.nn.functional as f

class MLP_policy(nn.Module):
    def __init__(self, matrix_size, num_channels, action_dim, is_conv, activation_function):
        super(MLP_policy, self).__init__()
        self.is_conv = is_conv
        if self.is_conv:
            out, ks, s = 8, 3, 1
            self.conv0 = nn.Conv2d(in_channels = num_channels, out_channels = out, kernel_size = ks, stride = s)
            n = int((matrix_size - ks)/s + 1)
            self.l1 = nn.Linear(out * n * n, 64)
        else:
            self.l1 = nn.Linear(num_channels * matrix_size * matrix_size, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)
        self.softmax = nn.Softmax(dim = -1)
        if activation_function == 'tanh':
            self.act = nn.Tanh()#torch.tanh
        elif activation_function == 'relu':
            self.act = nn.ReLU()#f.relu
        else:
            raise Exception('Unknown activation function')

    def forward(self, obs):
        # obs should be a 4 tuple
        if len(obs.size()) != 4:
            raise Exception('Input tensor should have dim = 4')
        x = obs
        if self.is_conv:
            x = self.act(self.conv0(x))
        else:
            pass
        x = x.view(x.size(0), -1)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.softmax(self.l3(x))
        return x

class LSTM_policy(nn.Module):
    def __init__(self, matrix_size, num_channels, action_dim, hidden_dim, is_conv, activation_function):
        super(LSTM_policy, self).__init__()
        self.is_conv = is_conv
        if self.is_conv:
            out, ks, s = 8, 3, 1
            self.conv0 = nn.Conv2d(in_channels = num_channels, out_channels = out, kernel_size = ks, stride = s)
            n = int((matrix_size - ks)/s + 1)
            lstm_input = out * n * n
        else:
            self.l0 = nn.Linear(num_channels * matrix_size * matrix_size, 64)
            lstm_input = 64
        
        self.lstm = nn.LSTMCell(lstm_input, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)
        #self.l2 = nn.Linear(64, 32)
        #self.l3 = nn.Linear(32, action_dim)
        self.softmax = nn.Softmax(dim = -1)
        if activation_function == 'tanh':
            self.act = nn.Tanh()#torch.tanh
        elif activation_function == 'relu':
            self.act = nn.ReLU()#f.relu
        else:
            raise Exception('Unknown activation function')

    def forward(self, obs, hx, cx):
        # obs should be a 4 tuple
        if len(obs.size()) != 4:
            raise Exception('Input tensor should have dim = 4')
        x = obs
        #hx = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(device)
        #cx = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(device)
        if self.is_conv:
            x = self.act(self.conv0(x))
            x = x.view(x.size(0), -1)
        else:
            x = self.act(self.l0(x.view(x.size(0), -1)))
        hx, cx = self.lstm(x, (hx, cx))
        x = self.softmax(self.fc(hx))
        return x, hx, cx

class MLP_value(nn.Module):
    def __init__(self, matrix_size, num_channels, is_conv, activation_function):
        super(MLP_value, self).__init__()
        self.is_conv = is_conv
        if self.is_conv:
            out, ks, s = 8, 3, 1
            self.conv0 = nn.Conv2d(in_channels = num_channels, out_channels = out, kernel_size = ks, stride = s)
            n = int((matrix_size - ks)/s + 1)
            self.l1 = nn.Linear(out * n * n, 64)
        else:
            self.l1 = nn.Linear(num_channels * matrix_size * matrix_size, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)
        if activation_function == 'tanh':
            self.act = nn.Tanh()#torch.tanh
        elif activation_function == 'relu':
            self.act = nn.ReLU()#f.relu
        else:
            raise Exception('Unknown activation function')

    def forward(self, obs):
        # obs should be a 4 tuple
        if len(obs.size()) != 4:
            raise Exception('Input tensor should have dim = 4')
        x = obs
        if self.is_conv:
            x = self.act(self.conv0(x))
        else:
            pass
        x = x.view(x.size(0), -1)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.l3(x)
        return x
