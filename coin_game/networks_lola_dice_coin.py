import torch
import torch.nn as nn
import torch.nn.functional as F

def get_ndim(n, ks, s):
    return int((n - ks)/s + 1)

class LSTM_policy(nn.Module):
    def __init__(self, num_channels, grid_size, hidden_dim, action_dim):
        super(LSTM_policy, self).__init__()
        out, ks, s = 20, 3, 1
        self.conv0 = nn.Conv2d(in_channels = num_channels, out_channels = out, kernel_size = ks, stride = s)
        self.bn0 = nn.BatchNorm2d(num_features = out)
        self.conv1 = nn.Conv2d(in_channels = out, out_channels = out, kernel_size = ks)
        self.bn1 = nn.BatchNorm2d(num_features = out)
        n = get_ndim(get_ndim(grid_size, ks, s), ks, s)
        self.lstm = nn.LSTMCell(input_size = out * n * n, hidden_size = hidden_dim)
        self.fc0 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, hx, cx):
        if len(x.size()) != 4:
            raise Exception('Input tensor dimension not equal to 4!')
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = self.softmax(self.fc0(hx))
        return x, hx, cx

class critic(nn.Module):
    def __init__(self, num_channels, grid_size):
        super(critic, self).__init__()
        out, ks, s = 20, 3, 1
        self.conv0 = nn.Conv2d(in_channels = num_channels, out_channels = out, kernel_size = ks, stride = s)
        self.bn0 = nn.BatchNorm2d(num_features = out)
        self.conv1 = nn.Conv2d(in_channels = out, out_channels = out, kernel_size = ks)
        self.bn1 = nn.BatchNorm2d(num_features = out)
        n = get_ndim(get_ndim(grid_size, ks, s), ks, s)
        #self.lstm = nn.LSTMCell(input_size = out * n * n, hidden_size = hidden_dim)
        self.fc0 = nn.Linear(out * n * n, 1)

    def forward(self, x):
        if len(x.size()) != 4:
            raise Exception('Input tensor dimension not equal to 4!')
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        #hx, cx = self.lstm(x, (hx, cx))
        x = self.fc0(x)
        return x