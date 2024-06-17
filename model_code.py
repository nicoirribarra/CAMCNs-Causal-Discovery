import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable

class Chomp1d(nn.Module):
    """PyTorch does not offer native support for causal convolutions, so it is implemented 
    (with some inefficiency) by simply using a standard convolution with zero padding on both sides, 
    and chopping off the end of the sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class FirstBlock(nn.Module):
    def __init__(self, target, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(FirstBlock, self).__init__()

        self.target = target
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.multihead_attn = nn.MultiheadAttention(n_inputs, n_inputs, batch_first = False)
        self.linear = nn.Linear(n_inputs,n_outputs)

        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1)
        self.linear.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        out = out.transpose(1,2)
        out = self.multihead_attn(out,out,out, need_weights = False)
        out = self.linear(out[0])
        return self.relu(out.transpose(1,2))

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.multihead_attn = nn.MultiheadAttention(n_inputs, n_inputs, batch_first = False)
        self.linear = nn.Linear(n_inputs,n_outputs)

        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1)
        self.linear.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        out = out.transpose(1,2)
        out = self.multihead_attn(out,out,out, need_weights = False)
        out = self.linear(out[0])
        return self.relu(out.transpose(1,2))

class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.multihead_attn = nn.MultiheadAttention(n_inputs, n_inputs, batch_first = False)
        self.linear = nn.Linear(n_inputs,n_outputs)
        self.linear_out = nn.Linear(n_inputs,n_inputs)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1)
        self.linear_out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        out = out.transpose(1,2)
        out = self.multihead_attn(out,out,out, need_weights = False)
        out = self.linear(out[0])
        return self.linear_out(out+x.transpose(1,2)).transpose(1,2)

class DepthwiseNet(nn.Module):
    def __init__(self, target, num_inputs, num_levels, kernel_size=2, dilation_c=2):
        super(DepthwiseNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_inputs
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            if l==0:
                layers += [FirstBlock(target, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            elif l==num_levels-1:
                layers+=[LastBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class ValBlock(nn.Module):
    def __init__(self, target, n_inputs, n_outputs):
        super(ValBlock, self).__init__()

        self.target = target
        self.multihead_attn = nn.MultiheadAttention(n_inputs, n_inputs, batch_first = True)
        self.linear = nn.Linear(n_inputs,n_outputs)
        self.linear_h = nn.Linear(n_inputs,n_inputs)
        self.relu_h = nn.PReLU(n_inputs)
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = x
        out = self.linear_h(out)
        out = self.linear(out)
        return self.relu(out)

class ADDSTCN(nn.Module):
    def __init__(self, target, input_size, num_levels, kernel_size, cuda, dilation_c):
        super(ADDSTCN, self).__init__()

        self.target=target
        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)

        self._attention = torch.ones(input_size,1)
        self._attention = Variable(self._attention, requires_grad=False)

        self.fs_attention = torch.nn.Parameter(self._attention.data)

        if cuda:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()

    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)

    def forward(self, x):
        y1=self.dwn(x*F.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1)
        return y1.transpose(1,2)

class TCFMODEL():
    def __init__(self, optimizername="Adam", cuda=True, seed=1234):
        self.optimizername = optimizername
        self.cuda = cuda
        self.seed = seed
    
    def train(self, epoch, traindata, traintarget, modelname, optimizer,log_interval,epochs):
        """Trains model by performing one epoch and returns attention scores and loss."""

        modelname.train()
        x, y = traindata[0:1], traintarget[0:1]

        optimizer.zero_grad()
        epochpercentage = (epoch/float(epochs))*100
        output = modelname(x)

        attentionscores = modelname.fs_attention

        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
            print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))

        return attentionscores.data, loss

    def train_val(self, epoch, traindata, traintarget, modelname, optimizer):
        """Trains model by performing one epoch and returns attention scores and loss."""

        modelname.train()
        x, y = traindata, traintarget

        optimizer.zero_grad()
        output = modelname(x)

        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        return loss
    
    def preparedata_val(self, file, target, pre, lag):
        """Reads data from csv file and transforms it to two PyTorch tensors: dataset x 
        and target time series y that has to be predicted."""
        df_data = file
        df_y = df_data.copy(deep=True)[df_data.copy().columns[target]]
        df_x = pd.DataFrame()
        df_x = df_data.copy(deep=True)[df_data.copy().columns[pre]]
        df_aux = df_x.copy()
        for i in range(1,lag+1):
          y = df_aux.shift(-i)
          df_x = pd.concat([df_x,y], axis=1)
        df_x = df_x.fillna(0.)
        data_x = df_x.values.astype('float32')
        data_y = df_y.values.astype('float32')
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)
        x, y = Variable(data_x), Variable(data_y)
        return x, y

    def VAL(self, file, targetidx, v, totaldelay_1, lr, eps):
        torch.manual_seed(self.seed)
        # Direct
        X_train, Y_train = self.preparedata_val(file, targetidx, v, totaldelay_1)
        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()
        input_channels = X_train.size()[2]
        model_1 = ValBlock(targetidx, input_channels, 1)
        if self.cuda:
            model_1.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
        optimizer = getattr(optim, self.optimizername)(model_1.parameters(), lr=lr)
        for ep in range(1, eps+1):
            loss_dir = self.train_val(ep, X_train, Y_train, model_1, optimizer)
        loss_dir = loss_dir.cpu().data.item()
        # Inverse
        X_train, Y_train = self.preparedata_val(file, v, targetidx, totaldelay_1)
        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()

        input_channels = X_train.size()[2]

        model_2 = ValBlock(v, input_channels, 1)
        if self.cuda:
            model_2.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

        optimizer = getattr(optim, self.optimizername)(model_2.parameters(), lr=lr)
        for ep in range(1, eps+1):
            loss_inv = self.train_val(ep, X_train, Y_train, model_2, optimizer)
        loss_inv = loss_inv.cpu().data.item()
        return loss_dir, loss_inv