import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda

import time
from numbers import Number


class Encoder(nn.Module):
    def __init__(self, K=16, n=2):
        super(Encoder, self).__init__()
        self.K = K
        self.encode = nn.Sequential(
            nn.Linear(784, 512//n),
            nn.ReLU(True),
            nn.Linear(512//n, 128//n),
            nn.ReLU(True),
            nn.Linear(128//n, 2*self.K))

    def forward(self, x):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        statistics = self.encode(x)
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5, beta=1)
        emb = self.reparametrize_n(mu, std)

        return (mu, std), emb

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std



class SNIBencoder(nn.Module):
    def __init__(self, K=16, n=2, log_std=1., log_std_trainable=True):
        super(SNIBencoder, self).__init__()
        self.K = K
        self.n = n
        self.encode = nn.Sequential(
            nn.Linear(784, 1024//n),
            nn.ReLU(True),
            nn.Linear(1024//n, 1024//n),
            nn.ReLU(True),
            nn.Linear(1024//n, self.K))
        if log_std_trainable:
            self.register_parameter('log_std', torch.nn.Parameter(torch.FloatTensor([log_std]), requires_grad=True))
        else:
            self.register_buffer('log_std', torch.FloatTensor([log_std]))

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        mu = self.encode(x)
        #std = F.softplus(statistics[:,self.K:]-5,beta=1)
        std = self.log_std.exp().expand(*mu.size())
        emb = self.reparametrize_n(mu, std, num_sample)

        return (mu, std), emb

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


class Decoder(nn.Module):
    def __init__(self, K=16):
        super(Decoder, self).__init__()
        self.K = K
        self.decode = nn.Sequential(
                nn.Linear(self.K, 10))

    def forward(self, x, num_sample=1):
        logit = self.decode(x)
        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return logit



class Reconstractor(nn.Module):
    def __init__(self, K=16):
        super(Reconstractor, self).__init__()
        self.K = K
        self.decode = nn.Sequential(
                nn.Linear(10, self.K))

    def forward(self, x):
        logit = self.decode(x)

        return logit


class Disc(nn.Module):
    def __init__(self, K=16, n=2):
        super(Disc, self).__init__()
        self.K = K
        self.encode = nn.Sequential(
            nn.Linear(K*2, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 1),
            nn.Sigmoid())

    def forward(self, x):
        if x.dim() > 2 : x = x.view(x.size(0),-1)
        out = self.encode(x)

        return out


class VIBencoder(nn.Module):

    def __init__(self, K=16):
        super(VIBencoder, self).__init__()
        self.K = K

        self.encode = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 2*self.K))


    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        statistics = self.encode(x)
        mu = statistics[:,:self.K]
        #std = F.softplus(statistics[:,self.K:]-5,beta=1)
        std = statistics[:,self.K:].exp()
        emb = self.reparametrize_n(mu, std, num_sample)



        return (mu, std), emb

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


class NIBencoder(nn.Module):
    def __init__(self, K=16, log_std=1., log_std_trainable=True):
        super(NIBencoder, self).__init__()
        self.K = K
        self.encode = nn.Sequential(
                nn.Linear(784, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, self.K))
        if log_std_trainable:
            self.register_parameter('log_std', torch.nn.Parameter(torch.FloatTensor([log_std]), requires_grad=True))
        else:
            self.register_buffer('log_std', torch.FloatTensor([log_std]))

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        mu = self.encode(x)
        #std = F.softplus(statistics[:,self.K:]-5,beta=1)
        std = self.log_std.exp().expand(*mu.size())
        emb = self.reparametrize_n(mu, std, num_sample)

        return (mu, std), emb

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()