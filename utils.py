import torch
from torch import nn
from torch.autograd import Variable
import copy
import numpy as np
import torch.nn.functional as F
import math
import yaml
import os
from typing import List
from ast import literal_eval

Cuda = False

class MIEvaluator(object):
    """
    Mutual Information estimator.
    """
    def __init__(self, encoder, decoder, device):
        """
        - func_encode:
            kwargs: x (batch, ...)
            output: {'emb': (batch, nz), 'param': (mu: (batch, nz), std: (batch, nz))}
        - func_decode:
            kwargs: z (batch, nz)
            output: (batch, num_classes)
        """
        # Modules
        self._Enc, self._Dec = encoder, decoder
        # Config
        self._device = device

    @torch.no_grad()
    def eval_mi_x_z_monte_carlo(self, dataloader):
        torch.cuda.empty_cache()
        # Get dataloaders
        x_dataloader = dataloader
        z_dataloader = copy.deepcopy(dataloader)
        ################################################################################################################
        # Eval H(X|Z)
        ################################################################################################################
        ent_x_z = []
        for batch_z_index, batch_z_data in enumerate(z_dataloader):
            # Get z. (batch, nz)
            (_, _), batch_z = self._Enc(batch_z_data[0].to(self._device))
            ############################################################################################################
            # Calculate H(X|batch_z)
            ############################################################################################################
            # 1. Get log p(batch_z|x). (batch, total_num_x)
            log_p_batch_z_x = []
            for batch_x_index, batch_x_data in enumerate(x_dataloader):
                # (1) Get params (mu, std). (batch, nz)
                (mu, std), _ = self._Enc(batch_x_data[0].to(self._device))
                # (2) Get log p(batch_z|batch_x). (batch, batch)
                log_p_batch_z_batch_x = gaussian_log_density_marginal(batch_z, (mu, std), mesh=True).sum(dim=2)
                # Accumulate
                log_p_batch_z_x.append(log_p_batch_z_batch_x)
            log_p_batch_z_x = torch.cat(log_p_batch_z_x, dim=1)
            # 2. Normalize to get log p(x|batch_z). (batch, total_num_x)
            log_p_x_batch_z = log_p_batch_z_x - torch.logsumexp(log_p_batch_z_x, dim=1, keepdim=True)
            # 3. Get H(X|batch_z). (batch, )
            ent_x_batch_z = (-torch.exp(log_p_x_batch_z) * log_p_x_batch_z).sum(dim=1)
            # Accumulate
            ent_x_z.append(ent_x_batch_z)
        ent_x_z = torch.cat(ent_x_z, dim=0).mean()
        ################################################################################################################
        # Eval H(X)
        ################################################################################################################
        ent_x = math.log(len(x_dataloader.dataset)+1e-8)
        ################################################################################################################
        # Eval I(X;Z) = H(X) - H(X|Z)
        ################################################################################################################
        ret = ent_x - ent_x_z
        # Return
        return ret

    @torch.no_grad()
    def eval_mi_y_z_variational_lb(self, dataloader, num_class):
        torch.cuda.empty_cache()
        ####################################################################################################################
        # Eval H(Y|Z) upper bound.
        ####################################################################################################################
        ent_y_z = []
        for batch_index, batch_data in enumerate(dataloader):
            # (1) Get image & label, embedding (batch, nz)
            batch_x, label = map(lambda _x: _x.to(self._device), batch_data)
            (_, _), batch_z = self._Enc(batch_x)
            # (2) Get H(Y|batch_z). (batch, )
            prob = torch.softmax(self._Dec(batch_z), dim=1)
            ent_y_batch_z = (-prob * torch.log(prob + 1e-10)).sum(dim=1)
            # Accumulate to result
            ent_y_z.append(ent_y_batch_z)
        ent_y_z = torch.cat(ent_y_z, dim=0).mean()
        ####################################################################################################################
        # Get H(Y)
        ####################################################################################################################
        ent_y = math.log(num_class)
        ####################################################################################################################
        # Eval I(Y;Z) = H(Y) - H(Y|Z)
        ####################################################################################################################
        ret = ent_y - ent_y_z
        # Return
        return ret


def gaussian_log_density_marginal(sample, params, mesh=False):
    """
    Estimate Gaussian log densities:
        For not mesh:
            log p(sample_i|params_i), i in [batch]
        Otherwise:
            log p(sample_i|params_j), i in [num_samples], j in [num_params]
    :param sample: (num_samples, dims)
    :param params: mu, std. Each is (num_params, dims)
    :param mesh:
    :return:
        For not mesh: (num_sample, dims)
        Otherwise: (num_sample, num_params, dims)
    """
    # Get data
    mu, std = params
    # Mesh
    if mesh:
        sample = sample.unsqueeze(1)
        mu, std = mu.unsqueeze(0), std.unsqueeze(0)
    # Calculate
    # (1) log(2*pi)
    constant = math.log(2 * math.pi)
    # (2) 2 * log std_i
    log_det_std = 2 * torch.log(std+1e-8)
    # (3) (x-mu)_i^2 / std_i^2
    dev_inv_std_dev = ((sample - mu) / std) ** 2
    # Get result
    log_prob_marginal = - 0.5 * (constant + log_det_std + dev_inv_std_dev)
    # Return
    return log_prob_marginal




def gaussian_kl_div(params1, params2='none', reduction='sum', average_batch=False):
    """
        0.5 * {
            sum_j [ log(var2)_j - log(var1)_j ]
            + sum_j [ (mu1 - mu2)^2_j / var2_j ]
            + sum_j (var1_j / var2_j)
            - K
        }
    :return:
    """
    assert reduction in ['sum', 'mean']
    # 1. Get params
    # (1) First
    mu1, std1 = params1
    # (2) Second
    if params2 == 'none':
        mu2 = torch.zeros(*mu1.size()).to(mu1.device)
        std2 = torch.ones(*std1.size()).to(std1.device)
    else:
        mu2, std2 = params2
    # 2. Calculate result
    result = 0.5 * (
        2 * (torch.log(std2) - torch.log(std1))
        + ((mu1 - mu2) / std2) ** 2
        + (std1 / std2) ** 2
        - 1)
    if reduction == 'sum':
        result = result.sum(dim=-1)
    else:
        result = result.mean(dim=-1)
    if average_batch:
        result = result.mean()
    # Return
    return result



def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor


class Weight_EMA_Update(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay)*state_dict[key] + (1-self.decay)*new_state_dict[key]
            #state_dict[key] = (1-self.decay)*state_dict[key] + (self.decay)*new_state_dict[key]

        self.model.load_state_dict(state_dict)




class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

class GaussianLayer(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + self.std * torch.randn_like(x)#.to(self.device)


class StatisticsNetwork(nn.Module):
    def __init__(self, x_dim, z_dim, device):
        super().__init__()
        self.layers = nn.Sequential(
            GaussianLayer(std=0.3, device=device),
            nn.Linear(x_dim + z_dim, 512),
            nn.ELU(),
            GaussianLayer(std=0.5, device=device),
            nn.Linear(512, 512),
            nn.ELU(),
            GaussianLayer(std=0.5, device=device),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def batch(self, x, y, batch_size=1, shuffle=True):
        assert len(x) == len(
            y), "Input and target data must contain same number of elements"
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        n = len(x)

        if shuffle:
            rand_perm = torch.randperm(n)
            x = x[rand_perm]
            y = y[rand_perm]

        batches = []
        for i in range(n // batch_size):
            x_b = x[i * batch_size: (i + 1) * batch_size]
            y_b = y[i * batch_size: (i + 1) * batch_size]

            batches.append((x_b, y_b))
        return batches

    def optimize(self, X, Y, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in self.batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
                #print(f"It {iter} - MI: {mu_mi / batch_size}")

        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")
        return final_mi

class T(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layers(x, z)


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + 1e-6) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema