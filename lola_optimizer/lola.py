import time
import math
import torch
import torch.autograd as autograd
from lola_optimizer.utils import zero_grad

# code is assuming f (for agent1) and g (for agent2) are 2 functions to minimize => set advantages accordingly

class LOLA(object):
    def __init__(self, x_params, y_params, lr, weight_decay = 0, device = torch.device('cpu'), collect_info = True):
        self.x_params = list(x_params)
        self.y_params = list(y_params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.collect_info = collect_info

    def zero_grad(self):
        zero_grad(self.x_params)
        zero_grad(self.y_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_grad_x, self.norm_grad_y, self.norm_grad_hvp_x, self.norm_grad_hvp_y
        else:
            raise ValueError('No update info stored. Set collect_info = True before calling this method')

    def step(self, lp_x, lp_y, mh):
        grad_x_lp_x = autograd.grad(lp_x, self.x_params, retain_graph = True)
        grad_x_lp_x = torch.cat([g.contiguous().view(-1,1) for g in grad_x_lp_x])

        grad_y_lp_y = autograd.grad(lp_y, self.y_params, retain_graph = True)
        grad_y_lp_y = torch.cat([g.contiguous().view(-1,1) for g in grad_y_lp_y])

        grad_y_mh = autograd.grad(mh, self.y_params, create_graph = True, retain_graph = True)
        grad_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_mh])
        grad_x_y_mh = autograd.grad(grad_y_mh, self.x_params, grad_outputs = grad_y_lp_y, retain_graph = True)
        grad_x_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_y_mh])

        grad_x_mh = autograd.grad(mh, self.x_params, create_graph = True, retain_graph = True)
        grad_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_mh])
        grad_y_x_mh = autograd.grad(grad_x_mh, self.y_params, grad_outputs = grad_x_lp_x)
        grad_y_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_x_mh])

        grad_hvp_x = grad_x_lp_x - self.lr * grad_x_y_mh
        grad_hvp_y = grad_y_lp_y - self.lr * grad_y_x_mh
        # delta_x = - self.lr * grad_hvp_x
        # delta_y = - self.lr * grad_hvp_y
		
        index = 0
        for p in self.x_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(- self.lr * grad_hvp_x[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != grad_hvp_x.numel():
            raise ValueError('Size mismatch')
        
        index = 0
        for p in self.y_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(- self.lr * grad_hvp_y[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != grad_hvp_y.numel():
            raise ValueError('Size mismatch')

        if self.collect_info:
            self.norm_grad_x = torch.norm(grad_x_lp_x, p = 2)
            self.norm_grad_y = torch.norm(grad_y_lp_y, p = 2)
            self.norm_grad_hvp_x = torch.norm(grad_hvp_x, p = 2)
            self.norm_grad_hvp_y = torch.norm(grad_hvp_y, p = 2)


class LOLA1(object):
    def __init__(self, x_params, y_params, lr, weight_decay = 0, device = torch.device('cpu'), collect_info = True):
        self.x_params = list(x_params)
        self.y_params = list(y_params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.collect_info = collect_info

    def zero_grad(self):
        zero_grad(self.x_params)
        zero_grad(self.y_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_grad_x, self.norm_grad_x_y_mh, self.norm_grad_y, self.norm_grad_y_x_mh
        else:
            raise ValueError('No update info stored. Set collect_info = True before calling this method')

    def step(self, lp_x_1, lp_x_2, lp_y_1, lp_y_2, mh_1, mh_2):
        grad_x_lp_x_1 = autograd.grad(lp_x_1, self.x_params, retain_graph = True)
        grad_x_lp_x_1 = torch.cat([g.contiguous().view(-1,1) for g in grad_x_lp_x_1])

        #grad_x_lp_x_2 = autograd.grad(lp_x_2, self.x_params, retain_graph = True)
        #grad_x_lp_x_2 = torch.cat([g.contiguous().view(-1,1) for g in grad_x_lp_x_2])

        #grad_y_lp_y_1 = autograd.grad(lp_y_1, self.y_params, retain_graph = True)
        #grad_y_lp_y_1 = torch.cat([g.contiguous().view(-1,1) for g in grad_y_lp_y_1])

        grad_y_lp_y_2 = autograd.grad(lp_y_2, self.y_params, retain_graph = True)
        grad_y_lp_y_2 = torch.cat([g.contiguous().view(-1,1) for g in grad_y_lp_y_2])

        grad_y_mh = autograd.grad(mh_1, self.y_params, create_graph = True, retain_graph = True)
        grad_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_mh])
        grad_x_y_mh = autograd.grad(grad_y_mh, self.x_params, grad_outputs = grad_y_lp_y_2, retain_graph = True)
        grad_x_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_y_mh])

        grad_x_mh = autograd.grad(mh_2, self.x_params, create_graph = True, retain_graph = True)
        grad_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_mh])
        grad_y_x_mh = autograd.grad(grad_x_mh, self.y_params, grad_outputs = grad_x_lp_x_1)
        grad_y_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_x_mh])

        grad_hvp_x = grad_x_lp_x_1 - self.lr * grad_x_y_mh
        grad_hvp_y = grad_y_lp_y_2 - self.lr * grad_y_x_mh
        # delta_x = - self.lr * grad_hvp_x
        # delta_y = - self.lr * grad_hvp_y
		
        index = 0
        for p in self.x_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(- self.lr * grad_hvp_x[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != grad_hvp_x.numel():
            raise ValueError('Size mismatch')
        
        index = 0
        for p in self.y_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(- self.lr * grad_hvp_y[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != grad_hvp_y.numel():
            raise ValueError('Size mismatch')

        if self.collect_info:
            self.norm_grad_x = torch.norm(grad_x_lp_x_1, p = 2)
            self.norm_grad_y = torch.norm(grad_y_lp_y_2, p = 2)
            self.norm_grad_x_y_mh = torch.norm(grad_x_y_mh, p = 2)
            self.norm_grad_y_x_mh = torch.norm(grad_y_x_mh, p = 2)

class LOLA2(object):
    def __init__(self, x_params, y_params, lr, weight_decay = 0, device = torch.device('cpu'), collect_info = False):
        self.x_params = list(x_params)
        self.y_params = list(y_params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.collect_info = collect_info

    def zero_grad(self):
        zero_grad(self.x_params)
        zero_grad(self.y_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_grad_x, self.norm_grad_x_y_mh, self.norm_grad_y, self.norm_grad_y_x_mh
        else:
            raise ValueError('No update info stored. Set collect_info = True before calling this method')

    def step(self, lp_x_1, lp_x_2, lp_y_1, lp_y_2, mh_1, mh_2):
        grad_x_lp_x_1 = autograd.grad(lp_x_1, self.x_params, retain_graph = True)
        grad_x_lp_x_1 = torch.cat([g.contiguous().view(-1,1) for g in grad_x_lp_x_1])

        grad_x_lp_x_2 = autograd.grad(lp_x_2, self.x_params, retain_graph = True)
        grad_x_lp_x_2 = torch.cat([g.contiguous().view(-1,1) for g in grad_x_lp_x_2])

        grad_y_lp_y_1 = autograd.grad(lp_y_1, self.y_params, retain_graph = True)
        grad_y_lp_y_1 = torch.cat([g.contiguous().view(-1,1) for g in grad_y_lp_y_1])

        grad_y_lp_y_2 = autograd.grad(lp_y_2, self.y_params, retain_graph = True)
        grad_y_lp_y_2 = torch.cat([g.contiguous().view(-1,1) for g in grad_y_lp_y_2])

        grad_y_mh = autograd.grad(mh_2, self.y_params, create_graph = True, retain_graph = True)
        grad_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_mh])
        grad_x_y_mh = autograd.grad(grad_y_mh, self.x_params, grad_outputs = grad_y_lp_y_1, retain_graph = True)
        grad_x_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_y_mh])

        grad_x_mh = autograd.grad(mh_1, self.x_params, create_graph = True, retain_graph = True)
        grad_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_mh])
        grad_y_x_mh = autograd.grad(grad_x_mh, self.y_params, grad_outputs = grad_x_lp_x_2)
        grad_y_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_x_mh])

        grad_hvp_x = grad_x_lp_x_1 + self.lr * grad_x_y_mh
        grad_hvp_y = grad_y_lp_y_2 + self.lr * grad_y_x_mh
		
        index = 0
        for p in self.x_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(self.lr * grad_hvp_x[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != grad_hvp_x.numel():
            raise ValueError('Size mismatch')
        
        index = 0
        for p in self.y_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(self.lr * grad_hvp_y[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != grad_hvp_y.numel():
            raise ValueError('Size mismatch')

        if self.collect_info:
            self.norm_grad_x = torch.norm(grad_x_lp_x_1, p = 2)
            self.norm_grad_y = torch.norm(grad_y_lp_y_2, p = 2)
            self.norm_grad_x_y_mh = torch.norm(grad_x_y_mh, p = 2)
            self.norm_grad_y_x_mh = torch.norm(grad_y_x_mh, p = 2)
