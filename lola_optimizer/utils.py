import torch
import torch.autograd as autograd

def zero_grad(params):
	for p in params:
		if p.grad is not None:
			p.grad.detach()
			p.grad.zero_()