import torch
import torch.autograd as autograd

def zero_grad(params):
	for p in params:
		if p.grad is not None:
			p.grad.detach()
			p.grad.zero_()

def conjugate_gradient(grad_x_lp, grad_y_lp, grad_x_mh, grad_y_mh, x_params, y_params, b, x = None, nsteps = 10, residual_tol = 1e-18, \
									lr = 0.01, device = torch.device('cpu')):
	
	if x is None:
		x = torch.zeros((b.shape[0], 1), device = device)
	r = b.clone().detach()
	p = r.clone().detach()
	rdotr = torch.dot(r.view(-1), r.view(-1))
	residual_tol *= rdotr

	for i in range(nsteps):
		h1 = autograd.grad(grad_x_mh, y_params, grad_outputs = p, retain_graph = True)
		h1 = torch.cat([g.contiguous().view(-1,1) for g in h1])
		h2 = autograd.grad(grad_y_mh, x_params, grad_outputs = h1, retain_graph = True)
		h2 = torch.cat([g.contiguous().view(-1,1) for g in h2])
		Ap = p - lr * lr * h2
		alpha = rdotr/torch.dot(p.view(-1), Ap.view(-1))
		x.data.add_(alpha * p)
		r.data.add_(-alpha * Ap)
		new_rdotr = torch.dot(r.view(-1), r.view(-1))
		beta = new_rdotr/rdotr
		p = r + beta * p
		rdotr = new_rdotr
		if rdotr < residual_tol:
			break

	return x, i+1

def general_conjugate_gradient(grad_x_lp, grad_y_lp, grad_x_mh, grad_y_mh, x_params, y_params, b, x = None, nsteps = 10, \
										residual_tol = 1e-16, lr_x = 0.01, lr_y = 0.01, device = torch.device('cpu')):
	
	if x is None:
		x = torch.zeros((b.shape[0], 1), device = device)
	lr_x = lr_x.sqrt()
	r = b.clone().detach()
	p = r.clone().detach()
	rdotr = torch.dot(r.view(-1), r.view(-1))
	residual_tol *= rdotr

	for i in range(nsteps):
		h1 = autograd.grad(grad_x_mh, y_params, grad_outputs = lr_x * p, retain_graph = True)
		h1 = torch.cat([g.contiguous().view(-1,1) for g in h1]).mul_(lr_y)
		h2 = autograd.grad(grad_y_mh, x_params, grad_outputs = h1, retain_graph = True)
		h2 = torch.cat([g.contiguous().view(-1,1) for g in h2]).mul_(lr_x)
		Ap = p - h2
		alpha = rdotr/torch.dot(p.view(-1), Ap.view(-1))
		x.data.add_(alpha * p)
		r.data.add_(-alpha * Ap)
		new_rdotr = torch.dot(r.view(-1), r.view(-1))
		beta = new_rdotr/rdotr
		p = r + beta * p
		rdotr = new_rdotr
		if rdotr < residual_tol:
			break

	return x, i+1