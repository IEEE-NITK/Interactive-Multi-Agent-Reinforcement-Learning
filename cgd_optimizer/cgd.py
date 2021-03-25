import time
import math
import torch
import torch.autograd as autograd
from cgd_optimizer.utils import zero_grad, conjugate_gradient, general_conjugate_gradient

# code is assuming f and g are 2 functions to minimize => set advantages accordingly

class CGD(object):
	def __init__(self, x_params, y_params, lr=0.01, weight_decay=0, device=torch.device('cpu'), solve_x=False, collect_info=True):
		self.x_params = list(x_params)
		self.y_params = list(y_params)
		self.lr = lr
		self.device = device
		self.weight_decay = weight_decay
		self.solve_x = solve_x
		self.collect_info = collect_info

		self.old_x = None
		self.old_y = None

	def zero_grad(self):
		zero_grad(self.x_params)
		zero_grad(self.y_params)

	def getinfo(self):
		if self.collect_info:
			return self.norm_bx, self.norm_by, self.time_for_cg, self.norm_gradx, self.norm_grady, \
														self.norm_cgx, self.norm_cgy, self.iter_num
		else:
			raise ValueError('No update info stored. Set collect_info = True before calling this method')


	def step(self, lp_x, lp_y, mh):

		############################ gradient #####################################
		# grad_x_lp_x * a
		grad_x_lp_x = autograd.grad(lp_x, self.x_params, create_graph = True, retain_graph = True)
		grad_x_lp_x = torch.cat([g.contiguous().view(-1,1) for g in grad_x_lp_x])

		# grad_y_lp_y * a
		grad_y_lp_y = autograd.grad(lp_y, self.y_params, create_graph = True, retain_graph = True)
		grad_y_lp_y = torch.cat([g.contiguous().view(-1,1) for g in grad_y_lp_y])
		###########################################################################

		############################ h-v product ##################################
		grad_y_mh = autograd.grad(mh, self.y_params, create_graph = True, retain_graph = True)
		grad_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_mh])
		# (D^2_xy g)(nabla_y g)
		grad_x_y_mh = autograd.grad(grad_y_mh, self.x_params, grad_outputs = grad_y_lp_y, retain_graph = True)
		grad_x_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_y_mh])

		grad_x_mh = autograd.grad(mh, self.x_params, create_graph = True, retain_graph = True)
		grad_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_mh])
		# (D^2_yx g)(nabla_x g)
		grad_y_x_mh = autograd.grad(grad_x_mh, self.y_params, grad_outputs = grad_x_lp_x, retain_graph = True)
		grad_y_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_x_mh])
		###########################################################################

		############## conjugate gradient to compute hessian inverse * vector ##############
		b_x = grad_x_lp_x - self.lr * grad_x_y_mh
		b_y = grad_y_lp_y - self.lr * grad_y_x_mh

		if self.collect_info:
			self.norm_bx = torch.norm(b_x, p = 2)
			self.norm_by = torch.norm(b_y, p = 2)
			self.time_for_cg = time.time()

		if self.solve_x:

			cg_y, self.iter_num = conjugate_gradient(grad_x_lp = grad_y_lp_y, grad_y_lp = grad_x_lp_x, 
														grad_x_mh = grad_y_mh, grad_y_mh = grad_x_mh,
														x_params = self.y_params, y_params = self.x_params,
														b = b_y, x = self.old_y, nsteps = b_y.shape[0], lr = self.lr, 
														device = self.device)

			grad_x_y_mh_cg_y = autograd.grad(grad_y_mh, self.x_params, grad_outputs = cg_y, retain_graph = False)
			grad_x_y_mh_cg_y = torch.cat([g.contiguous().view(-1,1) for g in grad_x_y_mh_cg_y])
			cg_x = grad_x_lp_x + (-self.lr * grad_x_y_mh_cg_y)
			self.old_x = cg_x

		else:

			cg_x, self.iter_num = conjugate_gradient(grad_x_lp = grad_x_lp_x, grad_y_lp = grad_y_lp_y, 
														grad_x_mh = grad_x_mh, grad_y_mh = grad_y_mh,
														x_params = self.x_params, y_params = self.y_params,
														b = b_x, x = self.old_x, nsteps = b_x.shape[0], lr = self.lr, 
														device = self.device)
			
			grad_y_x_mh_cg_x = autograd.grad(grad_x_mh, self.y_params, grad_outputs = cg_x, retain_graph = False)
			grad_y_x_mh_cg_x = torch.cat([g.contiguous().view(-1,1) for g in grad_y_x_mh_cg_x])
			cg_y = grad_y_lp_y + (-self.lr * grad_y_x_mh_cg_x)
			self.old_y = cg_y

		if self.collect_info:
			self.time_for_cg = time.time() - self.time_for_cg
		####################################################################################

		############################## update the parameters ##############################
		index = 0
		for p in self.x_params:
			if self.weight_decay != 0:
				p.data.add_(-self.weight_decay * p)
			p.data.add_(-self.lr * cg_x[index : index + p.numel()].reshape(p.shape))
			index += p.numel()
		if index != cg_x.numel():
			raise ValueError('CG size mismatch')

		index = 0
		for p in self.y_params:
			if self.weight_decay != 0:
				p.data.add_(-self.weight_decay * p)
			p.data.add_(-self.lr * cg_y[index : index + p.numel()].reshape(p.shape))
			index += p.numel()
		if index != cg_y.numel():
			raise ValueError('CG size mismatch')
		###################################################################################

		if self.collect_info:
			self.norm_gradx = torch.norm(grad_x_lp_x, p = 2)
			self.norm_grady = torch.norm(grad_y_lp_y, p = 2)
			self.norm_cgx = torch.norm(cg_x, p = 2)
			self.norm_cgy = torch.norm(cg_y, p = 2)

		self.solve_x = False if self.solve_x else True

class RCGD(object):
	def __init__(self, x_params, y_params, lr=0.01, beta = 0.99, eps = 1e-8, weight_decay=0, device=torch.device('cpu'), solve_x=False, collect_info=True):
		self.x_params = list(x_params)
		self.y_params = list(y_params)
		self.lr = lr
		self.weight_decay = weight_decay
		self.device = device
		self.solve_x = solve_x
		self.collect_info = collect_info
		self.beta = beta
		self.eps = eps

		self.sq_avgx = None
		self.sq_avgy = None
		self.old_x = None
		self.old_y = None
		self.count = 0

	def zero_grad(self):
		zero_grad(self.x_params)
		zero_grad(self.y_params)

	def getinfo(self):
		if self.collect_info:
			return self.norm_gradx, self.norm_grady, self.norm_cg_x, self.norm_cg_y, self.norm_b_x, self.norm_b_y, self.norm_vx, self.norm_vy, self.norm_sq_avgx, \
											self.norm_sq_avgy, self.lr_x_max, self.lr_y_max, self.time_for_cg, self.iter_num
		else:
			raise ValueError('No update info stored. Set collect_info = True before calling this method')

	def step(self, lp_x, lp_y, mh):

		self.count += 1
		############################ gradient #####################################
		grad_x_lp_x = autograd.grad(lp_x, self.x_params, create_graph = True, retain_graph = True)
		grad_x_lp_x = torch.cat([g.contiguous().view(-1,1) for g in grad_x_lp_x])

		grad_y_lp_y = autograd.grad(lp_y, self.y_params, create_graph = True, retain_graph = True)
		grad_y_lp_y = torch.cat([g.contiguous().view(-1,1) for g in grad_y_lp_y])
		###########################################################################

		######################### updated learning rate with RMSprop ######################
		if self.sq_avgx is None:
			self.sq_avgx = torch.zeros((grad_x_lp_x.shape[0],1), requires_grad = False, device = self.device)
		if self.sq_avgy is None:
			self.sq_avgy = torch.zeros((grad_y_lp_y.shape[0],1), requires_grad = False, device = self.device)

		self.sq_avgx = self.beta * self.sq_avgx + (1-self.beta) * (grad_x_lp_x ** 2)
		self.sq_avgy = self.beta * self.sq_avgy + (1-self.beta) * (grad_y_lp_y ** 2)
		
		bias_correction = 1 - self.beta ** self.count
		self.v_x = self.sq_avgx / bias_correction
		self.v_y = self.sq_avgy / bias_correction

		#self.sq_avgx = self.sq_avgx / ()
		#self.sq_avgy = self.sq_avgy / (1 - beta ** self.count)

		#lr_x = self.lr / (self.eps + self.v_x.sqrt())
		#lr_y = self.lr / (self.eps + self.v_y.sqrt())

		lr_x = math.sqrt(bias_correction) * self.lr / (self.eps + self.sq_avgx.sqrt())
		lr_y = math.sqrt(bias_correction) * self.lr / (self.eps + self.sq_avgy.sqrt())
		#################################################################################

		############################# scaled gradient ############################
		# check this with not detaching
		scaled_grad_x_lp_x = lr_x * grad_x_lp_x.detach()
		scaled_grad_y_lp_y = lr_y * grad_y_lp_y.detach()
		##########################################################################

		############################ h-v product ######################################
		grad_y_mh = autograd.grad(mh, self.y_params, create_graph = True, retain_graph = True)
		grad_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_mh])
		grad_x_y_mh = autograd.grad(grad_y_mh, self.x_params, grad_outputs = scaled_grad_y_lp_y, retain_graph = True)
		grad_x_y_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_y_mh])

		grad_x_mh = autograd.grad(mh, self.x_params, create_graph = True, retain_graph = True)
		grad_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_x_mh])
		grad_y_x_mh = autograd.grad(grad_x_mh, self.y_params, grad_outputs = scaled_grad_x_lp_x, retain_graph = True)
		grad_y_x_mh = torch.cat([g.contiguous().view(-1,1) for g in grad_y_x_mh])
		###############################################################################

		######################### b vector in CG ##########################
		b_x = (grad_x_lp_x - grad_x_y_mh).detach()
		b_y = (grad_y_lp_y - grad_y_x_mh).detach()
		###################################################################

		if self.collect_info:
			self.norm_bx = torch.norm(b_x, p = 2)
			self.norm_by = torch.norm(b_y, p = 2)
			self.time_for_cg = time.time()

		############## conjugate gradient to compute hessian inverse * vector #############
		if self.solve_x:
			b_y = b_y * lr_y.sqrt()
			cg_y, self.iter_num = general_conjugate_gradient(grad_x_lp = grad_y_lp_y, grad_y_lp = grad_x_lp_x,
																grad_x_mh = grad_y_mh, grad_y_mh = grad_x_mh,
																x_params = self.y_params, y_params =  self.x_params,
																b = b_y, x = self.old_y, nsteps = b_y.shape[0],
																lr_x = lr_y, lr_y = lr_x, device = self.device)
			cg_y = - lr_y.sqrt() * cg_y.detach()
			grad_x_y_mh_cg_y = autograd.grad(grad_y_mh, self.x_params, grad_outputs = cg_y, retain_graph = False)
			grad_x_y_mh_cg_y = torch.cat([g.contiguous().view(-1,1) for g in grad_x_y_mh_cg_y]).add_(grad_x_lp_x).detach_()
			cg_x = - lr_x * grad_x_y_mh_cg_y
			self.old_x = lr_x.sqrt() * grad_x_y_mh_cg_y

		else:
			b_x = b_x * lr_x.sqrt()
			cg_x, self.iter_num = general_conjugate_gradient(grad_x_lp = grad_x_lp_x, grad_y_lp = grad_y_lp_y,
																grad_x_mh = grad_x_mh, grad_y_mh = grad_y_mh,
																x_params = self.x_params, y_params =  self.y_params,
																b = b_x, x = self.old_x, nsteps = b_x.shape[0],
																lr_x = lr_x, lr_y = lr_y, device = self.device)
			cg_x = - lr_x.sqrt() * cg_x.detach()
			grad_y_x_mh_cg_x = autograd.grad(grad_x_mh, self.y_params, grad_outputs = cg_x, retain_graph = False)
			grad_y_x_mh_cg_x = torch.cat([g.contiguous().view(-1,1) for g in grad_y_x_mh_cg_x]).add_(grad_y_lp_y).detach_()
			cg_y = - lr_y * grad_y_x_mh_cg_x
			self.old_y = lr_y.sqrt() * grad_y_x_mh_cg_x
		###################################################################################

		if self.collect_info:
			self.time_for_cg = time.time() - self.time_for_cg

		############################# update the parameters ###############################
		index = 0
		for p in self.x_params:
			if self.weight_decay != 0:
				p.data.add_(- self.weight_decay * p)
			p.data.add_(cg_x[index : index + p.numel()].reshape(p.shape))
			index += p.numel()
		if index != cg_x.numel():
			raise ValueError('CG size mismatch')
		
		index = 0
		for p in self.y_params:
			if self.weight_decay != 0:
				p.data.add_(- self.weight_decay * p)
			p.data.add_(cg_y[index : index + p.numel()].reshape(p.shape))
			index += p.numel()
		if index != cg_y.numel():
			raise ValueError('CG size mismatch')
		###################################################################################

		if self.collect_info:
			self.norm_gradx = torch.norm(grad_x_lp_x, p = 2)
			self.norm_grady = torch.norm(grad_y_lp_y, p = 2)
			self.norm_cgx = torch.norm(cg_x, p = 2)
			self.norm_cgy = torch.norm(cg_y, p = 2)
			# moving averages with bias correction
			self.norm_vx = torch.norm(self.v_x, p = 2)
			self.norm_vy = torch.norm(self.v_y, p = 2)
			# moving averages without bias correction
			self.norm_sq_avgx = torch.norm(self.sq_avgx, p = 2)
			self.norm_sq_avgy = torch.norm(self.sq_avgy, p = 2)
			self.lr_x_max = lr_x.max()
			self.lr_y_max = lr_x.max()

		self.solve_x = False if self.solve_x else True