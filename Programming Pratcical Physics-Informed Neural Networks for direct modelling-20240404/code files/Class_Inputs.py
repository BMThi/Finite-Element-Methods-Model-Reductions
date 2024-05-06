#%% Library imports

import torch

#%% Inputs class definition

class Inputs:
    """
    Inputs class, used to define the collocation points sampled in the physical domain. 
    """
    def __init__(self, device, variable_sampling, 
                 N_coloc_variable, variable_boundaries, 
                 parametric, parameter_sampling, N_coloc_parameter,
                 parameter_boundaries, parameter_dim, 
                 test_size = 0, seed = None):
        
        # Seed for random colocation points and training/testing sets reproductibility
        if (seed != None):
            torch.manual_seed(seed)
        
        self.device = device
        self.parametric = parametric
        self.test_size = test_size
        
        self.variable_sampling = variable_sampling
        self.N_coloc_variable = N_coloc_variable
        
        self.variable_min = variable_boundaries[0]
        self.variable_max = variable_boundaries[1]
        
        if self.parametric:
            
            self.parameter_sampling = parameter_sampling
            self.N_coloc_parameter = N_coloc_parameter
            
            self.parameter_min = parameter_boundaries[0]
            self.parameter_max = parameter_boundaries[1]
            
            self.parameter_dim = parameter_dim
            
        else:
            
            self.N_coloc_parameter = 1
        
        self.grid_variable = torch.linspace(self.variable_min, 
                                            self.variable_max, 
                                            self.N_coloc_variable).view(-1, 1)
            
        if (variable_sampling == 'grid'):
            
            self.variable = self.grid_variable.detach().clone()
            
        elif (variable_sampling == 'random'):
            
            self.variable = (self.variable_max-self.variable_min)*torch.rand(self.N_coloc_variable, 1) + self.variable_min*torch.ones(self.N_coloc_variable, 1)
            
        if self.parametric:
            
            if (parameter_sampling == 'grid'):
                
                self.parameter_0 = torch.linspace(self.parameter_min, self.parameter_max, self.N_coloc_parameter).view(-1, 1)
                self.parameter = self.parameter_0.detach().clone()
                
                for i in range (self.parameter_dim - 1):
                    self.parameter = torch.hstack((self.parameter.repeat(self.N_coloc_parameter, 1), 
                                                   self.parameter_0.repeat_interleave(self.parameter.shape[0]).view(-1, 1)))
                
            elif (parameter_sampling == 'random'):
                
                self.parameter = (self.parameter_max-self.parameter_min)*torch.rand(self.N_coloc_parameter, self.parameter_dim) + self.parameter_min*torch.ones(self.N_coloc_parameter, self.parameter_dim)
                
            self.all = torch.hstack((self.variable.repeat(self.parameter.shape[0], 1), 
                                     self.parameter.repeat_interleave(self.variable.shape[0], dim = 0))).detach().clone()
            
            self.BC_in = torch.hstack((torch.tensor(self.variable_min).repeat(self.parameter.shape[0], 1), 
                                       self.parameter)).detach().clone()
            
            self.BC_out = torch.hstack((torch.tensor(self.variable_max).repeat(self.parameter.shape[0], 1), 
                                       self.parameter)).detach().clone()
            
            self.grid_parameter = torch.linspace(self.parameter_min, self.parameter_max, 10)
            
            self.grid_display = (self.parameter_max + self.parameter_min)/2 * torch.ones(self.N_coloc_variable * 10, self.all.shape[1]).detach().clone()
            self.grid_display[:, 0] = self.variable.flatten().repeat(10)
            self.grid_display[:, 1] = self.grid_parameter.repeat_interleave(self.N_coloc_variable)
            
            self.grid_RK4 = (self.parameter_max + self.parameter_min)/2 * torch.ones(10, self.parameter_dim).detach().clone()
            self.grid_RK4[:, 0] = self.grid_parameter
            
        else:
            
            self.all = self.variable.detach().clone()
            
            self.BC_in = self.grid_variable[0].detach().clone()
            self.BC_out = self.grid_variable[-1].detach().clone()
            
        self.all.requires_grad = True
        self.BC_in.requires_grad = True
        self.BC_out.requires_grad = True
        
        self.N_coloc = self.all.shape[0]
        
        self.N_coloc_train = int(self.N_coloc*(1-test_size))
        self.N_coloc_test = int(self.N_coloc*test_size)
        
        self.all_shuffled = self.all[torch.randperm(self.N_coloc)].detach().clone()
        
        self.train = self.all_shuffled[:self.N_coloc_train]
        self.train.requires_grad = True
        
        if (self.test_size > 0):
            self.test = self.all_shuffled[-self.N_coloc_test:]
            self.test.requires_grad = True