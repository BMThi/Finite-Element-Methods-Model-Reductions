#%% Libraries and modules imports

import torch 
print('Pytorch version :', torch.__version__)

from Class_PINN import PINN
from Class_Inputs import Inputs
from Backwater_model import J, compute_ref_solution
import display

#%% Cuda setup

# Device choice
use_GPU = True #Choose whether you want to use your GPU or not

if (torch.cuda.is_available() and use_GPU):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on {device} !')

#Additional info when using cuda
if (device.type == 'cuda'):
    print('GPU :', torch.cuda.get_device_name(0))
    print('Total memory :', 
          round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')

#Set default usage to CPU or GPU
torch.set_default_device(device)

#%% Definition of k

k = torch.tensor([40, 30, 50]) #You can modify the value of k here !

#%% Inputs definition

inputs = Inputs(device, 
                variable_sampling = 'grid', #Choose between grid and random for the variable colocation points sampling
                N_coloc_variable = 100, #Choose number of colocation points for the physical variable
                variable_boundaries = (0, 1000), #Choose the boundaries of the domain
                parametric = False, #Choose whether you want parmetric inputs or not
                parameter_sampling = 'grid', #Choose between grid and random for the parameter colocation points sampling
                N_coloc_parameter = 10, #Choose number of colocation points for the parameter (careful, if parameter_sampling == 'grid', it will be at the power of the parameter dimension !)
                parameter_boundaries = (20, 80), #Choose the boundaries of the domain
                parameter_dim = k.shape[0], #Choose the dimension of your parameter here
                test_size = 0, #Choose the testing set / training set ratio
                seed = 0) #Set the seed to any integer number for reproductible randomness

#%% PINN definition

model = PINN(device, inputs = inputs, k = k,
             layers = [1, 60, 60, 60, 1], #Choose the neural network architecture
             k_interpolation = 'P0', #Choose the interpolation method for the k function (P1 or P0)
             seed = 0) #Set the seed to any integer number for reproductible randomness

#%% Reference solution generation

ref_solution = compute_ref_solution(model, inputs, k = k, dx = 10) 

display.display_reference_solution(model, inputs, ref_solution) 

#%%Model training 

model.train_model(J, inputs, ref_solution, 
                  normalize_J = True, #Choose whether you want to normalize J by J_0 or not
                  pre_train_iter = 100, #Choose the number of iterations for pre-training
                  renormalize_J = False, #Choose whether you want to renormalize J after pre-training or not
                  offline_iter = 1000, #Choose the number of iterations for the offline training
                  display_freq = (50, 100)) #Choose the display frequency for the training informations (first value) and the results plot (second value)

#%% Display training and results

display.display_training(model, inputs, ref_solution)

display.display_results(model, inputs, ref_solution, plot_coloc = True)