#%% Libraries imports

import numpy as np
import torch
import matplotlib.pyplot as plt

#%% Reference solution

def display_reference_solution(model, inputs, ref_solution): 
    """
    Function used to display the data for the inverse problem. 
    """
        
    if (model.k_interpolation == 'P0'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k + 1)
        indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()
        
    if (model.k_interpolation == 'P1'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k)
        indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()

    fig, ax = plt.subplots()
    ax.set_title('Reference solution')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['solution'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            color='#1f77b4', label = '$h_{RK4}(x)$')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'g', label = 'b(x)')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['normal height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'y--', label = '$h_n(x)$')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['critical height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'r--', label = '$h_c(x)$')
    ax.scatter(subdomains.detach().clone().cpu().numpy(), 
               ref_solution['bathymetry'].detach().clone().cpu().numpy()[indices], 
               marker = '|', c = 'k', s = 100, label = 'sub.')
    
    ax.set_xlabel(r'$x \ [m]$')
    ax.set_ylabel(r'$y \ [m]$')
    
    ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                    ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                    0, color='green', alpha=.3)
    ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                    ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(),
                    ref_solution['solution'].flatten().detach().clone().cpu().numpy() + ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                    color='blue', alpha=.2)
    
    ax.set_ylim(top = 1.1*max(ref_solution['bathymetry'] + ref_solution['solution']).item(),
                bottom = 0)
    
    ax2 = ax.twinx()

    ax2.set_ylabel(r'$K_s \ [m^{1/3}/s]$', y = 0.14)  
    ax2.plot(ref_solution['domain'].flatten().detach().clone().cpu().numpy(),
             ref_solution['parameter_function'].flatten().detach().clone().cpu().numpy(),
             '-.', label = r'$K_s(x)$', c = 'grey')
    ax2.set_ylim(0, 300)
    custom_ticks = np.arange(0, 100, 20)  
    ax2.set_yticks(custom_ticks) 
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', 
               ncol = 2, prop={'size': 7.5})
        
    plt.show()
    
#%% Results
    
def display_results(model, inputs, ref_solution, plot_coloc = False):
    """
    Function used to display the result of the training. 
    """
    
    if not(inputs.parametric):

        if (model.k_interpolation == 'P0'):
            subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                        max(ref_solution['domain']).item(),
                                        model.dim_k + 1)
            indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()
            
        if (model.k_interpolation == 'P1'):
            subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                        max(ref_solution['domain']).item(),
                                        model.dim_k)
            indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()
        
        if inputs.parametric:
            inputs_display = torch.hstack((ref_solution['domain'].detach().clone(),
                                           model.k.repeat(ref_solution['domain'].shape[0], 1)))
        else:
            inputs_display = ref_solution['domain'].detach().clone()
            
        fig, ax = plt.subplots()
        ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
                ref_solution['solution'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
                color='#1f77b4', label = '$h_{RK4}(x)$')
        ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
                model(inputs_display).detach().clone().cpu().numpy()+ref_solution['bathymetry_inputs'].detach().clone().cpu().numpy(), 
                'k--', label = r'$\tilde{h}(x)$')
        ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
                ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
                'g', label = '$b(x)$')
        ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
                ref_solution['normal height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
                'y--', label = '$h_n(x)$')
        ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
                ref_solution['critical height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
                'r--', label = '$h_c(x)$')
        if plot_coloc:
            ax.scatter(inputs.grid_variable.detach().clone().cpu().numpy(), 
                       ref_solution['bathymetry_inputs'].detach().clone().cpu().numpy(), 
                       label = 'coloc', color='black', s = 10)
        ax.scatter(subdomains.detach().clone().cpu().numpy(), 
                   ref_solution['bathymetry'].detach().clone().cpu().numpy()[indices], 
                   marker = '|', c = 'k', s = 100, label = 'sub.')
    
        ax.set_xlabel(r'$x \ [m]$')
        ax.set_ylabel(r'$y \ [m]$')
        
        
        h_true = ref_solution['solution'].flatten().detach().clone().cpu().numpy()
        h_est = model(inputs_display).flatten().detach().clone().cpu().numpy()
    
        ax.set_title('Calibrated model at iteration {}, RMSE = {:.4e}'.format(model.iter, np.linalg.norm(h_true - h_est, ord = 2)/np.linalg.norm(h_true, ord = 2)))
        ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                        ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                        0, color='green', alpha=.3)
        ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                        ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(),
                        ref_solution['solution'].flatten().detach().clone().cpu().numpy() + ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                        color='blue', alpha=.2)
        
        ax.set_ylim(top = 1.1*max(ref_solution['bathymetry'] + ref_solution['solution']).item(),
                    bottom = 0)
    
        ax2 = ax.twinx()
    
        ax2.set_ylabel(r'$K_s \ [m^{1/3}/s]$', y = 0.14)  
        ax2.plot(ref_solution['domain'].flatten().detach().clone().cpu().numpy(),
                 ref_solution['parameter_function'].flatten().detach().clone().cpu().numpy(),
                 '-.', label = r'$K_s(x)$', c = 'grey')
        ax2.set_ylim(0, 300)
        custom_ticks = np.arange(0, 100, 20)  # Créez des ticks jusqu'à la moitié du graphique
        ax2.set_yticks(custom_ticks)  # Définissez les ticks personnalisés sur l'axe y
    
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', 
                   ncol = 2, prop={'size': 7.5})
        
    else:
        
        estimation = model(inputs.grid_display).detach().clone().cpu().numpy().reshape(10, inputs.N_coloc_variable).T
        sol = ref_solution['parametric solution'].detach().clone().cpu().numpy().reshape(10, ref_solution['domain'].shape[0]).T
        
        vmin = min(np.min(sol), np.min(estimation))
        vmax = max(np.max(sol), np.max(estimation))
        
        fig, axs = plt.subplots(1, 3, width_ratios = [1, 1, 0.08])
    
        fig.suptitle('Calibrated parametric model at iteration {}'.format(model.iter))
    
        ax1 = axs[0]
        ax1.imshow(sol, cmap = 'viridis', 
                  extent = [inputs.parameter_min, inputs.parameter_max, 
                            inputs.variable_max, inputs.variable_min], 
                  vmin = vmin, vmax = vmax, aspect = 'auto')
        
        if (inputs.parameter_dim > 1):
            ax1.set_title('Projected solution')
        else:
            ax1.set_title('Solution')
        ax1.set_xlabel('$K_s \ [m^{1/3}.s^{-1}]$')
        ax1.set_ylabel('$x \ [m]$')
        
        ax2 = axs[1]
        im_est = ax2.imshow(estimation, cmap = 'viridis', 
                           extent = [inputs.parameter_min, inputs.parameter_max, 
                                     inputs.variable_max, inputs.variable_min], 
                           vmin = vmin, vmax = vmax, aspect = 'auto')
        
        if (inputs.parameter_dim > 1):
            ax2.set_title('Projected estimation')
        else:
            ax2.set_title('Estimation')
        ax2.set_xlabel('$K_s \ [m^{1/3}.s^{-1}]$')
        
        cbar_ax = axs[2]
        fig.colorbar(im_est, cax=cbar_ax)
        cbar_ax.set_ylabel('$y \ [m]$')
            
        ax2.axes.get_yaxis().set_visible(False)
    
    plt.show()
    
#%% Training
    
def display_training(model, inputs, ref_solution): 
    """
    Function used to display the training of the Neural Network. 
    """
    
    if inputs.parametric:
        inputs_display = torch.hstack((ref_solution['domain'].detach().clone(),
                                       model.k.repeat(ref_solution['domain'].shape[0], 1)))
    else:
        inputs_display = ref_solution['domain'].detach().clone()
        
    h_true = ref_solution['solution'].flatten().detach().clone().cpu().numpy()
    h_est = model(inputs_display).flatten().detach().clone().cpu().numpy()
        
    print('#'*50)
    print('Final variable RMSE : {:.2e}'.format(np.linalg.norm(h_true - h_est, ord = 2)/np.linalg.norm(h_true, ord = 2)))
    print('#'*50)
    
    J_train = np.asarray(model.list_J_train)[model.list_iter_flag]
    if (inputs.test_size > 0):
        J_test = np.asarray(model.list_J_test)[model.list_iter_flag]
    
    fig, ax = plt.subplots()
    ax.set_title('J function during network optimization')
    ax.set_xlabel('L-BFGS iterations')
    ax.set_ylabel('J function') 
    ax.set_yscale('log')
    if (inputs.test_size > 0):
        ax.plot(J_train[:, 0], label = '$J$ train', color='#1f77b4')
        ax.plot(J_test[:, 0], label = '$J$ test', color='#ff7f0e')
        ax.plot(J_train[:, 1], '--*', label = '$J_{res}$ train', color='#1f77b4')
        ax.plot(J_test[:, 1], '--*', label = '$J_{res}$ test', color='#ff7f0e')
    else:
        ax.plot(J_train[:, 0], label = '$J$', color='black')
        ax.plot(J_train[:, 1], label = '$J_{res}$', color='#1f77b4')
    ax.plot(J_train[:, 2], ':', label = '$J_{BC}$', color='#1f77b4')
    ax.plot(J_train[:, 3], label = '$J_{pre-train}$', color='#ff7f0e')
    ax.legend(loc = 'best', ncol = 2, prop={'size': 8})

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(np.asarray(model.list_grad)[model.list_iter_flag], label = r'$\| \frac{\partial J}{\partial \theta} \|_2$')
    ax.set_xlabel('L-BFGS iterations')
    ax.set_ylabel('gradient norm')
    ax.set_title('gradient norm during optimization')
    ax.legend(loc = 'best')
    
    if (len(model.list_J_gradients) > 0):
        
        fig, ax = plt.subplots()
        data = np.asarray(model.list_J_gradients)[model.list_iter_flag]
        ax.set_yscale('log')
        ax.plot(data[:, 0], label = r'$\| \frac{\partial \ J_{res}}{\partial \ \theta} \|_2$')
        ax.plot(data[:, 1], label = r'$\| \frac{\partial \ J_{BC}}{\partial \ \theta} \|_2$')
        ax.plot(data[:, 2], label = r'$\| \frac{\partial \ J_{pre}}{\partial \ \theta} \|_2$')
        ax.set_xlabel('L-BFGS iterations')
        ax.set_ylabel('residual term gradient')
        ax.set_title('Losses gradients during optimization')
        ax.legend(loc = 'best')
        
        # boundaries = (100, 200)
        
        # fig, ax = plt.subplots()
        # data = np.asarray(model.list_J_gradients)[model.list_iter_flag]
        # ax.set_xlim(left = boundaries[0], right = boundaries[1])
        # ax.set_yscale('log')
        # ax.plot(data[:, 0], label = r'$\| \frac{\partial \ J_{res}}{\partial \ \theta} \|_2$')
        # ax.plot(data[:, 1], label = r'$\| \frac{\partial \ J_{BC}}{\partial \ \theta} \|_2$')
        # ax.plot(data[:, 2], label = r'$\| \frac{\partial \ J_{pre}}{\partial \ \theta} \|_2$')
        # ax.set_xlabel('L-BFGS iterations')
        # ax.set_ylabel('residual term gradient')
        # ax.set_title('Residual term gradient during optimization between [{:.0f}, {:.0f}]'.format(boundaries[0],boundaries[1]))
        # ax.legend(loc = 'best')
    
    if (not(inputs.parametric) and h_true.shape[0] == model.list_y[-1].shape[0]):
        
        h_est_iter = np.array(model.list_y).squeeze()[model.list_iter_flag].transpose()
    
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(np.linalg.norm(h_est_iter - h_true.reshape(-1, 1), ord = 2, axis = 0)/np.linalg.norm(h_true, ord = 2), label = r'$h$')
        ax.set_xlabel('L-BFGS iterations')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE on h between RK4 solution and NN solution')

