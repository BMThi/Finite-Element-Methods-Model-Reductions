#%% Library imports

import torch
import numpy as np

#%% Physical constants

g = 9.81 #Gravity in m/s²
q = 1 # Volumetric flow rate per unit width, in m^2/s
h_BC = 0.8 # Water height boundary condition for reference solution
regime = "subcritical" # Flow regime

hc = (q**2/g)**(1/3)

bathy = torch.tensor(np.load('bathy.npy'))/7
bathy_prime = bathy[1:] - bathy[:-1]
bathy_prime = torch.hstack((bathy_prime, bathy_prime[-1]))
bathy_x = torch.arange(bathy.shape[0])

#%% Helper functions

def numpy_interpolator(device, x, domain, y):
    """
    Simple numpy interpolator for Pytorch tensors. 
    """

    y_interpolated = torch.from_numpy(np.interp(x.detach().clone().cpu().numpy().flatten(), 
                                             domain.detach().clone().cpu().numpy().flatten(), 
                                             y.detach().clone().cpu().numpy().flatten())).view(-1, 1).to(device)
    
    return y_interpolated

def bathymetry_interpolator(device, x):
    """
    Bathyemetry function that outputs the value of the bathymetry and its derivative for any location x in the domain. 
    """

    b = numpy_interpolator(device, x, bathy_x, bathy)
    b_prime = numpy_interpolator(device, x, bathy_x, bathy_prime)

    return b, b_prime

def Ks_function(x, k, model, inputs):
    """
    Function used for the reconstruction of the spatially-distributed Ks parameter with P0 or P1 interpolation. 
    """
    
    if (model.k_interpolation == 'P0'):
        
        subdomains = torch.linspace(inputs.variable_min, inputs.variable_max,
                                    k.shape[0] + 1)      
        
        indices = (torch.bucketize(x, subdomains) - 1).clamp(min = 0, max = k.shape[0] - 1) 
        
        return k[indices]
    
    elif (model.k_interpolation == 'P1'):
        
        if (k.shape[0] == 1):
            return k*torch.ones(x.shape[0])
        
        else:
            
            subdomains = torch.linspace(inputs.variable_min, inputs.variable_max,
                                        k.shape[0])   
            
            subdomains_sizes = subdomains[1:] - subdomains[:-1]
            
            indices = (torch.bucketize(x, subdomains) - 1).clamp(min = 0, max = k.shape[0] - 1) 
        
            alpha = (x - subdomains[indices])/subdomains_sizes[indices]
        
            return k[indices+1]*alpha + k[indices]*(1-alpha)
        
def Ks_function_parametric(x, k, model, inputs):
    
    if (model.k_interpolation == 'P0'):
        
        subdomains = torch.linspace(inputs.variable_min, inputs.variable_max,
                                    k.shape[1] + 1) 
        
        indices = (torch.bucketize(x, subdomains) - 1).clamp(min = 0, max = k.shape[1] - 1) 
        
        return k[torch.arange(len(indices)), indices.flatten()].view(-1, 1)

    elif (model.k_interpolation == 'P1'):
        
        if (inputs.parameter_dim == 1):
            return k*torch.ones(x.shape[0])
    
        else:
            
            subdomains = torch.linspace(inputs.variable_min, inputs.variable_max,
                                        k.shape[1])   
            
            subdomains_sizes = subdomains[1:] - subdomains[:-1]
            
            indices = (torch.bucketize(x, subdomains) - 1).clamp(min = 0, max = k.shape[1] - 1) 
        
            alpha = (x - subdomains[indices])/subdomains_sizes[indices]
        
            return k[torch.arange(len(indices)), indices.flatten() + 1].view(-1, 1)*alpha + k[torch.arange(len(indices)), indices.flatten()].view(-1, 1)*(1-alpha)
    
#%% Loss functions

def J_res(model, inputs, domain):
    """
    Loss function for physical model residual calculated on a given domain. 
    """

    #You have to complete the J_res loss function here !
    b_prime = bathymetry_interpolator(model.device, domain)[1]

    h =  model(domain).clamp(min = 1e-6)
    h_x = torch.autograd.grad(h, domain, grad_outputs=torch.ones_like(h), create_graph=True, retain_graph=True)[0]

    Fr2 = q**2/(g*h**3)
    j = q**2/((Ks_function(domain, model.k, model, inputs))**2 * h**(10/3))

    residual = h_x + (b_prime+j)/(1-Fr2)

    return 1/domain.shape[0]*torch.norm(residual, p = 2)**2

def J_res_parametric(model, inputs, domain):
    """
    Loss function for physical model residual calculated on a given domain. 
    """

    #Once J_res and J_BC work fine, you have to complete the parametric version of J_res here !
    
    pass

def J_BC(h_tilde_BC):
    """
    Loss function for boundary condition. 
    """
    
    #You have to complete the expression of J_BC here !
    
    pass

def J_pre(model, domain):
    """
    Loss function for pre-training. 
    """
    
    predictions = model(domain)
    
    return 1/domain.shape[0]*torch.norm(predictions - h_BC, p = 2)**2

def J(model, inputs, domain):
    """
    Total J function.
    """
        
    if (regime == 'subcritical'):
        h_tilde_BC = model(inputs.BC_out)
    elif (regime == 'supercritical'):
        h_tilde_BC = model(inputs.BC_in)
        
    if inputs.parametric:
        
        if model.normalize_J:
            model.J_res_0 = J_res_parametric(model, inputs, domain).detach().clone()
            model.J_BC_0 = J_BC(h_tilde_BC).detach().clone()
            model.J_pre_0 = J_pre(model, domain).detach().clone()
            model.normalize_J = False
       
        return (model.lambdas['res']*1/model.J_res_0*J_res_parametric(model, inputs, domain),
                model.lambdas['BC']*1/model.J_BC_0*J_BC(h_tilde_BC),
                model.lambdas['pre']*1/model.J_pre_0*J_pre(model, domain))   
        
    else:
        
        if model.normalize_J:
            model.J_res_0 = J_res(model, inputs, domain).detach().clone()
            model.J_BC_0 = J_BC(h_tilde_BC).detach().clone()
            model.J_pre_0 = J_pre(model, domain).detach().clone()
            model.normalize_J = False
       
        return (model.lambdas['res']*1/model.J_res_0*J_res(model, inputs, domain),
                model.lambdas['BC']*1/model.J_BC_0*J_BC(h_tilde_BC),
                model.lambdas['pre']*1/model.J_pre_0*J_pre(model, domain))   

#%% Reference solution from classical solver RK4
    
def compute_ref_solution(model, inputs, k, dx):
    """
    Function used to compute the reference solution with RK4 method for solutions comparison. 
    """
    
    k = k.float()
    
    domain = torch.linspace(inputs.variable_min, inputs.variable_max, 
                            int((inputs.variable_max-inputs.variable_min)/dx)).view(-1, 1)
    
    bathy = bathymetry_interpolator(model.device, domain)[0]
    slope = -bathymetry_interpolator(model.device, domain)[1]
    
    bathy_inputs = bathymetry_interpolator(model.device, inputs.grid_variable)[0]
    
    def backwater_model(x, h, k):
        Fr = q/(g*h**3)**(1/2)
        
        if (regime == 'subcritical'):
            return -(numpy_interpolator(model.device, x, domain, slope)-(q/k)**2/h**(10/3))/(1-Fr**2)
        
        elif (regime == 'supercritical'):
            return (numpy_interpolator(model.device, x, domain, slope)-(q/k)**2/h**(10/3))/(1-Fr**2)
        
    def RK4_integrator(k):
        
        i = domain.shape[0]-1

        list_h = []
        list_h.append(h_BC)
        
        list_hn = []
        list_hn.append(((q**2/(numpy_interpolator(model.device, domain[i], domain, slope)*Ks_function(domain[i], k, model, inputs)**2))**(3/10)).item())
        if (regime == 'subcritical'):
        
            while(i > 0):
            
                k1 = backwater_model(domain[i], list_h[-1], Ks_function(domain[i], k, model, inputs))
                k2 = backwater_model(domain[i]-dx/2, list_h[-1]-dx/2*k1, Ks_function(domain[i], k, model, inputs))
                k3 = backwater_model(domain[i]-dx/2, list_h[-1]-dx/2*k2, Ks_function(domain[i], k, model, inputs))
                k4 = backwater_model(domain[i]-dx, list_h[-1]-dx*k3, Ks_function(domain[i], k, model, inputs))
                
                list_h.append((list_h[-1] + dx/6*(k1+2*k2+2*k3+k4)).item())
                
                if (numpy_interpolator(model.device, domain[i], domain, slope) < 0):
                    list_hn.append(np.nan)
                else:
                    list_hn.append(((q**2/(numpy_interpolator(model.device, domain[i], domain, slope)*Ks_function(domain[i], k, model, inputs)**2))**(3/10)).item())
                    
                if (list_h[-1] < hc):
                    raise Warning('You reached supercritical regime !')
                    break
                
                i -= 1
                
            return np.flip(np.array(list_h, dtype=np.float32)).reshape(-1, 1), np.flip(np.array(list_hn, dtype=np.float32)).reshape(-1, 1)
        
        elif (regime == 'supercritical'):
            
            while(i > 0):
            
                k1 = backwater_model(domain[i], list_h[-1], Ks_function(domain[i], k, model, inputs))
                k2 = backwater_model(domain[i]+dx/2, list_h[-1]+dx/2*k1, Ks_function(domain[i], k, model, inputs))
                k3 = backwater_model(domain[i]+dx/2, list_h[-1]+dx/2*k2, Ks_function(domain[i], k, model, inputs))
                k4 = backwater_model(domain[i]+dx, list_h[-1]+dx*k3, Ks_function(domain[i], k, model, inputs))
                
                list_h.append((list_h[-1] + dx/6*(k1+2*k2+2*k3+k4)).item())
                
                if (numpy_interpolator(model.device, domain[i], domain, slope) < 0):
                    list_hn.append(np.nan)
                else:
                    list_hn.append(((q**2/(numpy_interpolator(model.device, domain[i], domain, slope)*Ks_function(domain[i], k, model, inputs)**2))**(3/10)).item())
                    
                if (list_h[-1] > hc):
                    raise Warning('You reached subcritical regime !')
                    break
                
                i -= 1
                
            return np.array(list_h, dtype=np.float32).reshape(-1, 1), np.array(list_hn, dtype=np.float32).reshape(-1, 1)
        
    results = RK4_integrator(k)
    
    h = torch.tensor(results[0].copy())
    h_n = torch.tensor(results[1].copy())
    
    h_c = hc*torch.ones(domain.shape[0], 1)
        
    if not(inputs.parametric):
        
        return {'solution': h, 
                'dx': dx,
                'critical height':h_c, 
                'normal height':h_n, 
                'bathymetry':bathy, 'domain':domain, 
                'bathymetry_inputs':bathy_inputs,
                'parameter_function':Ks_function(domain, k, model, inputs)}
        
    else:
        
        list_h_parametric = []
        for Ks_parametric in inputs.grid_RK4:
            h_parametric = torch.tensor(RK4_integrator(Ks_parametric.view(-1, 1))[0].copy())
            list_h_parametric.append(h_parametric)
            if (len(list_h_parametric)%2 == 0):
                print('-'*65)
                print('Calculating parametric reference solution : {:.2f} % (Ks = {:.2f})'.format(len(list_h_parametric)*10,
                                                                                                  Ks_parametric[0].item()))
        
        return {'solution': h, 
                'dx': dx,
                'critical height':h_c, 
                'normal height':h_n, 
                'bathymetry':bathy, 'domain':domain, 
                'parameter_function':Ks_function(domain, k, model, inputs),
                'parametric solution': torch.cat(list_h_parametric)}
