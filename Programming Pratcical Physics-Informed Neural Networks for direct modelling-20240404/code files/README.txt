This folder contains all the codes you need for the Programming Pratical:

- Backwater_model.py contains all the information about the physical model, as well as the RK4 integrator for ther reference solution. You will have to modify the J_res and the J_obs loss functions, line 66 and 80 respectively;

- Class_Inputs.py contains the code for the class Inputs, where you define the inputs of your PINN, here it just refers to the 1D spatial domain;

- Clas_Observations.py contains the code for the class Observations, where you can generate the observations from the reference solution to perform the inference;

- CLass_PINN contains the code for the class PINN, which defines the PINN model as well as its training;

- display.py contains all the functions used to display the graphs you will see during the training;

- main.py contains the main code, from where you can control the inputs definition, model definition and training;

- main.ipynb is exactly the same as main.py but in a Jupyter notebook, if you prefer to use notebooks;

- bathy.npy contains the information about the bathymetry stored as a numpy array.