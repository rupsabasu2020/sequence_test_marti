import numpy as np
from math import e


#--------------------------------------------------------------------
# Two kinds of bounds: (a) Martingale-LIL (b) Martingale-Linear
#--------------------------------------------------------------------


#---------------------------
# (a) Martingale-LIL 
#---------------------------
def lil_parameters(alpha, Delta, k, add_kappa = 0):
    """Compute  parameters necessary for LIL- bound martingale
    -------------------------
    input parameters:
    --------
        alpha: (local-) level of test, also a parameter in computing martingale
        Delta: (global-) level of test, controls false positives
        add_kappa: theoretical requirement is that kappa is greater than value computed here
    -----------------------
    output values:
    --------
    kappa, t_0 : used for computing bounds later
    """
    kappa_num = 1/2 + 1/(23*e*e*e*e) - 0.4*alpha+ np.max((1/(6*e*e))- 0.1*alpha, 0)   
    kappa = (kappa_num/(1-alpha))+ add_kappa     
    t_0 = np.ceil(((e**4)*(1+np.sqrt(k))**2/(kappa*alpha*(1-alpha)))*np.log(1/Delta))  
    return  kappa, t_0




def lil_marti_bound(datapoints, alpha, Delta, k, kappa_val, t_0 = 0):
    """Computing martingale LIL bounds 
    -----------------------------------

    Input parameters:
    
    datapoints, alpha, Delta  ::::: given dataset, local-level and global level of test procedure respectively.

    -----------------------------------

    Output parameters:
    
    marti_bound: martingale LIL bounds (valid only after stopping time t_0)
    
    """

    datapoints = datapoints[int(t_0):, ]
    data_len = len(datapoints)
    times = np.linspace(t_0, t_0 + data_len, num = data_len, dtype = int)   
    V_t = times*alpha*(1-alpha)    
    double_log = 2*np.log(np.log((2*kappa_val*V_t)/(1- np.sqrt(k))))        # first double log term
    second_log = np.log(2/(Delta*np.log((1+np.sqrt(k))/(1-np.sqrt(k)))))    # second term with single log
    marti_bound = np.sqrt(4*kappa_val*V_t*(double_log + second_log)/(1-k))  # final bound (LIL martingale)
    return marti_bound






#--------------------------------------------------------------------
#  (b) Martingale-Linear
#--------------------------------------------------------------------



def linear_para(Delta, alpha, num_partitions, stop_time):
    """Computes parameters for sequential linear bounds
    ---------------------------------------
    Inputs:

    Delta, alpha : (global-) and (local-) level respectively
    num_partitions: number of sequential bounds desired    
    stop_time: is t_0 from LIL bounds, alternatively len(dataset can be used)
    ---------------------------------------
    Outputs:

    tau_j: points of intersection of partition (j-1) with partition (j)
    delta_j: level of test in partition j
    t_j : time points within tau_(j-1) and tau_j

    """
    i = np.linspace(1, num_partitions, num= num_partitions)  # time interval from (0, stop_time) with num_parts   
    delta_j = Delta*np.log(2)/((i+2)*np.log((i+2)**2))     # delta val at each part
    tau_O = 2*alpha*np.log(1/delta_j[0])                    # initial tau0
    t_j = tau_O + (i-1)*(stop_time- tau_O)/(num_partitions-1)   
    Nr = np.sqrt(np.log(1/delta_j[1:])*t_j[1:]) - np.sqrt(np.log(1/delta_j[0:-1])*t_j[0:-1])
    Dr = np.sqrt(np.log(1/delta_j[0:-1])*t_j[1:]) -np.sqrt(np.log(1/delta_j[1:])*t_j[0:-1])
    frac_part = Nr/Dr
    tau_j = np.sqrt(t_j[:-1]*t_j[1:])*frac_part     
    tau_j = np.insert(tau_j, 0, tau_O)               
    tau_j = np.insert(tau_j, len(tau_j), stop_time) 
    return tau_j, delta_j, t_j  


def linear_bound(delta_j, t_j):
    """ Computes sequential linear bounds
    ------------------------
    Inputs:

    delta_j, t_j: parameters computed by helper function linear_para
    ------------
    Outputs:

    slope, intercept of the linear bounds
    """
    slope = np.sqrt(np.log(1/delta_j)/(8*t_j))   
    intercept = np.sqrt(np.log(1/delta_j)*t_j/8)
    return slope, intercept



#-------------------------------------------------------------------
# plots the sequential linear bounds
#-------------------------------------------------------------------


def abline(slope, intercept, axes, number_xvals, end_val, styl_col = None, start_val = 0):
    """Generates plot of the linear bounds
    --------------------------------------
    Inputs:
    slope, intercept: computed by helper function linear bound
    axes : use plt unless subplots are used
    number_xvals: just for plotting the lines!
    start_val: tau_(j-1) # see helper function linear_para()
    end_val: tau_(j))    # see helper function linear_para()
    """
    x_vals = np.linspace(start_val, end_val, num=number_xvals)     # time axis
    y_vals = intercept + slope * x_vals      
    axes.plot(x_vals, y_vals, alpha = 1)
    return y_vals
