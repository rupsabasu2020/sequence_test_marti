import numpy as np


#---------------------------------------------------
# compute the martingale from sequence of point data
#----------------------------------------------------


def compute_martingale(datapoints, alpha, frac_data = 3):
  #Generate martingale for your dataset

    #inputs: datapoints, (local-level) alpha, frac_data: fraction for which initial threshold gamma is computed
    data_init = datapoints[0: len(datapoints)//frac_data]      # initial fraction of data , # influences the threshold gamma
    thresh_val = np.quantile(data_init, 1-alpha)    # threshold value 
    rand_variables = 1*(datapoints > thresh_val)    # component of first term  martingale
    first_term = np.cumsum(rand_variables)          # first term of martingale
    times = np.arange(len(datapoints))  # Times array
    marti = first_term- times*alpha   
    return first_term, marti, times
