import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functional_data_projection as fs
import compute_martingale as comp_marti
import compute_bounds as comp_bounds
plt.rcParams["axes.grid"] = True


#---- example file for biomechanical knee angles data ----- #
data= pd.read_csv('/Users/rupsabasu/Documents/DATASETS/pilot_study/hip_knee_ankle.csv')  
right_knee_z_values = data.iloc[25000:, 47]   # just removing some initial points and choosing the right cols
datamat, _ = fs.modified_data_1D(right_knee_z_values, height= 63, num_interpolate= 200) # convert functional data to matrix data 
point_data = fs.funcMat_to_point(datamat, num_interpolate= 200)   # matrix data to point data  # note that outliers beyond 3-std. are removed 

#------------------------ data is now ready for martingale construction! ------------#
alpha= 0.25
Delta = 0.1
partitions = 10
first_term, marting, time_marti = comp_marti.compute_martingale(point_data, alpha)

#-------LIL bounds, useful only in large time regimes------------------#
kappa, stopping_time = comp_bounds.lil_parameters(alpha, Delta, k=0.05,   add_kappa=0.005)   
marting_bound= comp_bounds.lil_marti_bound(point_data, alpha, Delta, k=0.05, kappa_val = kappa, t_0= stopping_time)

#------- sequential linear bounds, useful in all time regimes
tau, delta, t = comp_bounds.linear_para(Delta, alpha, num_partitions= partitions, stop_time= min(stopping_time, len(marting)))         # tau (intersections) and t  (minimality) are as defined in paper
slope, intercept = comp_bounds.linear_bound(delta, t) 



# plot to check
plt.plot(point_data, ".", alpha = 0.3, label = "point data")
plt.plot(marting, label ="martingale statistic")
for j in range(partitions):
    comp_bounds.abline(slope[j], intercept[j], plt, 100, start_val=tau[j], end_val=tau[j+1])
plt.legend()
plt.show()
