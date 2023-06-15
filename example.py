import numpy as np
import compute_martingale as comp_marti
import compute_bounds as comp_bounds
import matplotlib.pyplot as plt



# example sequence of data with change-point

len_dataset = 500

Y_1 = np.random.chisquare(df= 20, size=len_dataset//2) + np.random.normal(loc=0, scale=5, size=len_dataset//2)
Y_2 = np.random.chisquare(df = 36, size=len_dataset//2) + np.random.normal(loc=0.5, scale=5, size=len_dataset//2)
X = np.concatenate((Y_1, Y_2))
alpha=0.22
Delta=0.1
partitions = 10
#-------LIL bounds, useful only in large time regimes------------------#
kappa, stop_time = comp_bounds.lil_parameters(alpha, Delta, k=0.05,   add_kappa=0.0005)   
marting_bound= comp_bounds.lil_marti_bound(X, alpha=0.22, Delta=0.1, k=0.05, kappa_val = kappa, t_0= stop_time)

#------- sequential linear bounds, useful in all time regimes
tau, delta, t = comp_bounds.linear_para(Delta, alpha, num_partitions= partitions, stop_time= stop_time)         # tau (intersections) and t  (minimality) are as defined in paper
slope, intercept = comp_bounds.linear_bound(delta, t) 

first_term, marting, time_marti = comp_marti.compute_martingale(X, alpha)


if __name__=="__main__":
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(X, '.', label = 'point data')
    axs[1].plot(marting, label = "martingale")
    for j in range(partitions):
        comp_bounds.abline(slope[j], intercept[j], axs[2], 100, start_val=tau[j], end_val=tau[j+1])
    axs[0].legend()
    axs[1].legend()
    plt.show()