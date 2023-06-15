import numpy as np
from scipy.signal import find_peaks
from scipy import stats



#------------data preprocessing --------------------------------#

def peaks_troughs(data, height):
    """Curve registration
    ------------------------------
    recording the start of a curve by looking for a peak or a trough
    """
    data = data.values.flatten()
    peaks,_ = find_peaks(data, height)
    troughs,_ = find_peaks(-data, height)
    return peaks

def avg_percentage_stride(troughs, num_interpolate):
    troughs_shifted = np.roll(troughs, 1)  # Shift troughs array by 1 position
    troughs_shifted[0] = 0  # Set the first element to 0
    total_data_points = troughs - troughs_shifted  # Compute the number of data points between each peak/trough
    total_data_points = total_data_points[1:]  # Exclude the first element
    perc_div = np.linspace(0.0, 100.0, num=num_interpolate, endpoint=True) / 100
    data_points = total_data_points[:, np.newaxis] * perc_div  # Datapoints at interpolated points
    data_indices = data_points + troughs[:-1, np.newaxis]  # Locate the actual measurements in the dataset
    return data_indices

def modified_data_1D(data, height, num_interpolate):
    troughs_dataset = peaks_troughs(data, height)
    sampling_btn_troughs = avg_percentage_stride(troughs_dataset, num_interpolate)
    indices = sampling_btn_troughs.astype(int)
    dataset = data.to_numpy()[indices]
    return dataset, sampling_btn_troughs

def funcMat_to_point(data, perc_avg_data=0.1, num_interpolate=100, outlier_threshold = 3):
    model_data = data[:int(perc_avg_data * len(data))]  # Select the first 'perc_avg_data' portion of the data
    average_curve = np.mean(model_data, axis=0, keepdims=True)  # Compute the mean along the first axis
    l2_norms = np.linalg.norm(average_curve - data, axis=1)
    l2_norms = l2_norms[stats.zscore(l2_norms) < outlier_threshold]  # Remove outliers
    return l2_norms


#--------------------------------------------------------------#