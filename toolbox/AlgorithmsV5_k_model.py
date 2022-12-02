import numpy as np
import scipy.signal
from ecgdetectors import Detectors
import scipy.stats
import neurokit2 as nk
import time
import tensorflow as tf
from data_preprocessing_utils import preprocess_dataset_toolbox, normalizer_minus_one_to_one

def high_frequency_noise_filter(data, max_loss_passband, min_loss_stopband, sampling_frequency=500):
    order, normal_cutoff = scipy.signal.buttord(20, 30, max_loss_passband, 
                                                min_loss_stopband, fs=sampling_frequency)
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data

def baseline_filter(data, max_loss_passband, min_loss_stopband, sampling_frequency=500):
    order, normal_cutoff = scipy.signal.buttord(0.5, 8, max_loss_passband, 
                                                min_loss_stopband, fs=sampling_frequency)
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data

def stationary_signal_check(data, num_leads, window_length):
    result = [0]*num_leads
    for lead_nr in range(1, num_leads + 1):
        window_matrix = np.lib.stride_tricks.sliding_window_view(data[num_leads], window_length)[::10]
        for window in window_matrix:
            if np.amax(window) == np.amin(window):
                result[lead_nr-1] = 1
                break
    return result

def heart_rate_check(data, num_leads, heart_rate_limits, sampling_frequency, length_recording=10):
    result = [0]*num_leads
    for lead_nr in range(1, num_leads + 1):
        beats = Detectors(sampling_frequency).pan_tompkins_detector(data[lead_nr])
        if (len(beats) > ((heart_rate_limits[1]*length_recording)/60)) or \
           (len(beats) < ((heart_rate_limits[0]*length_recording)/60)):
            result[lead_nr-1] = 1
    return result

def signal_to_noise_ratio_check(data, num_leads, SNR_threshold, signal_freq_band, sampling_frequency=500):
    result = [0]*num_leads
    for lead_nr in range(1, num_leads + 1):
        f, power_spectral_density = scipy.signal.periodogram(data[lead_nr], fs=sampling_frequency, scaling="spectrum")
        if sum(power_spectral_density):
            signal_power = sum(power_spectral_density[(signal_freq_band[0]*10):(signal_freq_band[1]*10)])
            SNR = signal_power / (sum(power_spectral_density) - signal_power)
            if SNR < SNR_threshold:
                result[lead_nr-1] = 1
    return result

def CNN_quality_check(data, num_leads, path_model, name_model, sampling_frequency=500):
    ecg_data_list = []
    
    for lead_nr in range(1, num_leads + 1):
        ecg_data = normalizer_minus_one_to_one(data[lead_nr]) # normalize data -1 to 1
        ecg_data_list.append(np.asarray(ecg_data, dtype=np.float32))
        
    ecg_data_list = tf.constant(ecg_data_list)
    sample_ds = preprocess_dataset_toolbox(ecg_data_list)
    model = tf.keras.models.load_model(path_model+'/'+name_model)
    result = [0]*num_leads
    
    for lead_nr, (spectrogram) in enumerate(sample_ds.batch(1)):
        prediction = model(spectrogram)
        probability_0, probability_1 = np.array(tf.nn.softmax(prediction[0]))
        if probability_0 > 0.5: 
            result[lead_nr] = 1
            
    return result
    
def processing(ECG, num_leads, temp_freq, SNR_threshold, signal_freq_band, window_length, 
               heart_rate_limits, max_loss_passband, min_loss_stopband, sampling_frequency, 
               path_model, name_model, length_recording=10):
    second = time.time()
    
    if temp_freq != sampling_frequency:
        resampled_ECG = []
        for n in range(num_leads + 1):
            resampled_ECG.append(nk.signal_resample(ECG[n], sampling_rate=int(temp_freq), 
                                                    desired_sampling_rate=sampling_frequency, method="numpy"))
    else:
        resampled_ECG = ECG
    
    # heart_rate_check needs to be done on unfilered data
    signal_quality = [heart_rate_check(resampled_ECG, num_leads,  heart_rate_limits, sampling_frequency, length_recording)]
    
    filtered_ECG = [resampled_ECG[0]]
    # for loop starts at 1 because first dimension of data is datapoint nr
    for lead in range(1, num_leads + 1):
        filter_noise =  high_frequency_noise_filter(ECG[lead], max_loss_passband, min_loss_stopband, sampling_frequency)
        filter_baseline = baseline_filter(ECG[lead], max_loss_passband, min_loss_stopband, sampling_frequency)
        filtered_ECG.append(filter_noise - filter_baseline)
    
    signal_quality.append(stationary_signal_check(ECG, num_leads, window_length))
    signal_quality.append(signal_to_noise_ratio_check(ECG, num_leads, SNR_threshold, signal_freq_band, sampling_frequency,))
    signal_quality.append(CNN_quality_check(ECG, num_leads, path_model, name_model))
    
    # convert to list so we can later convert to unicode char in-place    
    overall_result = list(np.sum(signal_quality, axis=0)) 
    signal_quality.append(overall_result)

    for quality_index in signal_quality:    
        for lead in range(num_leads):
            if quality_index[lead]:
                quality_index[lead] = u"\u2716"
            else:
                quality_index[lead] = u"\u2714"

    print('Time needed for analysis:', time.time()-second)
    return signal_quality