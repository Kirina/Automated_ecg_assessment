def parameters_gui():
    parameters = {
#         'path_training_data' : 'C:\\Users\\kirin\\Documents\\Untitled Folder\\Code Python toolbox\\ECGAssess-main\\Code\\training_data', 
        'path_model' : 'C:\\Users\\kirin\\Documents\\Untitled Folder\\Code Python toolbox\\ECGAssess-main\\Code\\model', 
        'path_save_wav' : 'C:\\Users\\kirin\\Documents\\Untitled Folder\\Code Python toolbox\\ECGAssess-main\\Code\\wav_processing', 
        'name_model' : 'CNN_LSTM_model_cinc11cinc17', 
        'data_length' : 5000,   # datapoints
        'wav_data_range' : 32767, 
        'sampling_frequency' : 500, 
        'max_loss_passband' : 0.1,     # dB
        'min_loss_stopband' : 20,      # dB
        'SNR_threshold' : 0.5,
        'signal_freq_band' : [2, 40],      # from .. to .. in Hz
        'heart_rate_limits' : [25, 300],       # from ... to ... in beats per minute
        'length_recording' : 10,       # seconds
        'window_length' : 100, 
        'num_leads' : 12
    }
    
    return parameters