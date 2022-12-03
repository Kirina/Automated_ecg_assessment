def parameters():
    parameters = {
        'train_data' : 'path_to_generated_train_data',
        'test_data' : 'path_to_generated_test_data',
        'path_cinc11' : 'path_to_cinc11', 
        'annotation_file_train' : 'path_to_annotation_file_training_set', 
        'annotation_file_test' : 'path_to_annotation_file_testing_set',
        'path_model' : 'path_to_model_save_location', 
        'model_name' : 'CNN_LSTM_model_kramer',
        'data_length' : 5000,
        'wav_data_range' : 32767, 
        'sampling_frequency' : 500, 
        'length_recording' : 10,       # seconds
        'window_length' : 100, 
        'num_leads' : 12
    }
    
    return parameters
