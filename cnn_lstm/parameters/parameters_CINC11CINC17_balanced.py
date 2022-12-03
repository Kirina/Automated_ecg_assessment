def parameters():
    parameters = { 
        'path_training_data' : 'path_to_generated_cinc11cinc17_training_dataset',
        'path_BUTQDB_test' : 'path_to_generated_butqdb_test_dataset',
        'path_cinc17' : 'path_to_cinc17', 
        'path_cinc11' : 'path_to_cinc11', 
        'path_BUTQDB' : 'path_to_butqdb',
        'annotation_file_train' : 'path_to_annotation_file_training_set', 
        'annotation_file_test' : 'path_to_annotation_file_testing_set',
        'path_model' : 'path_to_model_save_location', 
        'model_name' : 'CNN_LSTM_model_cinc11cinc17_balanced', 
        'data_length' : 5000,
        'data_length_cinc17_300hz' : 3000,
        'wav_data_range' : 32767, 
        'sampling_frequency' : 500, 
        'length_recording' : 10,       # seconds
        'window_length' : 100, 
        'num_leads' : 12,
        'amount_data' : 2024
    }
    
    return parameters
