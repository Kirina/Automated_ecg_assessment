import numpy as np
import os
from scipy.io.wavfile import write
seed = 42
np.random.seed(seed)


def create_directory(output_path):
    """
    output_path: the path from source to the output folder 
    
    creates the folders needed to store the output data
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path+'/0'):
        os.makedirs(output_path+'/0')
    if not os.path.exists(output_path+'/1'):
        os.makedirs(output_path+'/1')
    if not os.path.exists(output_path+'/rest'):
        os.makedirs(output_path+'/rest')
        
        print('output directories created')
        
def create_wav_directory(output_path):
    """
    output_path: the path from source to the output folder 
    
    creates the folders needed to store the output data
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def normalizer(data, wav_data_range):
    """
    data: the data to be normalized
    
    returns: the data normalized to -32,768 -- 32,767 range
    """
    max_val = float(data.max())
    min_val = float(data.min())
    if max_val == min_val: 
#         print(data)
        data = data * 0
        return data
    
    data = (wav_data_range + wav_data_range) * ((data - min_val) / (max_val - min_val)) - wav_data_range
    return data        
        
def generate_store_kramer_data(annotations, recording_numbers, wav_data_range, label_count, path_cinc11, path_save, amount_data):
    """
    takes the recording numbers and uses it to find the corresponding annotation and the ECG file
    the ECG file is converted to .wav and stored at the path_save location
    returns: [num label zero, num label one]
    """
    for recording_index, recording in enumerate(recording_numbers): 
        annotation = annotations[recording_index]
        annotation = [1 - int(x) for x in annotation] 

        data_matrix = []
        with open(path_cinc11+f"\\{recording}.txt") as f:
            lines = f.readlines()
            
            for line in lines: 
                line_data_split = line.split('\n')[0].split(',')[1:]
                data_matrix.append(line_data_split)

            for i, data in enumerate(np.array(data_matrix).T):
                data = np.asarray(data, dtype=np.int16)
                data = normalizer(data, wav_data_range)
                label = int(annotation[i])
                
                if label == 0: 
                    data = np.asarray(data, dtype=np.int16)
                    write(path_save+"\\"f"{label}\\{recording}_{i}.wav", 500, data)
                    label_count[label] += 1
                
                elif label_count[label] < amount_data:    
                    data = np.asarray(data, dtype=np.int16)
                    write(path_save+"\\"f"{label}\\{recording}_{i}.wav", 500, data)  
                    label_count[label] += 1          
    return label_count