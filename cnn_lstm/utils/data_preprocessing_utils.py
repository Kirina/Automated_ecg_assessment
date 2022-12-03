import tensorflow as tf
import os
from parameters_gui import parameters_gui
seed = 42
tf.random.set_seed(seed)
        
def normalizer_minus_one_to_one(data, data_range=1):
    """
    data: the data to be normalized
    
    returns: the data normalized to -1 -- 1range
    """
    max_val = float(data.max())
    min_val = float(data.min())
    if max_val == min_val: 
#         print(data)
        data = data * 0
        return data
    
    data = (data_range + data_range) * ((data - min_val) / (max_val - min_val)) - data_range
    return data   

def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_input_length():
    parameter = parameters_gui()
    return parameter['data_length']

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = get_input_length()
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
      [input_len] - tf.shape(waveform),
      dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    commands = ['0' '1']
    spectrogram = get_spectrogram(audio)
    label_id = tf.math.argmax(label == commands)
    return spectrogram, label_id

def preprocess_dataset(files):
    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
    return output_ds 

def get_spectrogram_toolbox(audio):
    spectrogram = get_spectrogram(audio)
    return spectrogram

def preprocess_dataset_toolbox(data):
    """
    data: ECG recordings to be preprocessed
    
    returns: data converted to spectrogam as a dataset object
    """
    files_ds = tf.data.Dataset.from_tensor_slices((data))
    output_ds = files_ds.map(
      map_func=get_spectrogram_toolbox,
      num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds

