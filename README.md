# Automated ECG quality assessment

This repository conains code to assess the noise level present in ECG recordings. ECG recordings can get a binary label of either *acceptable* or *unacceptable*. Where *acceptable* means that the ECG recording contains little noise and is suitable to be used for analysis and diagnosis. And *unacceptable* means that the ECG recording contains too much noise and is unsuitable for analysis. 

The code is divided in two folders. The toolbox and CNN-LSTM:

## Toolbox 
This folder contains the code for a toolbox to display ECG recordings. Running the `toolbox/main.py` file will display the following toolbox:

![screenshot of toolbox](https://github.com/Kirina/Automated_ecg_assessment/blob/c9e659bf2a45ab2bf8e3cf973a6608e38baa158e/Toolbox_12_lead_example.png)

Clinking the `Import Data` button will open the file explorer where you can select an ECG recording to analyse. This data should be 10 seconds, 500 Hz and between 1 and 12 leads (data shape: nr_leads x 5000). The toolbox is currently able to accept files with the following extensions: .txt, .csv, .hea, .xls, .xlsx, and .wav.. Clicking the `Process` button will display a results table with the results of individual quality assessment algorithms and the final consensus.  

## CNN-LSTM

The CNN part of the algorithm consists of three blocks that downsample the data through convolutions. Each block consists of a CNN layer, a max pooling layer, and a dropout layer.

## Running Toolbox and CNN-LSTM

### Downloading and running Toolbox
The toolbox requires the CNN-LSTM model to run. A saved model in the `CNN_LSTM/saved_model` folder can be used. Alternatively, you can run and save the model yourself. How to do this is explained in the next section. 

The path to the model needs to be changed in the `parameters_gui` file. 
The `requirements_toolbox.txt` file in the toolbox folder contains all the packages needed to run the toolbox. 
`$ pip install -r requirements_toolbox.txt` should download all the packages needed to run the toolbox in your virtual environment.

Running the `toolbox/main.py` file will display the toolbox. 

### Downloading and running CNN-LSTM
The CNN-LSTM runs in python 3.8.11
The `requirements_cnn_lstm.txt` file contains all the packages needed to run the CNN-LSTM notebooks. 
`$ pip install -r requirements_cnn_lstm.txt` should download all the packages needed to run the CNN-LSTM notebooks. 

The CinC11 dataset used is the relabeled version by [Kramer](https://github.com/LinusKra/ECGAssess)

There are three versions of the CNN-LSTM notebook with different datasets:
#### CinC11 and CinC17 unbalanced
Uses the following files: `parameters_CINC11CINC17_unbalanced.py`, `generate_CINC11_CINC17_dataset.ipynb`, `generate_BUTQDB_dataset.ipynb`, `data_storage_utils.py`, `data_preprocessing_utils.py`, `CNN_LSTM_train_CINC11CINC17_unbalanced_test_butqdb.ipynb`

1. Change the path variable names in the `parameters_CINC11CINC17_unbalanced.py` file. 
2. Generate the datasets with `generate_CINC11_CINC17_dataset.ipynb` in this file change the parameters import to `from parameters_CINC11CINC17_unbalanced import parameters`. And generate the test dataset with `generate_BUTQDB_dataset.ipynb`.
3. Run `CNN_LSTM_train_CINC11CINC17_unbalanced_test_butqdb.ipynb`

#### CinC11 and CinC17 balanced
Uses the following files: `parameters_CINC11CINC17_balanced.py`, `generate_CINC11_CINC17_dataset.ipynb`, `generate_BUTQDB_dataset.ipynb`, `data_storage_utils.py`, `data_preprocessing_utils.py`, `CNN_LSTM_train_CINC11CINC17_balanced_test_butqdb.ipynb`

1. Change the path variable names in the `parameters_CINC11CINC17_balanced.py` file. 
2. Generate the datasets with `generate_CINC11_CINC17_dataset.ipynb` in this file change the parameters import to `from parameters_CINC11CINC17_balanced import parameters`. And generate the test dataset with `generate_BUTQDB_dataset.ipynb`.
3. Run `CNN_LSTM_train_CINC11CINC17_balanced_test_butqdb.ipynb`

#### Kramer
Uses the following files: `parameters_kramer.py`, `generate_kramer_dataset_train_and_test.ipynb`, `data_storage_utils.py`, `data_preprocessing_utils.py`, `CNN_LSTM_train_kramer_test_kramer.ipynb`
 
 1. Change the path variable names in the `parameters_kramer.py` file. 
2. Generate the datasets with `generate_kramer_dataset_train_and_test.ipynb` in this file change the parameters import to `from parameters_kramer import parameters`.
3. Run `CNN_LSTM_train_kramer_test_kramer.ipynb`
