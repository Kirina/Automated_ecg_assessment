# Automated ECG quality assessment

This repository conains code to assess the noise level present in ECG recordings. ECG recordings can get a binary label of either *acceptable* or *unacceptable*. Where *acceptable* means that the ECG recording contains little noise and is suitable to be used for analysis and diagnosis. And *unacceptable* means that the ECG recording contains too much noise and is unsuitable for analysis. 

The code is divided in two folders. The toolbox and CNN-LSTM:

## Toolbox 
This folder contains the code for a toolbox to display ECG recordings. Running the `main.py` file in the toolbox folder will display the following toolbox:

![screenshot of toolbox](https://github.com/Kirina/Automated_ecg_assessment/blob/c9e659bf2a45ab2bf8e3cf973a6608e38baa158e/Toolbox_12_lead_example.png)

Clinking the `Import Data` button will open the file explorer where you can select an ECG recording to analyse. This data should be 10 seconds, 500 Hz and between 1 and 12 leads (data shape: nr_leads x 5000). The toolbox is currently able to accept files with the following extensions: .txt, .csv, .hea, .xls, .xlsx, and .wav.. Clicking the `Process` button will display a results table with the results of individual quality assessment algorithms and the final consensus. 

### Downloading and running Toolbox
The `requirements_toolbox.txt` file in the toolbox folder contains all the packages needed to run the toolbox. 
`$ pip install -r requirements_toolbox.txt` should download all the packages needed to run the toolbox. 

## CNN-LSTM

### Downloading and running CNN-LSTM
The CNN-LSTM runs in python 3.8.11
The `requirements_cnn_lstm.txt` file contains all the packages needed to run the CNN-LSTM notebooks. 
`$ pip install -r requirements_cnn_lstm.txt` should download all the packages needed to run the CNN-LSTM notebooks. 

