
# This repository contains the code for the projects `Signal processing` and `DNA sequence analysis`


## Signal processing
```
There are 884 signals in the dataset
Number of rows with missing or NAN values in the dataset are 100
Features generated from the signal data are ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'fft_mean', 'fft_std', 'fft_max', 'num_peaks']
Three features with largest variance are ['variance', 'skewness', 'std']

```
7 signal datasets are provided in the folder `signal_datasets` were processed using Pythons Pandas, Numpy and Scipy libraries. 
The following operations were performed on the datasets:
- Cleaning of signals
- Imputing missing values
- Smoothing of signals to remove outliers
- Normalization of signals
- Feature extraction

### The images of the signals before and after processing are shown below:
![Signal 1](images/raw_signal_data.png)
![Signal 2](images/cleaned_signal_data.png)

### The boxplot of the features extracted from the signals are shown below:
![Boxplot](images/features_boxplot.png)

### The correlation matrix of the features extracted from the signals are shown below:
![Correlation matrix](images/heatmap.png)
