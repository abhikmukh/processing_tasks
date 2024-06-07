import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import  scipy.ndimage as ndimage


def create_data_frame(file_path: str) -> pd.DataFrame:
    """
    Function to read all csv files in a directory and concatenate them into a single dataframe
    :param file_path:
    :return: DataFrame
    """
    data_files = glob.glob(os.path.join(file_path, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in data_files], axis=0, ignore_index=True)
    return df


class CleanData:
    """
    Class to clean data by interpolating missing values and removing outliers
    """
    @staticmethod
    def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to interpolate missing values in a dataframe
        :param df:
        :return: DataFrame
        """
        df = df.interpolate(option="spline", order=3)
        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to remove outliers using medfilt median filter from a scipy library
        :param df:
        :return:
        """
        df = df.apply(signal.medfilt)
        return df

    @staticmethod
    def apply_median_filter(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """
        Function to apply median filter to a dataframe
        :param df:
        :param window_size:
        :return:
        """
        df = df.apply(lambda x: ndimage.median_filter(x, window_size))
        return df


class SignalProcessing:
    """
    Class to process signal data by applying low pass filter and normalising the signal
    """
    @staticmethod
    def low_pass_filter(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to apply low pass iirfilter to a dataframe
        :param df:
        :return:
        """
        b, a = signal.iirfilter(4, Wn=0.5, btype="lowpass", ftype="butter")
        df = df.apply(lambda x: signal.filtfilt(b, a, x))
        return df

    @staticmethod
    def normalised_signal(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to normalise a dataframe
        :param df:
        :return:
        """
        df = df.apply(lambda x: (x - np.mean(x)) / np.std(x))
        return df


class FeatureGenerator:
    """
    Class to generate features from signal data
    """

    def __init__(self, df):
        self.df = df

    @staticmethod
    def create_features(signal_data: np.array) -> dict:
        """
        Function to create a dictionary of features from signal data
        :param signal_data:
        :return:
        """
        features = dict()
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['variance'] = np.var(signal_data)
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)

        # FFT (Frequency Domain)
        fft_vals = np.abs(fft(signal_data))
        features['fft_mean'] = np.mean(fft_vals)
        features['fft_std'] = np.std(fft_vals)
        features['fft_max'] = np.max(fft_vals)
        # Peaks
        peaks, _ = signal.find_peaks(signal_data)
        features['num_peaks'] = len(peaks)

        return features

    def generate_feature_df(self) -> pd.DataFrame:
        """
        Function to generate a dataframe of features from a dictionary of features
        :return:
        """
        features_series = self.df.apply(lambda x: self.create_features(x.values))
        features_df = pd.DataFrame.from_dict(features_series.tolist())
        return features_df


class FeatureExtraction:
    """
    Class to extract 3 features with the largest variance
    """
    def __init__(self, df):
        self.df = df

    @staticmethod
    def min_max_scaling(series: pd.Series) -> pd.Series:
        """
        Function to perform min-max scaling on a series
        :param series:
        :return:
        """
        return (series - series.min()) / (series.max() - series.min())

    def extract_features(self) -> list:
        """
        Function to extract 3 features with the largest variance
        :return:
        """
        scaled_df = self.df.apply(self.min_max_scaling)
        variance_dict = (np.var(scaled_df)).to_dict()
        sorted_var_dict = dict(sorted(variance_dict.items(), key=lambda x: x[1], reverse=True))
        return list(sorted_var_dict.keys())[:3]


class VisualiseData:
    """
    Class to visualise data using plots
    """

    @staticmethod
    def plot_df(df: pd.DataFrame) -> plt.Figure:
        """
        Function to plot a dataframe
        :param df:
        :return:
        """
        return df.plot(figsize=(20, 10), legend=False, title="Signal Data")

    @staticmethod
    def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
        """
        Function to plot a heatmap of a dataframe
        :param df:
        :return:
        """

        return sns.heatmap(df.corr())

    @staticmethod
    def plot_boxplot(df: pd.DataFrame) -> plt.Figure:
        """
        Function to plot a boxplot of a dataframe
        :param df:
        :return:
        """
        return df.plot(figsize=(20, 10), kind="box", legend=False)

    @staticmethod
    def plot_violinplot(df: pd.DataFrame) -> plt.Figure:
        """
        Function to plot a violin plot of a dataframe
        :param df:
        :return:
        """
        return sns.violinplot(data=df)


if __name__ == "__main__":

    # Read raw data
    raw_data_df = create_data_frame("signal_datasets/")
    print(f"There are {raw_data_df.shape[0]} signals in the dataset")

    raw_data_df = raw_data_df.T  # Transpose the dataframe to have signals as columns

    # Data cleaning
    data_cleaner = CleanData()
    print(f"Number of rows with missing or NAN values in the dataset are {raw_data_df.isnull().sum().sum()}")
    data_df = data_cleaner.interpolate_missing_values(raw_data_df)
    data_df = data_cleaner.remove_outliers(data_df)

    # Signal processing
    data_normaliser = SignalProcessing()
    data_df = data_normaliser.low_pass_filter(data_df)
    data_df = data_normaliser.normalised_signal(data_df)

    # Feature generation
    feature_generator = FeatureGenerator(data_df)
    feature_df = feature_generator.generate_feature_df()
    print(f"Features generated from the signal data are {list(feature_df.columns)}")

    # Feature extraction
    feature_extractor = FeatureExtraction(feature_df)
    top_3_features = feature_extractor.extract_features()
    print(f"Three features with largest variance are {top_3_features}")

    # Visualise data
    visualiser = VisualiseData()
    plot1 = visualiser.plot_df(raw_data_df)
    plot1.figure.savefig("images/raw_signal_data.png")
    plt.close()  # Close the plot to avoid overlapping plots

    plot2 = visualiser.plot_df(data_df)
    plot2.figure.savefig("images/cleaned_signal_data.png")
    plt.close()

    figure, ax = plt.subplots(figsize=(10, 10))
    plot3 = visualiser.plot_heatmap(feature_df)
    plot3.figure.savefig("images/heatmap.png")
    plt.close()

    plot4 = visualiser.plot_boxplot(feature_df)
    plot4.figure.savefig("images/features_boxplot.png")
