import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import scipy.ndimage as ndimage
from sklearn.ensemble import IsolationForest
from Bio.Seq import MutableSeq
from Bio.SeqUtils import gc_fraction as gc
import random


############################################################################################################
# Functions and classes for task 1

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


class AnomalyDetection:
    """
    Class to detect anomalies in the signal data
    """
    def __init__(self, df):
        self.df = df

    @staticmethod
    def isolation_forest(signal_data) -> pd.Series:
        """
        Function to detect anomalies using Isolation Forest
        :param signal_data:
        :return:
        """
        model = IsolationForest(contamination=0.1)
        model.fit(signal_data)
        return model.predict(signal_data)

    def detect_anomalies(self) -> pd.Series:
        df_anomaly = self.df.apply(lambda x: AnomalyDetection.isolation_forest(x.values.reshape(-1, 1)))
        return df_anomaly

############################################################################################################
# Functions for task 2


def generate_dna_sequence(length: int, gc_content: float) -> str:
    """
    Function to generate a random DNA sequence with a given length and GC content
    :param length:
    :param gc_content:
    :return:
    """
    gc_bases = int(length * gc_content)
    at_bases = length - gc_bases

    gc_list = ['G'] * (gc_bases // 2) + ['C'] * (gc_bases // 2)
    at_list = ['A'] * (at_bases // 2) + ['T'] * (at_bases // 2)
    dna_list = gc_list + at_list
    random.shuffle(dna_list)
    dna_sequence = ''.join(dna_list)
    return dna_sequence


def generate_complimentary_sequence(dna_sequence: str, length: int) -> str:
    """
    Function to generate a complimentary sequence of a given DNA sequence
    :param dna_sequence:
    :param length:
    :return:
    """
    bio_seq = MutableSeq(dna_sequence)
    complimentary_dna_seq = bio_seq.complement()
    return complimentary_dna_seq[:length]


def generate_random_numbers(number_of_random_numbers: int, upper_limit:int) -> list:
    """
    Function to generate a list of random numbers with a upper limit of number
    :param number_of_random_numbers:
    :param upper_limit:
    :return:
    """
    random_numbers_list = [random.randint(0, upper_limit) for _ in range(number_of_random_numbers)]
    return random_numbers_list


def introduce_errors(dna_sequence: str, error_rate: float) -> tuple:
    """
    Function to introduce errors in a DNA sequence based on the error rate
    :param dna_sequence:
    :param error_rate:
    :return: tuple of the sequence with errors and the error positions
    """
    error_positions = {'deletion': [], 'insertion': [], 'substitution': []}
    error_types = ['deletion', 'insertion', 'substitution']
    bio_seq = MutableSeq(dna_sequence)
    sequence_length = len(bio_seq)

    num_errors = int(sequence_length * error_rate)

    sequence_info_dict = {}

    for _ in range(num_errors):
        error_type = random.choice(error_types)
        error_position = random.randint(0, sequence_length - 1)
        error_positions[error_type].append(error_position)

        if error_type == 'deletion':

            bio_seq.pop(error_position)
            sequence_length -= 1
        elif error_type == 'insertion':
            bio_seq.insert(error_position, random.choice('ACGT'))
            sequence_length += 1
        elif error_type == 'substitution':
            bio_seq[error_position] = random.choice('ACGT')

    return bio_seq, error_positions


if __name__ == "__main__":

    # Task 1: Signal Data Processing
    # Read raw data
    raw_data_df = create_data_frame("signal_datasets/")
    print(f"There are {raw_data_df.shape[0]} signals in the dataset")

    raw_data_df = raw_data_df.T  # Transpose the dataframe to have signals as columns

    # Data cleaning
    data_cleaner = CleanData()
    print(f"Number of rows with missing or NAN values in the dataset are {raw_data_df.isnull().sum().sum()}")
    data_df = data_cleaner.interpolate_missing_values(raw_data_df)

    # Anomaly detection and Remove outliers
    anomaly_detector = AnomalyDetection(data_df)
    # anomaly_df = anomaly_detector.detect_anomalies()  # Detect anomalies in the signal data
    data_df = data_cleaner.apply_median_filter(data_df, 5)  # Apply median filter to remove outliers

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

    # Task 2: DNA Sequence Analysis
    length_of_template = 100
    gc_content_of_template = 0.5

    # Generate a random DNA sequence
    template_dna_sequence = generate_dna_sequence(length=length_of_template, gc_content=gc_content_of_template)

    random_numbers = generate_random_numbers(number_of_random_numbers=100, upper_limit=100)

    # Generate the sequence libraries and create data for analysis
    error_rate_list = [0.02, 0.05, 0.1]
    error_rate_error_positions_list = []
    sequence_length_list_per_error_rate = {}
    gc_content_list_per_error_rate = {}

    for error_rate in error_rate_list:
        list_of_error_positions_dict = []
        sequence_set_dict = dict()
        sequence_length_list = []
        gc_content_list = []

        for number in random_numbers:
            complimentary_seq = generate_complimentary_sequence(dna_sequence=template_dna_sequence, length=number)
            sequence_with_errors, error_positions_dict = introduce_errors(dna_sequence=complimentary_seq,
                                                                          error_rate=error_rate)
            list_of_error_positions_dict.append(error_positions_dict)
            sequence_length_list.append(len(sequence_with_errors))
            gc_content_list.append(gc(sequence_with_errors))

            sequence_set_dict[error_rate] = list_of_error_positions_dict
        error_rate_error_positions_list.append(sequence_set_dict)
        sequence_length_list_per_error_rate[error_rate] = sequence_length_list
        gc_content_list_per_error_rate[error_rate] = gc_content_list

    # Plot the distributions of sequence length per error rates

    plot5 = sns.displot(sequence_length_list_per_error_rate, rug=True, kind="kde", label=True)
    plot5.set(xlabel="Sequence Length", ylabel="Density")
    plot5.figure.savefig('images/sequence_length_distribution.png')
    plt.close()

    # Plot the distribution of GC content per error rate

    plot6 = sns.displot(gc_content_list_per_error_rate, kind="kde", label=True)
    plot6.set(xlabel="GC Content", ylabel="Density")
    plot6.figure.savefig('images/gc_content_distribution.png')
    plt.close()

    # Plot the error position distribution per error type for 10% error rate

    for error_rate_dict in error_rate_error_positions_list:
        if float(0.1) in error_rate_dict.keys():
            for each_error_positions_dict in error_rate_dict.values():
                dis_plot_df = pd.DataFrame(each_error_positions_dict)
                dis_plot_df = (dis_plot_df.explode("deletion").explode("insertion").explode("substitution"))
                dis_plot_df = (dis_plot_df.fillna(0))
                plot7 = sns.displot(dis_plot_df, alpha=0.5, kind='kde', fill=True)
                plot7.set(xlabel="position", ylabel="Density")
                plot7.figure.savefig('images/error_distribution.png')
