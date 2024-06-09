import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Bio.Seq import MutableSeq
from Bio.SeqUtils import gc_fraction as gc
import seaborn as sns


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

    plot1 = sns.displot(sequence_length_list_per_error_rate, rug=True, kind="kde", label=True)
    plot1.figure.savefig('images/sequence_length_distribution.png')
    plt.close()

    # Plot the distribution of GC content per error rate

    plot2 = sns.displot(gc_content_list_per_error_rate, kind="kde", label=True)
    plot2.figure.savefig('images/gc_content_distribution.png')
    plt.close()

    # Plot the error position distribution per error type for 10% error rate

    for error_rate_dict in error_rate_error_positions_list:
        if float(0.1) in error_rate_dict.keys():
            for each_error_positions_dict in error_rate_dict.values():
                dis_plot_df = pd.DataFrame(each_error_positions_dict)
                dis_plot_df = (dis_plot_df.explode("deletion").explode("insertion").explode("substitution"))
                dis_plot_df = (dis_plot_df.fillna(0))
                plot2 = sns.displot(dis_plot_df, alpha=0.5, kind='kde', fill=True)
                plot2.set(xlabel="position")
                plot2.figure.savefig('images/error_distribution.png')



