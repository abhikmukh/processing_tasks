import pandas as pd
import numpy as np


def gc_content(dna_sequence: str) -> float:
    """
    Function to calculate the GC content of a DNA sequence
    :param dna_sequence:
    :return: float
    """
    gc_count = dna_sequence.upper().count("G") + dna_sequence.upper().count("C")
    gc_content = gc_count / len(dna_sequence)
    return gc_content


def reverse_complement(dna_sequence: str) -> str:
    """
    Function to generate the reverse complement of a DNA sequence
    :param dna_sequence:
    :return: str
    """
    complement_dict = {"A": "T", "T": "A", "C": "G", "G": "C"}
    reverse_complement_seq = "".join([complement_dict[base] for base in dna_sequence[::-1]])
    return reverse_complement_seq


def introduce_error(dna_sequence: str, error_rate: float) -> str:
    """
    Function to introduce errors into a DNA sequence
    :param dna_sequence:
    :param error_rate:
    :return: str
    """
    error_sequence = ""
    for base in dna_sequence:
        if np.random.uniform() < error_rate:
            error_sequence += np.random.choice([x for x in "ATCG" if x != base])
        else:
            error_sequence += base
    return error_sequence


def generate_random_sequence(length: int) -> str:

    """
    Function to generate a random DNA sequence of a given length
    :param length:
    :return: str
    """
    return "".join(np.random.choice(list("ATCG"), length))


def generate_random_sequences(num_sequences: int, min_length: int, max_length: int) -> list:
    """
    Function to generate a list of random DNA sequences
    :param num_sequences:
    :param min_length:
    :param max_length:
    :return: list
    """
    return [generate_random_sequence(np.random.randint(min_length, max_length)) for _ in range(num_sequences)]