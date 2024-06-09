import random
import pandas as pd
import numpy as np


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


def generate_dna_sequence(length, gc_content):
    gc_bases = int(length * gc_content)
    at_bases = length - gc_bases

    # Create a list with the correct number of 'G's and 'C's
    gc_list = ['G'] * (gc_bases // 2) + ['C'] * (gc_bases // 2)
    # Create a list with the correct number of 'A's and 'T's
    at_list = ['A'] * (at_bases // 2) + ['T'] * (at_bases // 2)

    # Combine the lists
    dna_list = gc_list + at_list

    # Shuffle the list to create a random sequence
    random.shuffle(dna_list)

    # Join the list into a string
    dna_sequence = ''.join(dna_list)

    return (dna_sequence)


def generate_complimentary_sequence(my_dna, length):
    replacement1 = my_dna.replace('A', 't')
    replacement2 = replacement1.replace('T', 'a')
    replacement3 = replacement2.replace('C', 'g')
    replacement4 = replacement3.replace('G', 'c')
    compl = replacement4.upper()
    return compl[:length]

def generate_random_numbers(N, n):
    random_numbers = [random.randint(0, n) for _ in range(N)]
    return random_numbers

# Example usage


# Set parameters
if __name__ == "__main__":
    length = 100
    gc_content = 0.5

    # Generate a random DNA sequence
    dna_sequence = generate_dna_sequence(length, gc_content)
    print(dna_sequence)
    N = 100  # Number of random numbers to generate
    n = 100  # Upper limit for the random numbers

    random_numbers = generate_random_numbers(N, n)
    for num in random_numbers:
        print(generate_complimentary_sequence(dna_sequence, num))


