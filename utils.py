import unidecode
import string
import random
import re
import time
import math
import torch
import matplotlib.pyplot as plt


from torch.autograd import Variable

CHUNK_LEN = 200
TRAIN_PATH = './data/dickens_train.txt'


def load_dataset(path):
    all_characters = string.printable
    file = unidecode.unidecode(open(path, 'r').read())
    return file


def random_chunk():
    '''
    Splits big string of data into chunks.
    '''
    file = load_dataset(TRAIN_PATH)
    start_index = random.randint(0, len(file) - CHUNK_LEN - 1)
    end_index = start_index + CHUNK_LEN + 1
    return file[start_index:end_index]


def char_tensor(strings):
    '''
    Each chunk of the training data needs to be turned into a sequence
    of numbers (of the lookups), specifically a LongTensor (used for integer values). 
    This is done by looping through the characters of the string and looking up 
    the index of each character.
    '''
    all_characters = string.printable
    tensor = torch.zeros(len(strings)).long()
    for c in range(len(strings)):
        # why are characters indeced like this? e.g. 10, 11, 12, 13, etc.
        tensor[c] = all_characters.index(strings[c])
    return Variable(tensor)


def random_training_set():
    '''
    Assembles a pair of input and target tensors for training from a random chunk.

    The inputs will be all characters up to the last, and the targets will be all 
    characters from the first. So if our chunk is "test" the inputs will correspond 
    to “tes” while the targets are “est”.
    '''
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


def time_since(since):
    """
    A helper to print the amount of time passed.
    """
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def plot_experiment_results(experiment_results, max_experiments_per_plot=10):
    """
    Plot experiment results from a dictionary.

    Args:
        experiment_results (dict): Dictionary where each key corresponds to a list of values.
        max_experiments_per_plot (int): Maximum number of experiments to include in each plot.

    Returns:
        None
    """
    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Experiment Results')

    # Plot each experiment
    for i, (experiment_name, results) in enumerate(experiment_results.items()):
        print(i, experiment_name, results)
        if i % max_experiments_per_plot == 0 and i > 0:
            # Start a new plot for every max_experiments_per_plot experiments
            plt.legend()
            plt.show()
            plt.figure(figsize=(10, 6))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss vs. Epoch for Different Experiments')

        plt.plot(range(1, len(results) + 1), results, label=experiment_name)

    # Add legend and show the final plot
    plt.legend(loc='upper left')
    plt.show()
