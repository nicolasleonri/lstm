import unidecode
import string
import random
import re
import time
import math
import torch

from torch.autograd import Variable

CHUNK_LEN = 200
TRAIN_PATH = './data/dickens_train.txt'


def load_dataset(path):
    all_characters = string.printable
    n_characters = len(all_characters)

    #file = open(path, 'r').read()
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
        tensor[c] = all_characters.index(strings[c]) #why are characters indeced like this? e.g. 10, 11, 12, 13, etc.
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