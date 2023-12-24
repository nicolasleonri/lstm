import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p], (hidden, cell)) 
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp, (hidden, cell))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def train(decoder, decoder_optimizer, inp, target):
    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[c], (hidden, cell))
        loss += criterion(output, target[c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN


def tuner(n_epochs=3000, print_every=100, plot_every=10, hidden_size=128, n_layers=2,
          lr=0.005, start_string='A', prediction_length=100, temperature=0.8):
        # YOUR CODE HERE
        #     TODO:
        #         1) Implement a `tuner` that wraps over the training process (i.e. part
        #            of code that is ran with `default_train` flag) where you can
        #            adjust the hyperparameters
        #         2) This tuner will be used for `custom_train`, `plot_loss`, and
        #            `diff_temp` functions, so it should also accomodate function needed by
        #            those function (e.g. returning trained model to compute BPC and
        #            losses for plotting purpose).

        ################################### STUDENT SOLUTION #######################
        return None
        ############################################################################

def plot_loss(lr_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models where X is len(lr_list),
    #         and plot the training loss of each model on the same graph.
    #         2) Don't forget to add an entry for each experiment to the legend of the graph.
    #         Each graph should contain no more than 10 experiments.
    ###################################### STUDENT SOLUTION ##########################
    pass
    ##################################################################################

def diff_temp(temp_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, try to generate strings by using different temperature
    #         from `temp_list`.
    #         2) In order to do this, create chunks from the test set (with 200 characters length)
    #         and take first 10 characters of a randomly chosen chunk as a priming string.
    #         3) What happen with the output when you increase or decrease the temperature?
    ################################ STUDENT SOLUTION ################################
    pass
    ##################################################################################

def custom_train(hyperparam_list):
    """
    Train model with X different set of hyperparameters, where X is 
    len(hyperparam_list).

    Args:
        hyperparam_list: list of dict of hyperparameter settings

    Returns:
        bpc_dict: dict of bpc score for each set of hyperparameters.
    """
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models with different
    #         set of hyperparameters and compute their BPC scores on the test set.

    ################################# STUDENT SOLUTION ##########################
    return None
    #############################################################################