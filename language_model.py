import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt
import random

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN, plot_experiment_results
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
    """
    Wrapper function for training process with adjustable hyperparameters.

    Args:
        n_epochs (int): Number of epochs for training.
        print_every (int): Frequency of printing training information.
        plot_every (int): Frequency of plotting training loss.
        hidden_size (int): Size of the RNN's hidden state.
        n_layers (int): Number of layers in the RNN.
        lr (float): Learning rate for training.
        start_string (str): Initial input for generating text.
        prediction_length (int): Length of the generated text.
        temperature (float): Controls the randomness of the generated text.

    Returns:
        list: List of training losses for each epoch.
    """

    all_losses = []
    all_texts = []

    all_characters = string.printable
    n_characters = len(all_characters)

    # Initialize  model, optimizer, and other necessary components
    decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        current_loss = train(decoder, decoder_optimizer,
                             *random_training_set())

        if epoch % print_every == 0:
            print(f'Epoch {epoch}/{n_epochs} | Loss: {current_loss:.4f}')
            output_text = generate(decoder, start_string,
                                   prediction_length, temperature)
            all_texts.append(output_text)

        if epoch % plot_every == 0:
            all_losses.append(current_loss)

    return all_losses, all_texts

    ############################################################################


def plot_loss(lr_list):
    ###################################### STUDENT SOLUTION ##########################
    """
    Plot training loss for models with different learning rates.

    Args:
        lr_list (list): List of learning rates.

    Returns:
        dict: Dictionary containing the training loss for each learning rate.
    """
    all_losses = {}

    for lr in lr_list:
        print(f"\nTraining model with learning rate: {lr}")
        loss, _ = tuner(lr=lr)
        all_losses[lr] = loss
        print(
            f"Training completed for learning rate {lr}. Final loss: {loss[-1]}")
        print('\n')

    plot_experiment_results(all_losses)

    return all_losses
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
    """
    Generate strings with different temperatures and print the results.

    Args:
        temp_list (list): List of temperatures to experiment with.

    Returns:
        dict: Dictionary containing generated strings and priming strings for each temperature.
    """
    temp_priming = {}

    CHUNK_LEN = 200

    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())

    for i, temp in enumerate(temp_list):
        print(f'\nIteration {i+1}: Trying temperature {temp}')
        test_chunks = [string[i:i+CHUNK_LEN]
                       for i in range(0, len(string) - CHUNK_LEN + 1)]

        # Take the first 10 characters of a randomly chosen chunk as a priming string
        priming_string = str(random.choice(test_chunks)[:10])

        # Use the tuner function to generate strings with different temperatures
        _, all_texts = tuner(start_string=priming_string, temperature=temp)

        # Store the generated string in the dictionary
        temp_priming[temp] = (all_texts, priming_string)
        print(
            f"For temperature {temp} and Priming string '{priming_string}', we got this last answer:")
        print(all_texts[-1])
        print('\n')

    for temp, (answers, priming_string) in temp_priming.items():
        print(
            f"For temperature {temp} and Priming string '{priming_string}', we got the following answers:")
        for idx, val in enumerate(answers):
            print("Answer", idx+1, ":", val)
        print('\n')

    return temp_priming
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
