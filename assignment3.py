import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time

from utils import char_tensor, random_training_set, time_since, CHUNK_LEN
from language_model import plot_loss, diff_temp, custom_train, train, generate
from model.model import LSTM


def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM'
    )

    parser.add_argument(
        '--default_train', dest='default_train',
        help='Train LSTM with default hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--custom_train', dest='custom_train',
        help='Train LSTM while tuning hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--plot_loss', dest='plot_loss',
        help='Plot losses chart with different learning rates',
        action='store_true'
    )

    parser.add_argument(
        '--diff_temp', dest='diff_temp',
        help='Generate strings by using different temperature',
        action='store_true'
    )

    args = parser.parse_args()

    all_characters = string.printable
    n_characters = len(all_characters)

    if args.default_train:
        n_epochs = 3000
        print_every = 100
        plot_every = 10
        hidden_size = 128
        n_layers = 2

        lr = 0.005
        decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

        start = time.time()
        all_losses = []
        loss_avg = 0

        for epoch in range(1, n_epochs+1):
            loss = train(decoder, decoder_optimizer, *random_training_set())
            loss_avg += loss

            if epoch % print_every == 0:
                print('[{} ({} {}%) {:.4f}]'.format(time_since(
                    start), epoch, epoch/n_epochs * 100, loss))
                print(generate(decoder, 'A', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

    if args.custom_train:
        ####################### STUDENT SOLUTION ###############################
        hyperparam_list = [
            {'n_epochs': 100, 'hidden_size': 32,
                'n_layers': 1, 'lr': 0.1, 'temperature': 1},
            {'hidden_size': 64, 'n_layers': 1, 'lr': 0.01, 'temperature': 0.5},
            {'hidden_size': 256, 'n_layers': 3, 'lr': 0.002, 'temperature': 0.9},
        ]
        ########################################################################
        bpc = custom_train(hyperparam_list)

        for keys, values in bpc.items():
            print("BPC {}: {}".format(keys, values))

    if args.plot_loss:
        ######################### STUDENT SOLUTION #############################
        lr_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.0005, 0.0001]
        ########################################################################
        plot_loss(lr_list)

    if args.diff_temp:
        ########################### STUDENT SOLUTION ###########################
        temp_list = [0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
        ########################################################################
        diff_temp(temp_list)


if __name__ == "__main__":
    main()
