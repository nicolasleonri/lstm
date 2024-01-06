import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_embeddings, hidden_size, embedding_size, n_layers):
        ############################ STUDENT SOLUTION ############################
        """
        LSTM model for character generation.

        Args:
            num_embeddings (int): The number of characters to embed.
            hidden_size (int): The size of the RNN's hidden state.
            embedding_size (int): The size of vectors (of characters to embed).
            n_layers (int): Number of layers.
        """
        super(LSTM, self).__init__()

        # Hyperparameters
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        # Layers
        self.layer_embedding = nn.Embedding(
            self.num_embeddings, self.embedding_size)
        self.layer_lstm = nn.LSTM(
            self.embedding_size, self.hidden_size, self.n_layers)
        # Project to output size
        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.embedding_size)
        ##########################################################################

    def forward(self, input, hidden=None, cell=None):
        ############################ STUDENT SOLUTION ############################
        """
        Forward pass through the LSTM model.

        Args:
            input (torch.Tensor): Input tensor.
            hidden (torch.Tensor): Initial hidden state for the RNN.

        Returns:
            torch.Tensor: Output tensor.
            tuple: Tuple containing the updated hidden and cell states.
        """

        if hidden is None and cell is None:
            # Initialize hidden state and cell state if not provided
            hidden, cell = self.init_hidden()

        # Embedding layer
        embedded = self.layer_embedding(input)

        # LSTM layer
        # should be: output, (hidden, cell) = self.lstm(embedded.view(1, 1, -1), (hidden, cell))
        output, (hidden, cell) = self.layer_lstm(
            embedded.view(1, 1, -1), hidden)

        # Decoder layer
        decoded_output = self.fc(output.view(1, -1))

        return decoded_output, (hidden, cell)
        ##########################################################################

    def init_hidden(self):
        ############################ STUDENT SOLUTION ############################
        """
        Initialize the hidden and cell states.

        Returns:
            tuple: Tuple containing the initialized hidden and cell states.
        """
        # Initialize hidden and cell state with zero vectors
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        cell = torch.zeros(self.n_layers, 1, self.hidden_size)

        return hidden, cell
        ##########################################################################
