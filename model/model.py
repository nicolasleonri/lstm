import torch
import torch.nn as nn


# Here is a pseudocode to help with your LSTM implementation. 
# You can add new methods and/or change the signature (i.e., the input parameters) of the methods.
class LSTM(nn.Module):
    def __init__(self, num_embeddings, hidden_size, embedding_size, n_layers, batch_first = False, padding_idx = 0):

        # Input:
        #decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)

        """Think about which (hyper-)parameters your model needs; i.e., parameters that determine the
        exact shape (as opposed to the architecture) of the model. There's an embedding layer, which needs 
        to know how many elements it needs to embed, and into vectors of what size. There's a recurrent layer,
        which needs to know the size of its input (coming from the embedding layer). PyTorch also makes
        it easy to create a stack of such layers in one command; the size of the stack can be given
        here. Finally, the output of the recurrent layer(s) needs to be projected again into a vector
        of a specified size."""

        '''
        1. one layer that maps the input character into its embedding, 
        2. one LSTM layer (which may itself have multiple layers) that operates on that embedding 
        and a hidden and cell state, 
        3. and a decoder layer that outputs the probability distribution.
        '''
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE

        """
        Args:
            num_embeddings (int): The number of characters to embed
            embedding_size (int): The size of vectors of characters to embed
            hidden_size (int): The size of the RNN's hidden state
            n_layers (int): Number of layers
        """

        # initialise hidden and cell state at t=0 with zero vectors and with the correct shapes
        super(LSTM, self).__init__()

        # Hyperparameters
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.padding_idx = padding_idx
        self.batch_first = batch_first

        # Layers
        self.layer_embedding = nn.Embedding(self.num_embeddings, self.embedding_size, padding_idx = self.padding_idx)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, batch_first = self.batch_first)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_size)  # Project to output size

        ##########################################################################
        pass

    def forward(self, input, hidden = None):
        """Your implementation should accept input character, hidden and cell state,
        and output the next character distribution and the updated hidden and cell state."""

        '''
        Args:
            input (torch.Tensor): an input data tensor.
                If self.batch_first: x_in.shape = (batch_size, seq_size, feat_size)
                Else: x_in.shape = (seq_size, batch_size, feat_size)
            initial_hidden (torch.Tensor): the initial hidden state for the RNN
        Returns:
            hiddens (torch.Tensor): The outputs of the RNN at each time step.
                If self.batch_first:
                hiddens.shape = (batch_size, seq_size, hidden_size)
            Else: hiddens.shape = (seq_size, batch_size, hidden_size)
        '''
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE

        if hidden is None:
            # Initialize hidden state if not provided
            batch_size = input.size(0) if self.batch_first else input.size(1)
            hidden = self.init_hidden(batch_size)

        # Embedding layer
        embedded = self.layer_embedding(input)

        # LSTM layer
        output, (hidden, cell) = self.lstm(embedded.view(1,1,-1), hidden)

        # Decoder layer
        decoded_output = self.fc(output.view(1, -1))

        return decoded_output, (hidden, cell)
        ##########################################################################

    def init_hidden(self):
        """Finally, you need to initialize the (actual) parameters of the model (the weight
        tensors) with the correct shapes."""
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE
        # Initialize hidden and cell state with zero vectors
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        cell = torch.zeros(self.n_layers,  1, self.hidden_size)

        return hidden, cell
        ##########################################################################