import torch
import torch.nn as nn


# Here is a pseudocode to help with your LSTM implementation. 
# You can add new methods and/or change the signature (i.e., the input parameters) of the methods.
class LSTM(nn.Module):
    def __init__(self):
        """Think about which (hyper-)parameters your model needs; i.e., parameters that determine the
        exact shape (as opposed to the architecture) of the model. There's an embedding layer, which needs 
        to know how many elements it needs to embed, and into vectors of what size. There's a recurrent layer,
        which needs to know the size of its input (coming from the embedding layer). PyTorch also makes
        it easy to create a stack of such layers in one command; the size of the stack can be given
        here. Finally, the output of the recurrent layer(s) needs to be projected again into a vector
        of a specified size."""
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE
        ##########################################################################
        pass

    def forward(self):
        """Your implementation should accept input character, hidden and cell state,
        and output the next character distribution and the updated hidden and cell state."""
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE
        ##########################################################################
        pass

    def init_hidden(self):
        """Finally, you need to initialize the (actual) parameters of the model (the weight
        tensors) with the correct shapes."""
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE
        ##########################################################################
        pass