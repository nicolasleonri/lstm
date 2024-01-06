import torch
import string as gutenberg

from utils import char_tensor


def compute_bpc(model, string):
    """
    Given a model and a string of characters, compute bits per character
    (BPC) using that model.

    Args:
        model: RNN-based model (RNN, LSTM, GRU, etc.)
        string: string of characters

    Returns:
        BPC for that set of string.
    """
    ################# STUDENT SOLUTION ################################
    # Helper function to generate text using the provided model
    def generate(decoder, prime_str, predict_len, temperature=0.8):
        hidden, cell = decoder.init_hidden()
        prime_input = char_tensor(prime_str)
        predicted = prime_str
        all_characters = gutenberg.printable

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            _, (hidden, cell) = decoder(prime_input[p], (hidden, cell))
        inp = prime_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = decoder(inp, (hidden, cell))

            # Sample from the network as a multinomial distribution
            EPSILON = 1e-8  # Small epsilon value to prevent division by very small numbers
            output_dist = output.data.view(-1).div(temperature + EPSILON).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            # Add predicted character to string and use as next input
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            inp = char_tensor(predicted_char)

        return predicted

    # Extract a portion of the string for testing
    test_string = string[:len(string)//6]

    # Calculate the length to predict
    length_to_predict = len(string) - len(test_string)

    # Generate the remaining portion of the string
    predicted_string = generate(model, test_string, length_to_predict)

    # Calculate BPC
    EPSILON = 1e-8  # Small epsilon value to prevent division by very small numbers
    bpc = -torch.sum(torch.log2((torch.tensor(
        [ord(c) for c in predicted_string]) + EPSILON) / 128)) / len(predicted_string)

    return bpc.item()
    ###################################################################
