import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time

from utils import char_tensor, random_training_set, time_since, CHUNK_LEN
from language_model import plot_loss, diff_temp, custom_train, train, generate
from model.model import LSTM

"""
DISCUSSIONS TO EACH TASK:

0. In a character-level model, the basic units are individual characters, while in 
   a word-level model, the basic units are entire words. This means that the first model
   predicts the next character based on the sequence of preceding characters, while in the
   second one, the model predicts the next word based on the sequence of preceding words 
   (similar to an n-gram model). This implies some practical differences, for example:

    (a) In the first case, the vocabulary size is typically much smaller since it includes 
    all unique characters in the dataset, instead of unique words. Of course, this aspect also
    depends on the corpus we're working on. However, we can assume that the number of unique
    characters will always be smaller than the number of unique words, as the second ones are
    a combination of the first ones.

    (b) Character-level models are more robust to rare words, misspellings, and variations in 
    language. They can handle any combination of characters, even those not seen during training.
    However, word-level models capture semantic meaning at a higher level, making them more 
    interpretable and potentially better at capturing context.

    (c) Character-level models may require more data to learn effective representations due to 
    their inability to represent semantic meaning as good as word-level models. They also may 
    take longer to train due to the increased complexity.

    (d) Character-level models can capture language variations, dialects, and even stylistic 
    elements more effectively since they are not constrained by a fixed vocabulary.

   In summary, character-level models offer more flexibility at the cost of increased complexity, 
   making them suitable for tasks where fine-grained details matter. Word-level models, on the 
   other hand, provide a more structured representation of language with potentially simpler 
   models. The choice depends on the specific requirements of the task at hand.

3. The use of the unidecode package to convert potential Unicode characters into plain ASCII may 
serve several purposes: 
    -character normalization (handling diacritics, accents or other special characters)
    -ensuring consistent encoding and compatibility with existing libraries
    -cleaning and standarizing data
    -enhacing readability
    
   While the use of the unidecode package to convert Unicode characters into plain ASCII offers 
   practical benefits in terms of consistency and compatibility, it introduces limitations, 
   particularly when dealing with languages beyond English. ASCII was originally designed to 
   represent characters in the American English language, and its use in the context of natural 
   language processing (NLP) may inadvertently lead to the standardization of English in NLP 
   applications.

   This standardization can be problematic for several reasons. Firstly, it assumes that the 
   linguistic characteristics and structures of other languages, such as Spanish or German, 
   can be adequately represented using the ASCII character set. As a result, the conversion 
   process may lead to the loss of linguistic nuances and essential elements that are crucial 
   for accurate representation.

   Furthermore, this standardization has implications for the broader applicability of NLP models, 
   especially those built on frameworks like PyTorch or TensorFlow. These frameworks are designed 
   with assumptions about the characteristics of the input data, and when the data is forcibly 
   standardized to ASCII, it may impact the models' performance and generalization capabilities, 
   particularly in multilingual or cross-lingual scenarios.

   In my opinion, we should consider utilizing Unicode encoding, which supports a vast range of 
   characters from various languages worldwide. Unicode includes characters with diacritics, 
   accents, and symbols, making it more suitable for multilingual applications.

4.1. See "./results/default.log". The --default_train flag, triggers a standard training process with 
   default hyperparameters of an LSTM (Long Short-Term Memory) model for character-level language 
   generation. The defined parameters will be used to initialize the model with the necesarry information, 
   i.e. input dimension, hidden dimension, number of layers and output dimensions; and will also serve 
   to specifiy the exact training process. An Adam optimizer is also employed.
   
   After initializing the model, the script enters a training loop that runs for a predefined number of 
   epochs. In each epoch, the train function is called with the LSTM model, Adam optimizer, and a random 
   training set. Training loss is computed and accumulated. 
   
   If the current epoch is a multiple of print_every, training progress is printed, including time elapsed, 
   epoch number, percentage completion, and the current loss. A sample text is also generated using the 
   generate function and printed. If the current epoch is a multiple of plot_every, the average loss over 
   the plotting interval is recorded. This information could be used to generate a plot of the 
   loss and analyze the model.

4.2. See "./results/custom.log". My approach was to test the following hyperparameters and compare their 
results between each other and with the default settings:

    (a) {'n_epochs': 1500, 'hidden_size': 32, 'n_layers': 1, 'lr': 0.1, 'temperature': 1}
    High learning rate (lr), low hidden size (hidden_size), high temperature and a single layer.

    (b) {'n_epochs': 3000, 'hidden_size': 64, 'n_layers': 1, 'lr': 0.01, 'temperature': 0.5}
    Moderate learning rate, moderate hidden size, moderate temperature and a single layer.

    (c) {'n_epochs': 3000, 'hidden_size': 256, 'n_layers': 3, 'lr': 0.002, 'temperature': 0.9}
    Low learning rate, substantial hidden size, high temperature and multiple layers.

    (d) {'n_epochs': 3000, 'hidden_size': 128, 'n_layers': 2, 'lr': 0.005, 'temperature': 1}
    High number of epochs, high hidden size and temperature, multiple layers and a low learning rate.

    The purpose of this testing was to compare the different performances (losses) in a variaty of 
    hyperparemeters' combination. This way I could identify how sensitive the model is to different 
    hyperparameters and which parameters have a significant impact on performance. Here are some 
    observations of each test:

    (a) Faster convergence, but there's a risk of overshooting. Training was less stable. There was also
    limited capacity to capture complex patterns, potentially resulting in underfitting. Increased randomness 
    in generated text, leading to more diverse but less focused outputs. 

    (b) Balanced convergence speed and stability. Moderately diverse and focused generated text. 
    Captured basic patterns but struggled with complexity.

    (c) Slower convergence. High capacity to capture complex patterns. Moderate diversity in generated text, 
    with some focus. Enhanced ability to capture hierarchical dependencies and complex patterns.

    (d) Good but slower convergence, but risk of overfitting. Good capacity to capture complex patterns. 
    Increased randomness in generated text. Improved ability to capture hierarchical dependencies.

    Considering this results, I propose following considerations when selecting hyperparameters:
    (a) A smaller learning rate converges slower, but generates a more stable learning. 
    (b) An increased hidden size improves the capacity to capture complex patterns.
    (c) A moderate temperature is less diversed, but more focused.
    (d) The more layers, the better. 
    (e) A high number of epochs helps to a better convergence, but can end up in overfitting. This is the 
    most difficult part to choose, in my opinion.

4.3. See "./results/output.jpg". Please consider that the number of epochs should be multiplied by 10, e.g.
    300 should be 3000.

4.4. See "./results/diff.log". When increasing the temperature value we observe changes in the diversity 
    and randomness of the generated text. With low temperatures, we got more deterministic and focused 
    text. With high temperatures, we got more diverse but less focused output. Intuitively, I did not 
    understand why a bigger temperature would end up in more randomized texts. However, according to 
    Hinton et al (2015), this hyperparameter is used to scale the logits before applying softmax. When 
    the temperature is 1, we compute the softmax directly on the logits. When the temperature is smaller, 
    the softmax ends up being higher needing less input to activate the layer, but also making it less 
    likely to sample from candidates with smaller numbers. This would explain why a higher temperature 
    ends up in a more randomized answer.

    There are some risks involving automatic text generation:

    (a) Biases and Unintended Outputs:
    Language models can inadvertently learn and reproduce biases present in training data, leading to 
    potentially biased or sensitive outputs.

    (b) Ethical Concerns:
    Generated text might contain inappropriate or harmful content, posing ethical concerns when used 
    without proper oversight.

    (c) Misinformation and Manipulation:
    There's a risk of generating text that spreads misinformation or manipulates public opinion if 
    the model has not been rigorously trained and validated.

    As these risks are present all the time, everyone involved in the production and use of automatic
    text generation is responsible for the final product. For example, the model builder is responsible
    for training the model and selecting hyperparameters, thus making him responsible for ensuring 
    ethical considerations during development. The script writer is responsible for selecting the corpus
    and how the model is utilized, making him/her responsible for the possible outputs of its implementation.
    Finally, the end user is ultimately responsible for the use and interpretation of generated outputs and 
    must exercise caution and critical judgment. In the best case-scenario, there is an open and 
    always available communication between the three instances.

    Some cautions and best practices are:

    (a) Bias Mitigation:
    Regularly evaluate and mitigate biases in training data. Use diverse datasets and consider fairness in 
    model training. In best case-scenario, also include minorities or unrepresented groups in the testing 
    phase.
    
    (b) Ethical Guidelines:
    Establish and adhere to ethical guidelines for content generation. Avoid generating harmful or 
    inappropriate content.
    
    (c) Human Oversight:
    Implement human oversight to review and filter generated content before public release.
    
    (d) Transparency:
    Be transparent about the limitations and potential biases of the model. Clearly communicate that the 
    generated text is machine-generated.
    
    (e) Legal Compliance:
    Ensure compliance with legal regulations, especially in sensitive domains such as healthcare or finance.
"""


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
            {'n_epochs': 1500, 'hidden_size': 32,
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
