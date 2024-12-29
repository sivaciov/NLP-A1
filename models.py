# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import nltk
from nltk.metrics.distance import edit_distance

from torch.sparse import log_softmax

from sentiment_data import *

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FFNN(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings):
        super(FFNN, self).__init__()

        self.word_embeddings = word_embeddings
        self.initialized_embedding_layer = word_embeddings.get_initialized_embedding_layer()
        self.indexer = word_embeddings.word_indexer
        # Extract the vocabulary from word_embeddings.word_indexer.objs_to_ints
        self.vocabulary = list(word_embeddings.word_indexer.objs_to_ints.keys())

        # create a prefix vocabulary
        prefix_vocab = {}
        for word in self.vocabulary:
            prefix = word[:3]
            if prefix not in prefix_vocab:
                prefix_vocab[prefix] = []
            prefix_vocab[prefix].append(word)
        self.prefix_vocab = prefix_vocab

        self.V = nn.Linear(word_embeddings.get_embedding_length(), 100)
        self.g = nn.ReLU()
        self.W = nn.Linear(100, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def words_to_indices(self, words: List[str]) -> torch.Tensor:
        indices = []
        for word in words:
            if self.indexer.contains(word):
                indices.append(self.indexer.index_of(word))
            else:
                indices.append(self.indexer.index_of("UNK"))
        return torch.tensor(indices)

    def words_to_indices_with_typos(self, words: List[str]) -> torch.Tensor:
        indices = []
        for word in words:
            if self.indexer.contains(word):
                indices.append(self.indexer.index_of(word))
            else:
                # Check for prefixes
                prefix = word[:3]
                candidates = self.prefix_vocab.get(prefix, [])

                if not candidates:
                    #print("No candidates found for prefix:", prefix)
                    indices.append(self.indexer.index_of("UNK"))
                    continue

                # Check for typos
                min_distance = 100
                closest_word = "UNK"
                for known_word in candidates:
                    distance = edit_distance(word, known_word)
                    if distance < min_distance:
                        min_distance = distance
                        closest_word = known_word
                if min_distance < 2:
                    #print("Closest word:", closest_word)
                    indices.append(self.indexer.index_of(closest_word))
                else:
                    #print("No close word found for:", word)
                    indices.append(self.indexer.index_of("UNK"))
        return torch.tensor(indices)


    def forward(self, words, has_typos=False):
        if has_typos:
            word_indices = self.words_to_indices_with_typos(words)
        else:
            word_indices = self.words_to_indices(words)


        embeddings = self.initialized_embedding_layer(word_indices.unsqueeze(0))
        #print("embeddings shape: ", embeddings.shape)
        #print(embeddings)

        average_embedding = embeddings.mean(dim=1)
        #print("average_embedding shape: ", average_embedding.shape)
        #print(average_embedding)

        hidden = self.V(average_embedding)
        #print("hidden shape: ", hidden.shape)
        #print(hidden)

        apply_g = self.g(hidden)
        #print("apply_g shape: ", apply_g.shape)
        #print(apply_g)

        output = self.W(hidden)
        #print("output shape: ", output.shape)
        #print(output)

        log_softmax = self.log_softmax(output)
        #print("log_softmax shape: ", log_softmax.shape)
        #print(log_softmax)

        return log_softmax

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, ffnn: FFNN):
        self.ffnn = ffnn

    def predict(self, ex_words: List[str], has_typos=False) -> int:
        log_probs = self.ffnn.forward(ex_words, has_typos)
        prediction = log_probs.argmax().item()
        return prediction


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    # set up custom training sets
    #my_train_exs = train_exs[:3000]
    my_train_exs = train_exs

    num_epochs = 100
    learning_rate = 0.0001
    ffnn = FFNN(word_embeddings)
    optimizer = optim.Adam(ffnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(num_epochs):
        ex_indices = [i for i in range(0, len(my_train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            words = my_train_exs[idx].words
            label = my_train_exs[idx].label

            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()

            log_probs = ffnn.forward(words)
            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = criterion(log_probs, torch.tensor([label]))
            total_loss += loss.item()
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

        # evaluate on on training set
        correct_train = 0
        for ex in my_train_exs:
            words = ex.words
            label = ex.label
            log_probs = ffnn.forward(words)
            prediction = log_probs.argmax().item()
            if prediction == label:
                correct_train += 1
            #print("; gold = " + repr(label) + "; pred = " + repr(prediction) + " with probs " + repr(log_probs))

        # evaluate on dev set
        correct_dev = 0
        for ex in dev_exs:
            words = ex.words
            label = ex.label
            log_probs = ffnn.forward(words)
            prediction = log_probs.argmax().item()
            if prediction == label:
                correct_dev += 1
            #print("; gold = " + repr(label) + "; pred = " + repr(prediction) + " with probs " + repr(log_probs))
        print("Total loss on epoch %i: %.4f" % (epoch, total_loss), " Train accuracy: ", correct_train / len(my_train_exs), " Dev accuracy: ", correct_dev / len(dev_exs))

        if correct_dev / len(dev_exs) > 0.78:
            break

    return NeuralSentimentClassifier(ffnn)

