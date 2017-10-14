import warnings
from asl_data import SinglesData
import numpy as np

def score_word(hmm_model, word_index, Xword, lengths, verbose=False):
    # with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        score = hmm_model.score(Xword, lengths)
        if verbose:
            print("scored for example {} with {} states".format(word_index, hmm_model.n_components))
        return score
    except:
        if verbose:
            print("failure SCORING for {} with {} states".format(word_index, hmm_model.n_components))
        return None

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered   by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for key_eg, (X_eg, lengths_eg) in test_set.get_all_Xlengths().items():
        bestModel = None
        bestScore = -np.inf
        scores = {}
        for word, model in models.items():
            score = score_word(model, key_eg, X_eg, lengths_eg, verbose=False)
            scores[word] = score
            if score is not None and score > bestScore:
                bestScore = score
                bestModel = model
                bestWord = word
        probabilities.append(scores)
        guesses.append(bestWord)
                
    return probabilities, guesses
