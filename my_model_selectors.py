import math
import statistics
import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_features = len(list(all_word_Xlengths.values())[0][0][0])

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
	



class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def train_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

            score = hmm_model.score(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model, score
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None, None
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        bestBIC = np.inf
        bestModel = None
        for n in range(self.min_n_components, self.max_n_components+1):
            model, score = self.train_model(n)
            #print(score)
            #print(self.n_features)
            if score is not None:
                p = n*n + 2*n*self.n_features - 1
                BIC = -2 * score + p * np.log(n)
                if BIC < bestBIC:
                    bestBIC = BIC
                    bestModel = model

        
        return bestModel
            
            


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def train_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

            score = hmm_model.score(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model, score
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None, None
            
    def score_word(self, num_states, hmm_model, word, X_word, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            score = hmm_model.score(X_word, lengths)
            if self.verbose:
                print("scored for {} with {} states".format(word, num_states))
            return score
        except:
            if self.verbose:
                print("failure SCORING for {} with {} states".format(word, num_states))
            return None

    def select(self):
        #print(self.this_word)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores
        bestDIC = -np.inf
        best_num_components = None
        for n in range(self.min_n_components, self.max_n_components+1): 
            hmmModel, thisScore = self.train_model(n)
            if thisScore is not None:
                wordScore = {}
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        score = self.score_word(n, hmmModel, word, X, lengths)
                        if score is not None:
                            wordScore[word] = score
                DIC = (thisScore - 
                       np.mean([wordScore[word] for word in wordScore.keys()]))
                #print('DIC for {} with n = {} is {}'.format(self.this_word, n, DIC))
                if DIC > bestDIC:
                    bestDIC = DIC 
                    best_num_components = n 
        #print('best num components is {}'.format(best_num_components))
        bestModel,_ = self.train_model(best_num_components)
        if bestModel is not None:
            print('returning model with {} components'.format(bestModel.n_components))
        else:
            print('Model cannot be trained for {}'.format(self.this_word))
        return bestModel




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def train_model(self, num_states, X_train, lengths_train, X_test, lengths_test):
        #print('enter')
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
            #print(hmm_model)
            #print('one')
            logL = hmm_model.score(X_test, lengths_test)
            #print('two')
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            
            return logL
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
			
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        try:
            split_method = KFold(n_splits=min(3, len(self.sequences)))
        except Exception as e:
            print(str(e))
            return None
        #print('splits:', split_method.get_n_splits())
        scores = {}
        #print(self.this_word)
        for n in range(self.min_n_components, self.max_n_components+1):
            #print('n=',n)
            avgScore = 0
            count = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                #print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
                X_train, lengths_train = np.asarray(combine_sequences(cv_train_idx, self.sequences))
                X_test, lengths_test = np.asarray(combine_sequences(cv_test_idx, self.sequences))

                score = self.train_model(n, X_train, lengths_train, X_test, lengths_test)
                if score is not None:
                    avgScore += score
                    count += 1
            if count != 0: # at least one fold didn't fail 
                avgScore = avgScore/count
            else: # all the folds failed
                avgScore = -np.inf
            #print(n, avgScore)
            scores[n] = avgScore
            
        best_num_components = max(scores, key=scores.get)

        return self.base_model(best_num_components)
		

