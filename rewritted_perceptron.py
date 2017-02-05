"""
rewritted_perceptron

Usage:
  rewritted_perceptron TRAIN TEST [--size_train=int] [--size_test=int] [--iteration_number=int] [--averaged] [--mira] [-h]
  rewritted_perceptron TRAIN TEST [--size_train=int] [--size_test=int] [--iteration_number=int] [--averaged] [--mira] [-h]

Arguments:
    TRAIN                   filename train corpus
    TEST                    filename test corpus

Options:
  -h --help                 Show this screen.
  --size_train=int          Size of train corpus. -1 means all corpus [default: -1].
  --size_test=int           Size of train corpus. -1 means all corpus [default: -1].
  --iteration_number=int    Number of iterations [default: 1].
  --averaged                Run averaged Perceptron [default: False].
  --mira                    Run mira Perceptron [default: False]

"""

from docopt import docopt
import numpy as np
import sys
import random

class Perceptron:
    def __init__(self, feature_class_weights_matrix, averaged=False, mira=False):

        self.__feature_class_weights_matrix = feature_class_weights_matrix
        self.__n_classes = len(self.__feature_class_weights_matrix[0])
        # print(self.__n_classes, "classes.")
        self.__n_features = len(self.__feature_class_weights_matrix)

        # averaging
        self.__averaged = averaged
        self.__feature_class_weights_matrix_sum = [[0 for c in range(self.__n_classes)] for f in range(self.__n_features)]

        #mira
        self.__mira = mira

    def train(self, corpus, n_iter):
        set_training_tuple_class_features = corpus

        # iteration number
        i_epoch = 0
        # n = len(set_training_tuple_class_features)
        counter = 1
        list_training_tuple_class_features = list(set_training_tuple_class_features)
        while i_epoch <= n_iter:
            random.shuffle(list_training_tuple_class_features)
            for example in list_training_tuple_class_features:
                i_class_ref = example[0]
                tpl_i_features_vector = example[1]
                n_vector_features = len(tpl_i_features_vector)

                if not self.__mira:
                    argmax = self.evaluate(tpl_i_features_vector, n_vector_features)

                    if argmax != i_class_ref:
                        for i_feature in range(n_vector_features):
                            if tpl_i_features_vector[i_feature] != -1:
                                self.__feature_class_weights_matrix[tpl_i_features_vector[i_feature]][argmax] -= 1
                                self.__feature_class_weights_matrix[tpl_i_features_vector[i_feature]][i_class_ref] += 1

                                self.__feature_class_weights_matrix_sum[tpl_i_features_vector[i_feature]][argmax] -= \
                                    counter
                                self.__feature_class_weights_matrix_sum[tpl_i_features_vector[i_feature]][i_class_ref] += \
                                    counter
                else:
                    oracle_score = self.evaluate_class_score(tpl_i_features_vector, n_vector_features, i_class_ref)
                    # print("oracle", i_class_ref, oracle_score)
                    argmax, score_max = self.evaluate(tpl_i_features_vector, n_vector_features, get_score=True)

                    if argmax != i_class_ref:
                        step = 1 - (oracle_score - score_max) / (n_vector_features ** 2)
                        step = max((step, 0))
                        step = min((step, 1))

                        for i_feature in range(n_vector_features):
                            self.__feature_class_weights_matrix[tpl_i_features_vector[i_feature]][argmax] -= step
                            self.__feature_class_weights_matrix[tpl_i_features_vector[i_feature]][i_class_ref] += step

                            self.__feature_class_weights_matrix_sum[tpl_i_features_vector[i_feature]][argmax] -= \
                                counter
                            self.__feature_class_weights_matrix_sum[tpl_i_features_vector[i_feature]][i_class_ref] += \
                                counter
                counter += 1
            i_epoch += 1

        if self.__averaged:
            for i in range(self.__n_features):
                for j in range(self.__n_classes):
                    self.__feature_class_weights_matrix[i][j] -= \
                        1 / float(counter) * self.__feature_class_weights_matrix_sum[i][j]

    def evaluate(self, tpl_i_features_vector, n_vector_features, get_score=False):
        # initializing scores (we try to maximize the score)
        lst_classes_scores = [0 for _ in range(self.__n_classes)]

        for i_class in range(self.__n_classes):
            lst_classes_scores[i_class] = self.evaluate_class_score(tpl_i_features_vector, n_vector_features, i_class)


        argmax = 0
        max = lst_classes_scores[argmax]
        for i_class in range(self.__n_classes):
            if lst_classes_scores[i_class] > max:
                max = lst_classes_scores[i_class]
                argmax = i_class

        if get_score is True:
            return (argmax, max)
        else:
            return argmax

    def evaluate_class_score(self, tpl_i_features_vector, n_vector_features, i_class):
        score = 0
        for i_feature in range(n_vector_features):
            if tpl_i_features_vector[i_feature] != -1:
                try:
                    score += self.__feature_class_weights_matrix[tpl_i_features_vector[i_feature]][
                            i_class]
                except IndexError:
                    print("[WARNING] Feature %s not found in feature_class_weights_matrix. 0 value added instead."
                          % (tpl_i_features_vector[i_feature]))
                    score += 0
        return score


    def test(self, corpus):

        set_testing_tuple_class_features = corpus

        error_count = 0
        i = 0

        # example: (class, vector)
        list_testing_tuple_class_features = list(set_testing_tuple_class_features)
        random.shuffle(list_testing_tuple_class_features)
        for example in list_testing_tuple_class_features:
            i_class_ref = example[0]

            tpl_i_features_vector = example[1]
            n_features = len(tpl_i_features_vector)

            argmax = self.evaluate(tpl_i_features_vector, n_features)
            # print(i_class_ref, argmax)
            if argmax != i_class_ref:
                error_count += 1

            i += 1

        return (error_count / i) * 100

    def get_weight_matrix(self):
        return self.__feature_class_weights_matrix


def get_class_feature_matrix_from_corpus(corpus):
    feature_matrix = []
    classes = set()
    n_classes = 0
    features = set()
    n_features = 0
    for example in corpus:
        if example[0] not in classes:
            classes.add(example[0])
            n_classes += 1
        for f in example[1]:
            if f not in features:
                features.add(f)
                n_features += 1

    for f in range(max(features) + 1):
        feature_matrix.append([0 for _ in range(max(classes) + 1)])

    return feature_matrix


def get_set_corpus(filename, size=-1):
    set_tuple_class_features = set()
    with open(filename, 'r') as f:
        i = 0
        s_line = f.readline()
        while s_line != "" and (i < size or size == -1) :
            if s_line.strip() != "":
                # for each example
                lst_splitted_line = s_line.split("\t")
                # get the class
                i_class_ref = int(lst_splitted_line[0].strip())
                # get the features
                tpl_i_features_vector = tuple([int(elm.strip()) for elm in lst_splitted_line[1:]])

                set_tuple_class_features.add((i_class_ref, tpl_i_features_vector))

            i += 1
            s_line = f.readline()

    return set_tuple_class_features


if __name__ == '__main__':
    # filename_train = '/home/luc/Documents/projets/perceptron/macaon2_sources/maca_data2/fr/maca_trans_parser/train.cff'
    # filename_test = '/home/luc/Documents/projets/perceptron/macaon2_sources/maca_data2/fr/maca_trans_parser/test.cff'
    arguments = docopt(__doc__)
    # print(arguments)

    set_corpus_train = get_set_corpus(arguments["TRAIN"], size=int(arguments["--size_train"]))

    set_corpus_test = get_set_corpus(arguments["TEST"], size=int(arguments["--size_test"]))

    set_corpus_train_total = get_set_corpus(arguments["TRAIN"])
    initialized_matrix = get_class_feature_matrix_from_corpus(set_corpus_train_total)
    # initialized_matrix = get_class_feature_matrix_from_corpus(set_corpus_train)
    p = Perceptron(initialized_matrix, averaged=arguments['--averaged'], mira=arguments['--mira'])
    # p = Perceptron(initialized_matrix, mira=True)
    p.train(set_corpus_train, int(arguments["--iteration_number"]))

    error_rate = p.test(set_corpus_test)
    print(arguments["TRAIN"].split("/")[-1] + "," +
          arguments["TEST"].split("/")[-1] + "," +
          arguments["--size_train"] + "," +
          arguments["--size_test"] + "," +
          arguments["--iteration_number"] + "," +
          str(error_rate))

