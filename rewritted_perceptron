import numpy as np

class Perceptron:
    def __init__(self, feature_class_weights_matrix, averaged=False):

        self.__feature_class_weights_matrix = feature_class_weights_matrix
        self.__n_classes = len(self.__feature_class_weights_matrix[0])
        self.__n_features = len(self.__feature_class_weights_matrix)

        self.__averaged = averaged
        self.__feature_class_weights_matrix_sum = [[0 for c in range(self.__n_classes)] for f in range(self.__n_features)]

    def train(self, filename, n_iter):
        print("Start training")
        # load training set in memory
        set_training_tuple_class_features = set()
        with open(filename, 'r') as f:
            for s_line in f:
                # for each example
                lst_splitted_line = s_line.split("\t")
                # get the class
                i_class_ref = int(lst_splitted_line[0])
                # get the features
                tpl_i_features_vector = tuple([int(elm) for elm in lst_splitted_line[1:]])

                set_training_tuple_class_features.add((i_class_ref, tpl_i_features_vector))

        # iteration number
        i_epoch = 0
        n = len(set_training_tuple_class_features)
        i = 0
        counter = 1
        print(n, "examples.")
        while i_epoch <= n_iter:
            print("Iteration n°", i_epoch)
            for example in set_training_tuple_class_features:
                if i > n:
                    break
                elif i % 9999 == 0:
                    print("\t", i + (i%10000))
                i_class_ref = example[0]
                tpl_i_features_vector = example[1]
                n_vector_features = len(tpl_i_features_vector)

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
                i += 1
                counter += 1
            i = 0
            i_epoch += 1

        if self.__averaged:
            for i in range(self.__n_features):
                for j in range(self.__n_classes):
                    self.__feature_class_weights_matrix[i][j] -= \
                        1 / float(counter) * self.__feature_class_weights_matrix_sum[i][j]

    def evaluate(self, tpl_i_features_vector, n_vector_features):
        # initializing scores (we try to maximize the score)
        lst_i_classes_scores = [0 for _ in range(self.__n_classes)]

        for i_feature in range(n_vector_features):
            for i_class in range(self.__n_classes):
                try:
                    lst_i_classes_scores[i_class] += self.__feature_class_weights_matrix[tpl_i_features_vector[i_feature]][
                        i_class]
                except IndexError:
                    print("[WARNING] Feature %s not found in feature_class_weights_matrix. 0 value added instead."
                          % (tpl_i_features_vector[i_feature]))
                    lst_i_classes_scores[i_class] += 0

        argmax = 0
        max = lst_i_classes_scores[argmax]
        for i_class in range(self.__n_classes):
            if lst_i_classes_scores[i_class] > max:
                max = lst_i_classes_scores[i_class]
                argmax = i_class

        return argmax

    def test(self, filename):
        print("Start testing")
        # load testing set in memory
        set_testing_tuple_class_features = set()
        with open(filename, 'r') as f:
            for s_line in f:
                # for each example
                lst_splitted_line = s_line.split("\t")
                # get the class
                i_class_ref = int(lst_splitted_line[0])
                # get the features
                tpl_i_features_vector = tuple([int(elm) for elm in lst_splitted_line[1:]])

                set_testing_tuple_class_features.add((i_class_ref, tpl_i_features_vector))

        error_count = 0
        n = len(set_testing_tuple_class_features)
        print(n, "examples.")
        i = 0
        for example in set_testing_tuple_class_features:
            if i % 9999 == 0:
                print("\t", i)
            i_class_ref = example[0]
            tpl_i_features_vector = example[1]
            n_features = len(tpl_i_features_vector)

            argmax = self.evaluate(tpl_i_features_vector, n_features)

            if argmax != i_class_ref:
                error_count += 1
            i += 1

        return (error_count / len(set_testing_tuple_class_features)) * 100

    def get_weight_matrix(self):
        return self.__feature_class_weights_matrix

def perceptron_avg(filename, feature_matrix, n_iter):
    new_perceptron = Perceptron()

def get_class_feature_matrix_from_cff(filename):
    feature_matrix = []
    classes = set()
    features = set()
    with open(filename, 'r') as f:
        for line in f:
            splitted_line = line.split("\t")
            classes.add(int(splitted_line[0]))
            for f in splitted_line[1:]:
                features.add(int(f))

    for f in range(max(features) + 1):
        feature_matrix.append([0 for _ in range(max(classes) + 1)])

    print(len(features), min(features), max(features))
    print(len(classes), min(classes), max(classes))

    return feature_matrix

if __name__ == '__main__':
    filename_train = '/home/luc/Documents/work/maca_data2/fr/maca_trans_parser/train.cff'
    filename_test = '/home/luc/Documents/work/maca_data2/fr/maca_trans_parser/test.cff'
    initialized_matrix = get_class_feature_matrix_from_cff(filename_train)
    p = Perceptron(initialized_matrix)
    p.train(filename_train, 5)
    print(p.test(filename_test), "% d'erreurs.(normal)")

    p = Perceptron(initialized_matrix, averaged=True)
    p.train(filename_train, 5)
    print(p.test(filename_test), "% d'erreurs. (averaged)")
