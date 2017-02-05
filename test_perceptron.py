"""
Pour moyenné seulement:

    corpus complet (train et test):

        varier nombre iteration -> resultat

    corpus complet (train et train):

        varier nomber iteration -> resultat

    faire varier taille corpus train + test complet
"""
import multiprocessing
import os


def os_call(cmd_line):
    print(cmd_line)
    os.system(cmd_line)


def varying_iteration_command_lines(root_command_line, number_iteration_max):
    number_iterations = range(1, number_iteration_max + 1)
    l_command_line = []
    for n in number_iterations:
        l_command_line.append(root_command_line % n)
    return l_command_line

if __name__ == "__main__":

    command_lines = []

    corpus_train_filename = '/home/luc/Documents/projets/perceptron/macaon2_sources/maca_data2/fr/maca_trans_parser/train.cutoff.cff'
    corpus_test_filename = '/home/luc/Documents/projets/perceptron/macaon2_sources/maca_data2/fr/maca_trans_parser/test.cff'

    root_cmd_line = "/home/luc/virtualenvs/machinelearning/bin/python3 rewritted_perceptron.py"

    root_cmd_line_iteration = "--iteration_number=%s"
    root_cmd_line_tee = "| tee -a test_results/%s"

    root_cmd_line_averaged = "--averaged"
    root_cmd_line_mira = "--mira"

    # vanilla
    command_lines.extend(varying_iteration_command_lines(" ".join([root_cmd_line,
                                                                   corpus_train_filename,
                                                                   corpus_test_filename,
                                                                   root_cmd_line_iteration,
                                                                   root_cmd_line_tee % "vanilla_iteration_1-15.csv"]),
                                                         15))

    # averaged
    command_lines.extend(varying_iteration_command_lines(" ".join([root_cmd_line,
                                                                   corpus_train_filename,
                                                                   corpus_test_filename,
                                                                   root_cmd_line_iteration,
                                                                   root_cmd_line_averaged,
                                                                   root_cmd_line_tee % "averaged_iteration_1-15.csv"]),
                                                         15))

    # mira
    command_lines.extend(varying_iteration_command_lines(" ".join([root_cmd_line,
                                                                   corpus_train_filename,
                                                                   corpus_test_filename,
                                                                   root_cmd_line_iteration,
                                                                   root_cmd_line_mira,
                                                                   root_cmd_line_tee % "mira_iteration_1-15.csv"]),
                                                         15))

    # mira averaged
    command_lines.extend(varying_iteration_command_lines(" ".join([root_cmd_line,
                                                                   corpus_train_filename,
                                                                   corpus_test_filename,
                                                                   root_cmd_line_iteration,
                                                                   root_cmd_line_mira,
                                                                   root_cmd_line_averaged,
                                                                   root_cmd_line_tee % "mira_averaged_iteration_1-15.csv"]),
                                                         15))
    pool = multiprocessing.Pool(4)
    pool.map(os_call, command_lines)


