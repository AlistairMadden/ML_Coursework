import csv
import cv2
import os
import numpy as np
import math
import random


def combine():
    ########### Load Data Set

    path_to_data = "./data/HAPT-data-set-DU"

    attribute_list = []
    label_list = []
    id_list = []

    # load training data
    reader = csv.reader(open(os.path.join(path_to_data, "Train/x_train.txt"), "rt", encoding='ascii'), delimiter=' ')
    for row in reader:
        # attributes in columns 0-561
        attribute_list.append(list(row[i] for i in (range(0, 561))))

    reader = csv.reader(open(os.path.join(path_to_data, "Train/y_train.txt"), "rt", encoding='ascii'), delimiter=' ')
    for row in reader:
        # attributes in column 1
        label_list.append(row[0])

    reader = csv.reader(open(os.path.join(path_to_data, "Train/subject_id_train.txt"), "rt", encoding='ascii'),
                        delimiter=' ')
    for row in reader:
        id_list.append(row[0])

    # load test data
    reader = csv.reader(open(os.path.join(path_to_data, "Test/x_test.txt"), "rt", encoding='ascii'), delimiter=' ')
    for row in reader:
        # attributes in columns 0-561
        attribute_list.append(list(row[i] for i in (range(0, 561))))

    reader = csv.reader(open(os.path.join(path_to_data, "Test/y_test.txt"), "rt", encoding='ascii'), delimiter=' ')
    for row in reader:
        # attributes in column 1
        label_list.append(row[0])

    reader = csv.reader(open(os.path.join(path_to_data, "Test/subject_id_test.txt"), "rt", encoding='ascii'),
                        delimiter=' ')
    for row in reader:
        id_list.append(row[0])

    # append labels and ids
    for i in range(len(attribute_list)):
        attribute_list[i].append(label_list[i])
        attribute_list[i].append(id_list[i])

    return attribute_list


if __name__ == "__main__":
    entry_list = combine()

    for k in range(0, 10):

        random_subjects = random.sample(range(1, 31), 20)

        attribute_training_list = []
        attribute_testing_list = []
        labels_training_list = []
        labels_testing_list = []
        id_training_list = []
        id_testing_list = []

        for entry in entry_list:
            # if in training set, else in test set
            if int(entry[-1]) in random_subjects:
                attribute_training_list.append(list(entry[i] for i in (range(0, 561))))
                labels_training_list.append([entry[-2]])
                id_training_list.append([entry[-1]])
            else:
                attribute_testing_list.append(list(entry[i] for i in (range(0, 561))))
                labels_testing_list.append([entry[-2]])
                id_testing_list.append([entry[-1]])

        writer = csv.writer(open("attributes_test_grouped" + str(k) + ".txt", "wt", encoding='ascii', newline=''),
                            delimiter=' ')
        writer.writerows(attribute_testing_list)

        writer = csv.writer(open("attributes_train_grouped" + str(k) + ".txt", "wt", encoding='ascii', newline=''),
                            delimiter=' ')
        writer.writerows(attribute_training_list)

        writer = csv.writer(open("labels_test_grouped" + str(k) + ".txt", "wt", encoding='ascii', newline=''),
                            delimiter=' ')
        writer.writerows(labels_testing_list)

        writer = csv.writer(open("labels_train_grouped" + str(k) + ".txt", "wt", encoding='ascii', newline=''),
                            delimiter=' ')
        writer.writerows(labels_training_list)

        writer = csv.writer(open("ids_test_grouped" + str(k) + ".txt", "wt", encoding='ascii', newline=''),
                            delimiter=' ')
        writer.writerows(id_testing_list)

        writer = csv.writer(open("ids_train_grouped" + str(k) + ".txt", "wt", encoding='ascii', newline=''),
                            delimiter=' ')
        writer.writerows(id_training_list)
