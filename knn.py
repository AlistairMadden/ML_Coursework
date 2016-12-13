#####################################################################

# Usage: Run after KXFV_data_generator.py has been run. Range of sample counts configurable from the main section of
# this file.

# Author : Alistair Madden, alistair.madden@durham.ac.uk
#          using significant portions of code produced by
#          Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import os
import numpy as np
import time


def knn(attribute_training_filename, label_training_filename, attribute_testing_filename, label_testing_filename,
        data_path="./", k_range=range(1, 2)):
    ########### Define classes

    # define mapping of classes
    classes = {'WALKING': 1, 'WALKING_UPSTAIRS': 2, 'WALKING_DOWNSTAIRS': 3, 'SITTING': 4, 'STANDING': 5, 'LAYING': 6,
               'STAND_TO_SIT': 7, 'SIT_TO_STAND': 8, 'SIT_TO_LIE': 9, 'LIE_TO_SIT': 10, 'STAND_TO_LIE': 11,
               'LIE_TO_STAND': 12}

    # and the inverse of the classes dictionary
    inv_classes = {v: k for k, v in classes.items()}

    start_load = time.clock()
    ########### Load Data Set

    # Training data - as currently split

    attribute_list = []
    label_list = []

    reader = csv.reader(open(os.path.join(data_path, attribute_training_filename), "rt", encoding='ascii'),
                        delimiter=' ')
    for row in reader:
        # attributes in columns 0-561
        attribute_list.append(list(row[i] for i in (range(0, 561))))

    reader = csv.reader(open(os.path.join(data_path, label_training_filename), "rt", encoding='ascii'), delimiter=' ')
    for row in reader:
        # attributes in column 1
        label_list.append(row[0])

    training_attributes = np.array(attribute_list).astype(np.float32)
    training_labels = np.array(label_list).astype(np.float32)

    # Testing data - as currently split

    attribute_list = []
    label_list = []

    reader = csv.reader(open(os.path.join(data_path, attribute_testing_filename), "rt", encoding='ascii'),
                        delimiter=' ')
    for row in reader:
        # attributes in columns 0-561
        attribute_list.append(list(row[i] for i in (range(0, 561))))

    reader = csv.reader(open(os.path.join(data_path, label_testing_filename), "rt", encoding='ascii'), delimiter=' ')
    for row in reader:
        # attributes in column 1
        label_list.append(row[0])

    testing_attributes = np.array(attribute_list).astype(np.float32)
    testing_labels = np.array(label_list).astype(np.float32)

    end_load = time.clock()

    print("File load time = " + str(end_load - start_load))

    start_train = time.clock()
    ############ Perform Training -- k-NN

    # define kNN object

    knn = cv2.ml.KNearest_create()

    # set to use BRUTE_FORCE neighbour search as KNEAREST_KDTREE seems to  break
    # on this data set (may not for others - http://code.opencv.org/issues/2661)

    knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE)

    # set default 3, can be changed at query time in predict() call

    knn.setDefaultK(k_range[-1])

    # set up classification, turning off regression

    knn.setIsClassifier(True)

    # perform training of k-NN

    knn.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels)

    end_train = time.clock()

    print("Training time = " + str(end_train - start_train))

    start_test = time.clock()
    ############ Perform Testing -- k-NN

    all_k_result = []

    # for every value of k we wish to test
    for k in k_range:

        # confusion matrix to store all results
        confusion_matrix = [[0 for x in range(12)] for y in range(12)]

        # for each testing example
        for i in range(0, len(testing_attributes[:, 0])):
            # perform k-NN prediction (i.e. classification)

            # (to get around some kind of OpenCV python interface bug, vertically stack the
            #  example with a second row of zeros of the same size and type which is ignored).

            sample = np.vstack((testing_attributes[i, :],
                                np.zeros(len(testing_attributes[i, :])).astype(np.float32)))

            # now do the prediction returning the result, results (ignored) and also the responses
            # + distances of each of the k nearest neighbours
            # N.B. k at classification time must be < maxK from earlier training

            _, results, neigh_response, distances = knn.findNearest(sample, k)

            confusion_matrix[(int(results[0]) - 1)][(int(testing_labels[i]) - 1)] += 1

        k_result = []

        # for every class
        for classification in range(12):

            # reset metrics
            tn = 0  # predicted not classification and actual is not classification
            fp = 0  # predicted as classification, but actual is different
            fn = 0  # predicted as not classification, but actual is classification

            # set classification name and true positive rate
            classification_name = inv_classes[classification + 1]
            tp = confusion_matrix[classification][classification]

            # for every row (predicted classification)
            for predicted in range(12):
                # for every column (actual classification)
                for actual in range(12):

                    # true negative
                    if predicted != classification and actual != classification:
                        tn += confusion_matrix[predicted][actual]

                    # false positive
                    if predicted == classification and actual != classification:
                        fp += confusion_matrix[predicted][actual]

                    # false negative
                    if predicted != classification and actual == classification:
                        fn += confusion_matrix[predicted][actual]

            k_result.append([classification_name, k, tp, tn, fp, fn])

        all_k_result.append(k_result)

    end_test = time.clock()

    print("Testing time = " + str(end_test - start_test))

    return all_k_result


if __name__ == "__main__":

    # holds a unique row for a given k and classification
    x_fold_validations = []

    k_range = range(1, 3)

    x_fold_validation_range = range(0, 10)

    # For every cross fold validation
    for i in x_fold_validation_range:

        # Get results for every class vs all the others
        x_fold_validation = knn("attributes_train" + str(i) + ".txt", "labels_train"+str(i)+".txt",
                                "attributes_test"+str(i)+".txt", "labels_test"+str(i)+".txt", "./", k_range)

        print()
        print("XFV = " + str(i + 1))
        print()

        x_fold_validations.append(x_fold_validation)

    classes = {'WALKING': 1, 'WALKING_UPSTAIRS': 2, 'WALKING_DOWNSTAIRS': 3, 'SITTING': 4, 'STANDING': 5, 'LAYING': 6,
               'STAND_TO_SIT': 7, 'SIT_TO_STAND': 8, 'SIT_TO_LIE': 9, 'LIE_TO_SIT': 10, 'STAND_TO_LIE': 11,
               'LIE_TO_STAND': 12}

    inv_classes = {v: k for k, v in classes.items()}

    total_result = []

    # combine(average) the results for each cross fold validation
    for classification in inv_classes:
        for k in range(len(k_range)):
            k_result = [inv_classes[classification], k_range[k], 0, 0, 0, 0]

            # to make indices work
            classification -= 1

            for i in x_fold_validation_range:

                k_result[2] += x_fold_validations[i][k][classification][2]
                k_result[3] += x_fold_validations[i][k][classification][3]
                k_result[4] += x_fold_validations[i][k][classification][4]
                k_result[5] += x_fold_validations[i][k][classification][5]

            total_result.append(k_result)

            # reset index hack
            classification += 1

    total_result.insert(0, ["classification", "k_value", "tp", "tn", "fp", "fn"])
    writer = csv.writer(open("KNN_k_1-50.csv", "wt", encoding='ascii', newline=''), delimiter=',')
    writer.writerows(total_result)
