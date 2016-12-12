#####################################################################

# Example : load HAPT data set only
# basic illustrative python script

# For use with test / training datasets : HAPT-data-set-DU

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import os
import numpy as np
import time


def svm(attribute_training_filename, label_training_filename,
        attribute_testing_filename, label_testing_filename, data_path="./"):
    ########### Define classes

    classes = {'WALKING': 1, 'WALKING_UPSTAIRS': 2, 'WALKING_DOWNSTAIRS': 3,
               'SITTING': 4, 'STANDING': 5, 'LAYING': 6, 'STAND_TO_SIT': 7, 'SIT_TO_STAND': 8,
               'SIT_TO_LIE': 9, 'LIE_TO_SIT': 10, 'STAND_TO_LIE': 11, 'LIE_TO_STAND': 12
               }  # define mapping of classes
    inv_classes = {v: k for k, v in classes.items()}

    start_load = time.clock()
    ########### Load Data Set

    # Training data - as currenrtly split

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
    training_labels = np.array(label_list).astype(np.int32)

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
    testing_labels = np.array(label_list).astype(np.int32)

    end_load = time.clock()

    print("File load time = " + str(end_load - start_load))

    start_train = time.clock()
    ############ Perform Training -- SVM

    use_svm_autotrain = False

    # define SVM object

    svm = cv2.ml.SVM_create()

    # set parameters (some specific to certain kernels)

    svm.setC(1.0)  # penalty constant on margin optimization
    svm.setType(cv2.ml.SVM_C_SVC)  # multiple class (2 or more) classification
    # set kernel
    # choices : # SVM_LINEAR / SVM_RBF / SVM_POLY / SVM_SIGMOID / SVM_CHI2 / SVM_INTER

    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setGamma(0.5)  # used for SVM_RBF kernel only, otherwise has no effect
    svm.setDegree(3)  # used for SVM_POLY kernel only, otherwise has no effect

    # set the relative weights importance of each class for use with penalty term

    # svm.setClassWeights(np.float32([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # svm.setClassWeights(np.float32([20, 20, 20, 20, 20, 20, 1, 1, 1, 1, 1, 1]))

    # define and train svm object

    if (use_svm_autotrain):

        # use automatic grid search across the parameter space of kernel specified above
        # (ignoring kernel parameters set previously)

        # if it is available : see https://github.com/opencv/opencv/issues/7224

        svm.trainAuto(
            cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_labels.astype(int)),
            kFold=10)
    else:

        # use kernel specified above with kernel parameters set previously
        print(cv2.ml.ROW_SAMPLE)
        svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels)

    end_train = time.clock()

    print("Training time = " + str(end_train - start_train))

    start_test = time.clock()
    ############ Perform Testing -- SVM

    correct = 0  # handwritten digit correctly identified
    wrong = 0  # handwritten digit wrongly identified

    # confustion matrix to store all results
    confusion_matrix = [[0 for x in range(12)] for y in range(12)]

    # for each testing example

    for i in range(0, len(testing_attributes[:, 0])):
        # (to get around some kind of OpenCV python interface bug, vertically stack the
        #  example with a second row of zeros of the same size and type which is ignored).

        sample = np.vstack((testing_attributes[i, :],
                            np.zeros(len(testing_attributes[i, :])).astype(np.float32)))

        # perform SVM prediction (i.e. classification)

        _, results = svm.predict(sample, cv2.ml.ROW_SAMPLE)

        # and for undocumented reasons take the first element of the resulting array
        # as the result

        confusion_matrix[(int(results[0]) - 1)][(int(testing_labels[i]) - 1)] += 1

        # print("Test data example : {} : result =  {}".format((i+1), int(result[0])))
        #
        # # record results as either being correct or wrong
        #
        # if (result[0] == testing_labels[i]) : correct+=1
        # elif (result[0] != testing_labels[i]) : wrong+=1

    result = []

    # for every class
    for classification in range(12):

        # reset metrics
        tp = 0  # predicted class and actual is classification
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
                if (predicted != classification and actual != classification):
                    tn += confusion_matrix[predicted][actual]

                # false positive
                if (predicted == classification and actual != classification):
                    fp += confusion_matrix[predicted][actual]

                # false negative
                if (predicted != classification and actual == classification):
                    fn += confusion_matrix[predicted][actual]

        result.append([classification_name, tp, tn, fp, fn])

    end_test = time.clock()

    print("Testing time = " + str(end_test - start_test))

    return result


if (__name__ == "__main__"):

    total_summary = []

    x_fold_validations = []

    # x_fold_validations = [[["WALKING", 1, 2, 3, 4, 5], ["WALKING", 1, 2, 3, 4, 5], ["STANDING", 1, 2, 3, 4, 5]]]
    # For every cross fold validation
    for i in range(10):
        # Get results for every class vs all the others
        x_fold_validations.append(svm("attributes_train_grouped" + str(i) + ".txt", "labels_train_grouped" + str(i) +
                                      ".txt", "attributes_test_grouped" + str(i) + ".txt", "labels_test_grouped" +
                                      str(i) + ".txt"))

    # Create a dictionary of classification : [tp, tn, fp, fn]
    combined_dict = {'WALKING': [0, 0, 0, 0], 'WALKING_UPSTAIRS': [0, 0, 0, 0], 'WALKING_DOWNSTAIRS': [0, 0, 0, 0],
                     'SITTING': [0, 0, 0, 0], 'STANDING': [0, 0, 0, 0], 'LAYING': [0, 0, 0, 0],
                     'STAND_TO_SIT': [0, 0, 0, 0], 'SIT_TO_STAND': [0, 0, 0, 0],
                     'SIT_TO_LIE': [0, 0, 0, 0], 'LIE_TO_SIT': [0, 0, 0, 0], 'STAND_TO_LIE': [0, 0, 0, 0],
                     'LIE_TO_STAND': [0, 0, 0, 0]
                     }
    for x_fold_validation in x_fold_validations:
        for classification in x_fold_validation:
            stats = combined_dict[classification[0]]
            # for each tp, tn, fp, fn
            for stat in range(len(stats)):
                stats[stat] += classification[stat + 1]
            combined_dict[classification[0]] = stats

    for classification in combined_dict:
        total_summary.append([classification, combined_dict[classification][0], combined_dict[classification][1],
                              combined_dict[classification][2], combined_dict[classification][3]])

    total_summary.insert(0, ["classification", "tp", "tn", "fp", "fn"])
    writer = csv.writer(open("SVM-Linear-Grouped.csv", "wt", encoding='ascii', newline=''), delimiter=',')
    writer.writerows(total_summary)
