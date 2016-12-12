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


def dtree(attribute_training_filename, label_training_filename,
          attribute_testing_filename, label_testing_filename, data_path="./", min_sample_count=1):
    ########### Define classes

    classes = {'WALKING': 1, 'WALKING_UPSTAIRS': 2, 'WALKING_DOWNSTAIRS': 3,
               'SITTING': 4, 'STANDING': 5, 'LAYING': 6, 'STAND_TO_SIT': 7, 'SIT_TO_STAND': 8,
               'SIT_TO_LIE': 9, 'LIE_TO_SIT': 10, 'STAND_TO_LIE': 11, 'LIE_TO_STAND': 12
               }  # define mapping of classes
    inv_classes = {v: k for k, v in classes.items()}

    start_load = time.clock()
    ########### Load Data Set

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

    print("File load time: " + str(end_load - start_load) + "s")

    start_train = time.clock()
    ############ Perform Training -- Decision Tree

    # define decision tree object

    dtree = cv2.ml.DTrees_create()

    # set parameters (changing may or may not change results)

    dtree.setCVFolds(1)  # the number of cross-validation folds/iterations - fix at 1
    dtree.setMaxCategories(2)  # max number of categories (use sub-optimal algorithm for larger numbers)
    dtree.setMaxDepth(20)  # max tree depth
    dtree.setMinSampleCount(min_sample_count)  # min sample count
    dtree.setRegressionAccuracy(0)  # regression accuracy: N/A here
    dtree.setTruncatePrunedTree(True)  # throw away the pruned tree branches
    dtree.setUse1SERule(True)  # use 1SE rule => smaller tree
    dtree.setUseSurrogates(False)  # compute surrogate split, no missing data

    # specify that the types of our attributes is ordered with a categorical class output
    # and we have 562 of them (561 attributes + 1 class label)

    var_types = np.array([cv2.ml.VAR_NUMERICAL] * 561 + [cv2.ml.VAR_CATEGORICAL], np.uint8)

    # train decision tree object
    dtree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_labels, varType=var_types))

    end_train = time.clock()

    print("Training time: " + str(end_train - start_train) + "s")

    start_test = time.clock()
    ############ Perform Testing -- Decision Tree

    # confusion matrix to store all results
    confusion_matrix = [[0 for x in range(12)] for y in range(12)]

    # for each testing example
    for i in range(0, len(testing_attributes) - 2):

        # print(i)

        _, result = dtree.predict(testing_attributes[i, :], cv2.ml.ROW_SAMPLE)

        # and for undocumented reasons take the first element of the resulting array
        # as the result

        confusion_matrix[(int(result[0]) - 1)][(int(testing_labels[i]) - 1)] += 1
        # print(i)

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

    print("Test time: " + str(end_test - start_test) + "s")

    return result


if (__name__ == "__main__"):

    total_summary = []

    for max_depth in range(2, 101):

        x_fold_validations = []

        # x_fold_validations = [[["WALKING", 1, 2, 3, 4, 5], ["WALKING", 1, 2, 3, 4, 5], ["STANDING", 1, 2, 3, 4, 5]]]
        # For every cross fold validation
        for i in range(10):
            # Get results for every class vs all the others
            x_fold_validations.append(dtree("attributes_train" + str(i) + ".txt", "labels_train" + str(i) + ".txt",
                                            "attributes_test" + str(i) + ".txt", "labels_test" + str(i) + ".txt", "./",
                                            max_depth))


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
            total_summary.append([classification, max_depth, combined_dict[classification][0],
                                  combined_dict[classification][1], combined_dict[classification][2],
                                  combined_dict[classification][3]])

    print("summary produced")
    total_summary.insert(0, ["classification", "tp", "tn", "fp", "fn"])
    writer = csv.writer(open("dtree_cat-2_depth-2-100_count-10.csv", "wt", encoding='ascii', newline=''), delimiter=',')
    writer.writerows(total_summary)
    print("file written")
