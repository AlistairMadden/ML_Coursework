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

def knn(attribute_training_filename, label_training_filename,
        attribute_testing_filename, label_testing_filename, data_path="./"):

    ########### Define classes

    classes = {'WALKING': 1, 'WALKING_UPSTAIRS': 2, 'WALKING_DOWNSTAIRS': 3,
    'SITTING': 4, 'STANDING': 5, 'LAYING': 6, 'STAND_TO_SIT': 7, 'SIT_TO_STAND': 8,
    'SIT_TO_LIE': 9, 'LIE_TO_SIT': 10, 'STAND_TO_LIE': 11, 'LIE_TO_STAND': 12
    } # define mapping of classes
    inv_classes = {v: k for k, v in classes.items()}

    ########### Load Data Set

    # Training data - as currenrtly split

    attribute_list = []
    label_list = []

    reader=csv.reader(open(os.path.join(data_path, attribute_training_filename),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in columns 0-561
            attribute_list.append(list(row[i] for i in (range(0,561))))

    reader=csv.reader(open(os.path.join(data_path, label_training_filename),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in column 1
            label_list.append(row[0])

    training_attributes=np.array(attribute_list).astype(np.float32)
    training_labels=np.array(label_list).astype(np.float32)

    # Testing data - as currently split

    attribute_list = []
    label_list = []

    reader=csv.reader(open(os.path.join(data_path, attribute_testing_filename),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in columns 0-561
            attribute_list.append(list(row[i] for i in (range(0,561))))

    reader=csv.reader(open(os.path.join(data_path, label_testing_filename),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in column 1
            label_list.append(row[0])

    testing_attributes=np.array(attribute_list).astype(np.float32)
    testing_labels=np.array(label_list).astype(np.float32)

    ############ Perform Training -- k-NN

    # define kNN object

    knn = cv2.ml.KNearest_create();

    # set to use BRUTE_FORCE neighbour search as KNEAREST_KDTREE seems to  break
    # on this data set (may not for others - http://code.opencv.org/issues/2661)

    knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE);

    # set default 3, can be changed at query time in predict() call

    knn.setDefaultK(25);

    # set up classification, turning off regression

    knn.setIsClassifier(True);

    # perform training of k-NN

    knn.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels);

    ############ Perform Testing -- k-NN

    # confustion matrix to store all results
    confusion_matrix = [[0 for x in range(12)] for y in range(12)]

    # for each testing example

    for i in range(0, len(testing_attributes[:,0])) :

        # perform k-NN prediction (i.e. classification)

        # (to get around some kind of OpenCV python interface bug, vertically stack the
        #  example with a second row of zeros of the same size and type which is ignored).

        sample = np.vstack((testing_attributes[i,:],
                            np.zeros(len(testing_attributes[i,:])).astype(np.float32)))

        # now do the prediction returning the result, results (ignored) and also the responses
        # + distances of each of the k nearest neighbours
        # N.B. k at classification time must be < maxK from earlier training

        _, results, neigh_respones, distances = knn.findNearest(sample, k = 25);

        confusion_matrix[(int(results[0]) - 1)][(int(testing_labels[i]) - 1)] += 1

    # output summmary statistics

    # create dictionary to compare statistics for each classification versus all
    # other classifications
    class_vs_the_rest = {}

    # for every attribute
    for attribute in range(12):

        # reset metrics
        tp = 0 # predicted attribute and actual is attribute
        tn = 0 # predicted not attribute and actual is not attribute
        fp = 0 # predicted as attribute, but actual is different
        fn = 0 # predicted as not attribute, but actual is attribute

        # set attribute name and true positive rate
        attribute_name = inv_classes[attribute + 1]
        tp = confusion_matrix[attribute][attribute]

        # for every row (predicted classification)
        for predicted in range(12):
            # for every column (actual classification)
            for actual in range(12):

                # true negative
                if (predicted != attribute and actual != attribute):
                    tn += confusion_matrix[predicted][actual]

                # false positive
                if (predicted == attribute and actual != attribute):
                    fp += confusion_matrix[predicted][actual]

                # false negative
                if (predicted != attribute and actual == attribute):
                    fn += confusion_matrix[predicted][actual]

        class_vs_the_rest[attribute_name] = [tp, tn, fp, fn]

    total = len(testing_attributes[:,0])

    for attribute in class_vs_the_rest:

        tp = class_vs_the_rest[attribute][0] # predicted attribute and actual is attribute
        tn = class_vs_the_rest[attribute][1] # predicted not attribute and actual is not attribute
        fp = class_vs_the_rest[attribute][2] # predicted as attribute, but actual is different
        fn = class_vs_the_rest[attribute][3] # predicted as not attribute, but actual is attribute

        correct = tp + tn
        wrong = fp + fn

        print();
        print("Testing Data Set Performance Summary - " + attribute + " vs the rest");
        print("TP : {}%".format(round((tp / float(total)) * 100, 2)));
        print("TN : {}%".format(round((tn / float(total)) * 100, 2)));
        print("FP : {}%".format(round((fp / float(total)) * 100, 2)));
        print("FN : {}%".format(round((fn / float(total)) * 100, 2)));
        print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));
        print("Total Wrong : {}%".format(round((wrong / float(total)) * 100, 2)));

if (__name__ == "__main__"):
    for i in range(10):
        knn("attributes_train" + i + ".txt", "labels_train"+i+".txt", "attributes_test"+i+"txt", "labels_test"+i+"txt")
