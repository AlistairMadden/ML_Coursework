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

########### Define classes

classes = {'WALKING': 1, 'WALKING_UPSTAIRS': 2, 'WALKING_DOWNSTAIRS': 3,
'SITTING': 4, 'STANDING': 5, 'LAYING': 6, 'STAND_TO_SIT': 7, 'SIT_TO_STAND': 8,
'SIT_TO_LIE': 9, 'LIE_TO_SIT': 10, 'STAND_TO_LIE': 11, 'LIE_TO_STAND': 12
} # define mapping of classes
inv_classes = {v: k for k, v in classes.items()}

########### Load Data Set

path_to_data = "../data/HAPT-data-set-DU" # edit this

# Training data - as currenrtly split

attribute_list = []
label_list = []

reader=csv.reader(open(os.path.join(path_to_data, "Train/x_train.txt"),"rt", encoding='ascii'),delimiter=' ')
for row in reader:
        # attributes in columns 0-561
        attribute_list.append(list(row[i] for i in (range(0,561))))

reader=csv.reader(open(os.path.join(path_to_data, "Train/y_train.txt"),"rt", encoding='ascii'),delimiter=' ')
for row in reader:
        # attributes in column 1
        label_list.append(row[0])

training_attributes=np.array(attribute_list).astype(np.float32)
training_labels=np.array(label_list).astype(np.int32)

# Testing data - as currently split

attribute_list = []
label_list = []

reader=csv.reader(open(os.path.join(path_to_data, "Test/x_test.txt"),"rt", encoding='ascii'),delimiter=' ')
for row in reader:
        # attributes in columns 0-561
        attribute_list.append(list(row[i] for i in (range(0,561))))

reader=csv.reader(open(os.path.join(path_to_data, "Test/y_test.txt"),"rt", encoding='ascii'),delimiter=' ')
for row in reader:
        # attributes in column 1
        label_list.append(row[0])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_labels=np.array(label_list).astype(np.int32)

############ Perform Training -- SVM

use_svm_autotrain = False;

# define SVM object

svm = cv2.ml.SVM_create();



# set parameters (some specific to certain kernels)

svm.setC(1.0); # penalty constant on margin optimization
svm.setType(cv2.ml.SVM_C_SVC); # multiple class (2 or more) classification
# set kernel
# choices : # SVM_LINEAR / SVM_RBF / SVM_POLY / SVM_SIGMOID / SVM_CHI2 / SVM_INTER

svm.setKernel(cv2.ml.SVM_LINEAR);
svm.setGamma(0.5); # used for SVM_RBF kernel only, otherwise has no effect
svm.setDegree(3);  # used for SVM_POLY kernel only, otherwise has no effect

# set the relative weights importance of each class for use with penalty term

svm.setClassWeights(np.float32([1,1,1,1,1,1,1,1,1,1,1,1]));

# define and train svm object

if (use_svm_autotrain) :

    # use automatic grid search across the parameter space of kernel specified above
    # (ignoring kernel parameters set previously)

    # if it is available : see https://github.com/opencv/opencv/issues/7224

    svm.trainAuto(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int)), kFold=10);
else :

    # use kernel specified above with kernel parameters set previously

    svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels);

############ Perform Testing -- SVM

correct = 0 # handwritten digit correctly identified
wrong = 0   # handwritten digit wrongly identified

# confustion matrix to store all results
confusion_matrix = [[0 for x in range(12)] for y in range(12)]

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # (to get around some kind of OpenCV python interface bug, vertically stack the
    #  example with a second row of zeros of the same size and type which is ignored).

    sample = np.vstack((testing_attributes[i,:],
                        np.zeros(len(testing_attributes[i,:])).astype(np.float32)));

    # perform SVM prediction (i.e. classification)

    _, results = svm.predict(sample, cv2.ml.ROW_SAMPLE);

    # and for undocumented reasons take the first element of the resulting array
    # as the result

    confusion_matrix[(int(results[0]) - 1)][(int(testing_labels[i]) - 1)] += 1

    # print("Test data example : {} : result =  {}".format((i+1), int(result[0])));
    #
    # # record results as either being correct or wrong
    #
    # if (result[0] == testing_labels[i]) : correct+=1
    # elif (result[0] != testing_labels[i]) : wrong+=1

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
    
#####################################################################
