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
training_labels=np.array(label_list).astype(np.float32)

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
testing_labels=np.array(label_list).astype(np.float32)

############ Perform Training -- Decision Tree

# define decision tree object

dtree = cv2.ml.DTrees_create();

# set parameters (changing may or may not change results)

dtree.setCVFolds(1);       # the number of cross-validation folds/iterations - fix at 1
dtree.setMaxCategories(20); # max number of categories (use sub-optimal algorithm for larger numbers)
dtree.setMaxDepth(150);       # max tree depth
dtree.setMinSampleCount(1); # min sample count
# dtree.setPriors(np.float32([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]));  # the array of priors, the bigger weight, the more attention to the assoc. class
                                     #  (i.e. a case will be judjed to be maligant with bigger chance))
dtree.setRegressionAccuracy(0);      # regression accuracy: N/A here
dtree.setTruncatePrunedTree(True);   # throw away the pruned tree branches
dtree.setUse1SERule(True);      # use 1SE rule => smaller tree
dtree.setUseSurrogates(False);  # compute surrogate split, no missing data

# specify that the types of our attributes is ordered with a categorical class output
# and we have 7 of them (6 attributes + 1 class label)

var_types = np.array([cv2.ml.VAR_NUMERICAL] * 561 + [cv2.ml.VAR_CATEGORICAL], np.uint16)

# train decision tree object

dtree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_labels.astype(int), varType = var_types));
dtree.save("trained.xml");

############ Perform Testing -- Decision Tree

correct = 0
wrong = 0

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform decision tree prediction (i.e. classification)

    _, result = dtree.predict(testing_attributes[i,:], cv2.ml.ROW_SAMPLE);

    # and for undocumented reasons take the first element of the resulting array
    # as the result

    print("Test data example : {} : result =  {}".format((i+1), inv_classes[int(result[0])]));

    # record results as tp/tn/fp/fn

    if (result[0] == testing_labels[i]) : correct+=1
    elif (result[0] != testing_labels[i]) : wrong+=1

# output summmary statistics

total = correct + wrong

print();
print("Testing Data Set Performance Summary");
print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));

#####################################################################
