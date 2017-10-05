# coding: utf-8
import csv
import numpy as np


test_x = np.empty((0, 3), int)
test_y = np.empty((0, 1), int)
with open('sample.csv') as f:
    reader = csv.reader(f)

    for row in reader:
        test_x = np.append(test_x, np.array([[int(row[0]), int(row[1]), int(row[2])]]), axis=0)
        test_y = np.append(test_y, np.array([[int(row[3])]]), axis=0)

print test_x
print test_y


t1 = np.array([190., 330., 660.])
t2 = np.array([190., 330., 660.]).reshape(3, 1)
print t1
print t2
