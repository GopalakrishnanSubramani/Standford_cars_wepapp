from result import y_predicted, y_expected, class_names, class_mapping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
import numpy as np
import csv
np.set_printoptions(threshold=sys.maxsize)

txt_file = "src/class_list.txt"

with open('src/class_list.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)[0]

print(len(data))

if __name__ == '__main__':
    # cm =confusion_matrix(y_expected, y_predicted, labels=class_mapping)
    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_expected,y_predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{np.asarray(disp.confusion_matrix)}")
    # print(classification_report(y_predicted, y_expected, target_names=class_names))
    # plt.show()

    pass