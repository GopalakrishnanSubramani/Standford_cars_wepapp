import torch
from datasets import get_datasets, get_data_loaders,dataset_test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import build_model
import numpy as np
from sklearn.metrics import classification_report

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__=='__main__':
    class_mapping = []
    train_dataset, val_dataset, class_names = get_datasets()
    print(len(val_dataset))
    for idx,name in enumerate(class_names):
        class_mapping.append(name)

    path = "outputs/model_196.pth"
    # load back the model
    model = build_model(pretrained=False)
    model = model.to('cpu')
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["model_state_dict"])

    # get a sample from the validation dataset for inference
    y_predicted= []
    y_expected = []
    report = classification_report(y_predicted, y_expected, target_names=class_mapping)
    for i in range(len(val_dataset)):
        input, target = val_dataset[i][0][None, ...], val_dataset[i][1]
        # make an inference
        predicted, expected = predict(model, input, target,
                                    class_mapping)
        y_predicted.append((predicted))
        y_expected.append(expected)

    cm =confusion_matrix(y_predicted, y_expected, labels=class_mapping)
    print(classification_report(y_predicted, y_expected, target_names=class_mapping))
