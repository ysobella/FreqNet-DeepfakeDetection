import torch
import numpy as np
from networks.freqnet import freqnet
from sklearn.metrics import average_precision_score, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader

def validate(model, opt):
    """
    Runs validation on a dataset (val or test).
    """
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Compute accuracy metrics
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5) if np.any(y_true == 0) else None
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5) if np.any(y_true == 1) else None
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return acc, ap, r_acc, f_acc, y_true, y_pred
