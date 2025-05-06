import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import numpy as np

def dice_score(pred, target, smooth=1e-5):
    pred = torch.argmax(pred, dim=1)  # [B, D, H, W]
    target = target.squeeze(1)  # If target is [B, 1, D, H, W]
    intersection = (pred == target).float().sum()
    return (2. * intersection + smooth) / (pred.numel() + target.numel() + smooth)

def calculate_metrics(pred, target, num_classes=4):
    pred = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    precision = precision_score(target, pred, average='macro', zero_division=0)
    recall = recall_score(target, pred, average='macro', zero_division=0)
    f1 = f1_score(target, pred, average='macro', zero_division=0)
    jaccard = jaccard_score(target, pred, average='macro', zero_division=0)
    acc = np.mean(pred == target)

    # Specificity and Sensitivity are derived from confusion matrix
    specificity, sensitivity = 0.0, 0.0
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(target, pred, labels=list(range(num_classes)))
        tn = np.sum(np.tril(cm, -1))
        fp = np.sum(np.triu(cm, 1))
        fn = fp  # Symmetric for macro-averaged
        tp = np.trace(cm)
        specificity = tn / (tn + fp + 1e-5)
        sensitivity = tp / (tp + fn + 1e-5)
    except:
        pass

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'dice': (2 * precision * recall) / (precision + recall + 1e-5)
    }