#metrics.py

import torch
from matplotlib import pyplot as plt


def compute_metrics(y_true, y_pred, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    metrics = {}

    accuracy = torch.sum(torch.diag(cm)).item() / torch.sum(cm).item()
    metrics['accuracy'] = accuracy

    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []
    fpr_list = []
    fnr_list = []

    for i in range(num_classes):
        TP = cm[i, i].item()
        FP = cm[:, i].sum().item() - TP
        FN = cm[i, :].sum().item() - TP
        TN = cm.sum().item() - (TP + FP + FN)

        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        fpr = FP / (FP + TN + 1e-8)
        fnr = FN / (FN + TP + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        f1_list.append(f1)
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    # Macro average
    metrics['precision'] = sum(precision_list) / num_classes
    metrics['recall'] = sum(recall_list) / num_classes
    metrics['specificity'] = sum(specificity_list) / num_classes
    metrics['f1_score'] = sum(f1_list) / num_classes
    metrics['fpr'] = sum(fpr_list) / num_classes
    metrics['fnr'] = sum(fnr_list) / num_classes

    return metrics, cm

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
