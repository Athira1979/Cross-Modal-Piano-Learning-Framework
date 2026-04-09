#metrics.py

import torch 

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
    
    for i in range(num_classes):
        TP = cm[i, i].item()
        FP = cm[:, i].sum().item() - TP
        FN = cm[i, :].sum().item() - TP
        TN = cm.sum().item() - (TP + FP + FN)

        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        f1_list.append(f1) 
 
    metrics['precision'] = sum(precision_list) / num_classes
    metrics['recall'] = sum(recall_list) / num_classes
    metrics['specificity'] = sum(specificity_list) / num_classes
    metrics['f1_score'] = sum(f1_list) / num_classes 

    return metrics, cm

