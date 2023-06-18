import numpy as np

def f1_score_np(input, target, thresh=0.5):
    # Convert output probabilities to binary values (0 or 1)
    input_binary = (input > thresh).astype(int)
    # Calculate true positives, false positives and false negatives
    true_positives = (input_binary * target).sum()
    false_positives = (input_binary * (1 - target)).sum()
    false_negatives = ((1 - input_binary) * target).sum()
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1_score