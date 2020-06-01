def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for ind in range(len(predictions)):
        i = predictions[ind]
        j = gt[ind]
        if j==1 and i==1:
            tp += 1
        elif j==1 and i==0:
            fn += 1
        elif j==0 and i==1:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp  + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = sum(prediction == ground_truth) / len(prediction)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = sum(prediction == ground_truth) / len(prediction)

    return accuracy
