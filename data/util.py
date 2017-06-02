import matplotlib.pyplot as plt
import numpy as np


def scoring(truth, predicted, visualize=False):
    """Precision, and recall of predicted sequence based on truth and optionally plot both sequences

    Args:
        truth (list of bool): Ground truth
        predicted (list of bool): Predicted sequence
    Returns:
        [precision, recall, number of seconds of true]
    """
    if visualize:
        plt.plot(np.array(range(len(truth)))/30.0, truth*5, label="truth")
        plt.plot(np.array(range(len(truth)))/30.0, predicted*2, label="predicted")
        plt.ylim([0,6])
        plt.legend()
        plt.show()

    pred_locs = np.where(predicted==1)[0]
    for loc in pred_locs:
        predicted[loc-15:loc+15] = 1
    true_correct = sum(np.logical_and(predicted,truth==1))
    true = sum(truth)
    detected = sum(predicted)

    precision = true_correct/float(detected)
    recall = true_correct/true

    return [precision, recall, true/30.0]