import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
plt.rcParams["font.family"] = "Arial"

import scenting_classification.modules.Utils as Utils

def evaluate_for_ROC(loader, model, threshold=0.5):
    with torch.no_grad():
        model.eval()

        y_trues = []
        y_preds = []
        for i, (X, y) in enumerate(loader):
            # Run batch through model
            X = Utils.convert_X_for_resnet(X)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)

            # Find thresholed predictions
            thresholded_preds = probs[:,1] > threshold

            y_trues.append(y.tolist())
            y_preds.append(thresholded_preds.tolist())

    return y_trues, y_preds

def evaluation_dataframe(thresholds, loader, model):
    thresholded_outputs = {f'threshold_{t:.02f}': {'true_ys': {}, 'predicted_ys': {}} for t in thresholds}

    for i, (key, val) in enumerate(thresholded_outputs.items()):
        y_trues, y_preds = evaluate_for_ROC(loader, model, threshold=thresholds[i])
        y_trues = np.array(y_trues).flatten()
        y_preds = np.array(y_preds).flatten()
        thresholded_outputs[key]['true_ys'] = y_trues
        thresholded_outputs[key]['predicted_ys'] = y_preds

    truth_outputs = pd.DataFrame(thresholded_outputs)

    # Compute TP, FP, TN, FN and add to df
    for key, val in thresholded_outputs.items():
        # Find TP, FP, TN, FN for each threshold
        scenting_preds = truth_outputs[key]['predicted_ys'] == 1
        scenting_trues = truth_outputs[key]['true_ys'] == 1
        nonscenting_preds = truth_outputs[key]['predicted_ys'] == 0
        nonscenting_trues = truth_outputs[key]['true_ys'] == 0

        # Compute measurements
        tp = sum(np.logical_and(scenting_preds, scenting_trues))
        tn = sum(np.logical_and(nonscenting_preds, nonscenting_trues))
        fp = sum(np.logical_and(scenting_preds, nonscenting_trues))
        fn = sum(np.logical_and(nonscenting_preds, scenting_trues))

        # Compute TPR and FPR
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        # Save all measurements
        thresholded_outputs[key]['tp'] = tp
        thresholded_outputs[key]['tn'] = tn
        thresholded_outputs[key]['fp'] = fp
        thresholded_outputs[key]['fn'] = fn
        thresholded_outputs[key]['tpr'] = tpr
        thresholded_outputs[key]['fpr'] = fpr

    truth_outputs = pd.DataFrame(thresholded_outputs)
    return truth_outputs


def compute_AUC(fprs, tprs, loader):

    sorted_idxs = np.argsort(fprs)
    sorted_fprs = np.array(fprs)[sorted_idxs]
    sorted_tprs = np.array(tprs)[sorted_idxs]

    auc = 0
    for i in range(len(sorted_fprs)-1):
        dx = np.abs(sorted_fprs[i+1] - sorted_fprs[i])
        y_half = sorted_tprs[i] + np.abs(sorted_tprs[i+1] - sorted_tprs[i])/2
        auc += y_half*dx
    return auc

def plot_ROC(truth_outputs, loader, color, save_path):
    tprs = list(truth_outputs.loc['tpr'])
    fprs = list(truth_outputs.loc['fpr'])

    auc = compute_AUC(fprs, tprs, loader)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)

    ax.plot(fprs, tprs, linewidth=2, c=color)
    ax.plot([0, 1], [0, 1], '--', linewidth=2, c='pink')
    ax.annotate(f'AUC: {auc:0.3f}', xy=(0.7,0.05), fontsize=10)  #0.7,0.05

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.grid(b=True, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.yaxis.grid(b=True, color=(0,0,0), alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'{loader.title()} set ROC')
    plt.savefig(f'{save_path}/{loader}_roc.png', bbox_inches='tight', dpi=150)

def compute_fscore(tp, fn, fp, loader):
    test_recall = tp / (tp + fn)
    test_precision = tp / (tp + fp)
    test_f_score = (2*test_recall*test_precision) / (test_recall + test_precision)
    return f'{loader.title()} recall: {test_recall:0.4f}, Precision: {test_precision:0.4f}, F1 Score: {test_f_score:0.4f}'

def plot_confusion_matrix(truth_outputs, loader, colormap, save_path):
    test_cm = np.zeros((2,2), dtype=np.int)

    tp = truth_outputs['threshold_0.50']['tp']
    fp = truth_outputs['threshold_0.50']['fp']
    tn = truth_outputs['threshold_0.50']['tn']
    fn = truth_outputs['threshold_0.50']['fn']

    test_cm[0,0] = tp
    test_cm[0,1] = fp
    test_cm[1,0] = fn
    test_cm[1,1] = tn

    # annot_kws = {"ha": 'center',"va": 'center'}

    fig, ax = plt.subplots(figsize=(5,5), dpi=100)
    sns.heatmap(test_cm, cmap=colormap, fmt="d", annot=True,  ax=ax,
                xticklabels=['Scenting', 'Non-scenting'],
                yticklabels=['Scenting', 'Non-Scenting'], )
    ax.set_ylim([0,2])
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(f'{loader.title()} set confusion matrix')

    print(compute_fscore(tp, fn, fp, loader))
    plt.savefig(f'{save_path}/{loader}_cm.png', bbox_inches='tight', dpi=150)
