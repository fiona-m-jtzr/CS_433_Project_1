####################### CONTENT #######################

#This file contains useful functions, only for the analysis pipeline



####################### IMPORTS #######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



####################### DISPLAY UTILS #######################

def plot_heatmap(y_te, y_pred, title) : 
    #Confusion matrix
    cm = confusion_matrix(y_te, y_pred)

    #Percentage confusion matrix
    cm_ratio = cm/len(y_te)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, ax=axes[0], annot=True, cmap="coolwarm")
    axes[0].set_title(f"Heatmap {title}")

    sns.heatmap(cm_ratio, ax=axes[1], annot=True, cmap="coolwarm")
    axes[1].set_title(f"Ratio heatmap {title}")

    fig.tight_layout()
    fig.savefig(f"heatmap_{title}.png", dpi=300)
    plt.show()



####################### LOGISTIC REGRESSION UTILS #######################

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return(1/(1+np.exp(-t)))