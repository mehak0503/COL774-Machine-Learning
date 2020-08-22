import seaborn as sn
import matplotlib.pyplot as plt

def plot_confusion(cm, ticks, title):
    """Plot the confusion matrix."""

    #cm = make_confusion(actual, predicted)
    #acc = " - %.2f%% accuracy" % (accuracy(actual, predicted) * 100)

    # Could replace the above line by sklearn.metrics.confusion_matrix
    # cm = confusion_matrix(actual, predicted)

    plt.figure(figsize=(10, 7))

    ax = sn.heatmap(cm, fmt="d", annot=True, cbar=False,
                    cmap=sn.cubehelix_palette(15),
                    xticklabels=ticks, yticklabels=ticks)
    ax.set(xlabel="Predicted", ylabel="Actual")

    # Move X-Axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title(title, y=1.10)

    plt.savefig(title + ".png")
    plt.close()

#cm = [[14230,3867,1295,420,357],[2756,3266,3489,997,330],[1356,1791,5485,5014,885],[1126,883,3329,17213,6807],[3525,517,1070,15283,38427]]
#plot_confusion(cm,[1,2,3,4,5],"NB")
#cm = [[14265,3903,1200,458,343], [2765, 3439,3301,1012,321], [1366,1822,5418,5078,847], [1115,910,3121,17423,6789], [3518,488,1065, 15295, 38456]]
#plot_confusion(cm,[1,2,3,4,5],"Naive Bayes Prediction on test data")


