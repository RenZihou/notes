import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.svm import SVC  # Support Vector Classifier


# test_X, test_y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=.8)
test_X, test_y = make_circles(n_samples=100, factor=.1, noise=.1)

svc_model = SVC(kernel='rbf', C=1, gamma=.1)
svc_model.fit(test_X, test_y)


def plot_svc(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='k', levels=(-1, 0, 1), alpha=.5, linestyles=('--', '-', '--'))

    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=250, linewidth=1, facecolors='none', edgecolors='black')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return None


if __name__ == '__main__':
    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y, s=50, cmap='autumn')
    plot_svc(svc_model, plot_support=True)
    plt.show()
