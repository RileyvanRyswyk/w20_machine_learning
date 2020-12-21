import matplotlib.pyplot as plt


def plot_decision_surface(X_, Y_, X, Y, tit):
    # Inputs
    #   X_ : grid data points
    #   Y_ : grid predicted labels
    #   X  : train data points
    #   Y  : train ground truth labels

    plt.subplot()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(tit)
    plt.axis([min(X_[:, 0]), max(X_[:, 0]), min(X_[:, 1]), max(X_[:, 1])])

    plt.plot(X_[Y_ == 1][:, 0], X_[Y_ == 1][:, 1], c='cyan', marker=(8, 2, 0), linewidth=0.5)
    plt.plot(X_[Y_ == -1][:, 0], X_[Y_ == -1][:, 1], c='yellow', marker=(8, 2, 0), linewidth=0.5)

    plt.plot(X[:, 0:1][Y == 1], X[:, 1:2][Y == 1], c='b', marker='x', linestyle='none', markersize=5)
    plt.plot(X[:, 0:1][Y == -1], X[:, 1:2][Y == -1], c='r', marker='o', linestyle='none', markersize=5,
             fillstyle='none')
    
    plt.show()
