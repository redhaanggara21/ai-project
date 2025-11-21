import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(X, y, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in set(y):
        idx = [i for i,v in enumerate(y) if v==label]
        ax.scatter(X[idx,0], X[idx,1], X[idx,2], label=str(label))
    ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('X3')
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
