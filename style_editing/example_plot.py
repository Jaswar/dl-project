import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

if __name__ == '__main__':
    np.random.seed(42)

    x0 = np.random.multivariate_normal([0.1, 0.5], np.eye(2) * 0.01, 100)
    x1 = np.random.multivariate_normal([0.5, 0.1], np.eye(2) * 0.01, 100)
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(100), np.ones(100)], axis=0)

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(x, y)
    w = svm.coef_[0]
    b = svm.intercept_[0]

    min_x = min(x[:, 0])
    max_x = max(x[:, 0])
    min_y = min(x[:, 1])
    max_y = max(x[:, 1])

    x_points = np.linspace(min_x, max_x, 100)
    y_points = -(w[0] / w[1]) * x_points - b / w[1]

    plt.scatter(x0[:, 0], x0[:, 1], label='With glasses')
    plt.scatter(x1[:, 0], x1[:, 1], label='Without glasses')

    plt.plot(x_points, y_points, c='red')
    w = -w / np.linalg.norm(w) / 10
    mid = len(x_points) // 2
    plt.arrow(x_points[mid], y_points[mid], w[0], w[1], head_width=0.02, head_length=0.02, fc='black', ec='black')

    plt.xlabel('x')
    plt.ylabel('y')

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend()
    plt.show()
