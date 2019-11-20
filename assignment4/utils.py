from matplotlib import pyplot as plt
import numpy as np


def get_data(kind='outlier'):
    data = np.loadtxt('sample_data.csv', delimiter=',')
    if kind != 'outlier':
        data = data[:-2]
    X = data[:, :2]
    y = data[:, 2]
    return X, y


def show_boundary(X, y, predict_func):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)
    plt.axis('off')

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.title('Decision boundary')
    plt.show()


class PlotBuilder:

    def __init__(self, plot, n_samples, X, y):
        self.X = X
        self.y = y
        self.plot = plot
        x0_data = np.array(plot[0].get_xdata())
        y0_data = np.array(plot[0].get_ydata())
        x1_data = np.array(plot[1].get_xdata())
        y1_data = np.array(plot[1].get_ydata())

        self.class_0 = np.zeros((len(x0_data), 2))
        self.class_1 = np.zeros((len(x1_data), 2))

        self.class_0[:, 0] = x0_data
        self.class_0[:, 1] = y0_data

        self.class_1[:, 0] = x1_data
        self.class_1[:, 1] = y1_data

        plot[0].figure.canvas.mpl_connect('button_press_event', self)
        plot[1].figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.plot[0].axes:
            return

        if event.inaxes != self.plot[1].axes:
            return

        if event.button == 1:
            # right click
            self.class_0 = np.vstack((self.class_0,
                                      [event.xdata,
                                       event.ydata]))
        elif event.button == 3:
            # left click
            self.class_1 = np.vstack((self.class_1,
                                      [event.xdata,
                                       event.ydata]))

        self.X = np.vstack((self.class_0, self.class_1))
        self.y = np.array([0] * len(self.class_0) + [1] * len(self.class_1))
        self.plot[0].set_data(self.class_0[:, 0], self.class_0[:, 1])
        self.plot[1].set_data(self.class_1[:, 0], self.class_1[:, 1])
        self.plot[0].figure.canvas.draw()
        self.plot[1].figure.canvas.draw()


def show_plot(kind='outlier'):
    X, y = get_data(kind)
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('A random data sample')
    class_1, = ax.plot(X_0[:, 0], X_0[:, 1], 'go')
    class_2, = ax.plot(X_1[:, 0], X_1[:, 1], 'ro')
    p = PlotBuilder([class_1, class_2], len(X), X, y)

    plt.show()
    return p.X, p.y, p
