from sklearn.datasets import make_circles, make_moons, make_blobs
from matplotlib import pyplot as plt
import numpy as np
import gzip


def get_data(shape='circles', n_samples=200, std=0.1):
    if (shape == 'sample'):
        X, y = make_blobs(n_samples, centers=2)
    elif(shape == 'circles'):
        X, y = make_circles(n_samples, noise=std)
    elif(shape == 'spiral'):
        X, y = make_moons(n_samples, noise=std)
    else:
        print("Unknow shape! Drawing random sample data.")
        X = np.random.rand(n_samples, 2) * std
        y = np.random.randint(0, 2, n_samples)
    return X, y


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


def show_plot(shape='cirles', n_samples=200, std=0.1):
    X, y = get_data(shape, n_samples, std)
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to add data points')
    class_1, = ax.plot(X_0[:, 0], X_0[:, 1], 'go')
    class_2, = ax.plot(X_1[:, 0], X_1[:, 1], 'ro')
    p = PlotBuilder([class_1, class_2], n_samples, X, y)

    plt.show()
    return p.X, p.y, p


def load_mnist(kind):
    labels_path = './{}-labels-idx1-ubyte.gz'.format(kind)
    images_path = './{}-images-idx3-ubyte.gz'.format(kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images.reshape(len(images), 28, 28), labels


def visualize_mnist(X, y):
    label_idx = [y == i for i in range(10)]
    ims = np.array([X[label_idx[i]][:10] for i in range(10)])

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    f, axarr = plt.subplots(10, 10, figsize=(12, 12))
    for i in range(10):
        axarr[0][i].set_title(classes[i])
        for j in range(10):
            axarr[i][j].imshow(ims[j][i], 'gray')
            axarr[i][j].axis('off')

    plt.show()
