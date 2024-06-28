import cv2
import numpy as np
from matplotlib.patches import Polygon


def crop_and_pad(image, newdim):
    outw, outh = newdim

    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels > 1:
        out = np.zeros((outh, outw, nchannels))
    else:
        out = np.zeros((outh, outw))

    image_w, image_h = image.shape[1], image.shape[0]

    cropx, cropy = min(outw, image_w), min(outh, image_h)
    if nchannels > 1:
        out[:cropy, :cropx, :] = image[:cropy, :cropx, :]
    else:
        out[:cropy, :cropx] = image[:cropy, :cropx]
    return out


def plot_points_as_box(xys, axis, fill=None, **plot_params):
    if not isinstance(xys, np.ndarray):
        xys = np.array(xys)
    xs = [*xys[:, 0], xys[0, 0]]
    ys = [*xys[:, 1], xys[0, 1]]
    p = axis.plot(xs, ys, **plot_params)
    if fill:
        poly = Polygon(list(zip(xs, ys)), closed=True, facecolor=p[0].get_color(), alpha=0.3)
        axis.add_patch(poly)
    return p[0].get_color()


def loadrgba(filename, rgba=True):
    filename = str(filename)
    return cv2.cvtColor(
        cv2.imread(filename), cv2.COLOR_BGR2RGBA if rgba else cv2.COLOR_BGR2RGB
    )


def remove_box(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
