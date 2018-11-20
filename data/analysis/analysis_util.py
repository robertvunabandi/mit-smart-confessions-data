import random
from typing import Dict, List, Union, Tuple

import constants

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import numpy as np

import data.data_util


# todo - words that appear in high frequency confessions vs. low
ALL_CONFESSIONS_PATH = "all_confessions/all"


def convert_to_numpy_array(array: List) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return np.array(array)


def get_data_labels() -> List[Tuple]:
    return [
        row[data.data_util.CONFESSION_REACTION_INDEX]
        for row in data.data_util.load_text_with_every_label(ALL_CONFESSIONS_PATH)
    ]


def get_data_texts() -> List[str]:
    return [
        row[data.data_util.CONFESSION_TEXT_INDEX]
        for row in data.data_util.load_text_with_every_label(ALL_CONFESSIONS_PATH)
    ]


def get_frequency(array: List[int]) -> Dict[int, int]:
    frequencies = {}
    for i in array:
        frequencies[i] = frequencies.get(i, 0) + 1
    return frequencies


class Histogram:
    def __init__(self, points: List[Union[float, int]], n_bins: int = 20) -> None:
        self.points = convert_to_numpy_array(points)
        self.n_bins = n_bins

    def plot(self, title: str) -> None:
        fig, axs = plt.subplots(1, 2, tight_layout=True)
        count_in_bins, _, patches = axs[0].hist(self.points, bins=self.n_bins)
        Histogram.color_patches(count_in_bins, patches)
        axs[0].set_title("_")
        axs[0].ylabel = "Frequency"

        # have a second plot for percentage display and format the
        # y-axis to display percentage
        p_count_in_bins, _, p_patches = axs[1].hist(self.points, bins=self.n_bins, density=True)
        Histogram.color_patches(p_count_in_bins, p_patches)
        axs[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        axs[1].set_title("_")
        axs[1].ylabel = "Percentage"

        plt.suptitle(title, fontsize=12)
        # plot the result
        plt.show()

    @staticmethod
    def color_patches(count_in_bins, patches) -> None:
        # color code by height
        fractions = count_in_bins / count_in_bins.max()
        # normalize the data to 0..1 for the full range of the colormap
        norm = matplotlib.colors.Normalize(fractions.min(), fractions.max())
        # loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fractions, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)


class Histogram2D:
    def __init__(
            self,
            x_points: List[Union[float, int]],
            y_points: List[Union[float, int]],
            n_bins: int = 20) -> None:
        self.xs = convert_to_numpy_array(x_points)
        self.ys = convert_to_numpy_array(y_points)
        self.n_bins = n_bins

    def plot(self, title: str, save: bool = False, fname: str = None) -> None:
        fig, axs = plt.subplots(1, 1, figsize=(15, 15), tight_layout=True)
        axs.hist2d(self.xs, self.ys, bins=self.n_bins)
        axs.set_title("_")
        plt.suptitle(title, fontsize=12)
        if save:
            assert fname is not None, "please specify an fname for saving this figure"
            plt.savefig("outputs/hist2d_%s.png" % fname)
        plt.show()


class Scatterplot:
    def __init__(self, x_points: List[Union[float, int]], y_points: List[Union[float, int]]) -> None:
        self.xs = convert_to_numpy_array(x_points)
        self.ys = convert_to_numpy_array(y_points)

    def plot(self, title: str, xlabel: str, ylabel: str, save: bool = False, fname: str = None) -> None:
        plt.scatter(self.xs, self.ys, s=2, alpha=0.15)
        plt.suptitle(title, fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save:
            assert fname is not None, "please specify an fname for saving this figure"
            plt.savefig("outputs/scatter_%s.png" % fname)
        plt.show()


if __name__ == '__main__':
    hist = Histogram([100 * random.random() for _ in range(1000)])
    hist.plot("LongTitle" * 5)
