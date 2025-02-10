import gnlse
from random import sample
import matplotlib.pyplot as plt
import os


import numpy as np
import torch


def grid(grid_dir):
    # returns grid file from gnlse simulation (used to visualize data)
    grid_dir = os.path.join(grid_dir, 'grid.mat')
    data = gnlse.read_mat(grid_dir)
    return data


def visualize_single_data(data: tuple, grid: dict, ax_freq=None, ax_time=None):
    '''
    :param data: single item of GnlseDataset class
    :param grid: data read from grid file
    :param ax_freq:
    :param ax_time:
    :return: ax_freq, ax_time
    '''

    # splitting data into frequency domain and time domain (in dataset they are merged together)
    data = data[0][0]
    data_freq = np.array(data[: int(len(data) / 2)])
    data_time = np.array(data[int(len(data) / 2):])

    # getting grid and splitting it
    grid_freq = grid['frequency']
    grid_time = grid['time']

    if ax_freq is None:
        ax_freq = plt.axes()

    # creating frequency plot
    ax_freq.set_yscale('log')
    ax_freq.scatter(grid_freq, data_freq, s=2)
    ax_freq.set_xlabel('Wavelength [nm]')
    ax_freq.set_ylabel('Intensity [a.u.]')

    if ax_time is None:
        ax_time = plt.axes()

    # creating time plot
    ax_time.scatter(grid_time, data_time, s=2)
    ax_time.set_xlabel('Delay [ps]')

    return ax_freq, ax_time


def data_description(data, separator='\n') -> str:
    names = ['Beta', 'Gamma', 'Peak power', 'Duration']
    units = ['[ps**2/km]', '[1/W/m]', '[W]', '[ps]']
    if isinstance(data, tuple):
        # preparing data description into string
        description = data[1]
        description = list(description)  # ON GPU IT MAY WORK DIFFERENTLY!!!!
    elif isinstance(data, torch.TensorType):
        description = list(data)
    else:
        raise TypeError('Data can be either tuple from GnlseDataset or dictionary')

    description = [f'{names[i]}={round(float(description[i]), 3)} {units[i]}' for i in range(4)]

    description = separator.join(description)

    return description


def data_description_ax(s, ax=None, from_data=True):
    if from_data:
        s = data_description(s, separator='\n')
    if ax is None:
        ax = plt.axes()
    ax.text(0, 0.5, s, fontsize=12, horizontalalignment='left', verticalalignment='center')
    ax.axis('off')

    return ax


def test_data_single_figure(dataset, grid: dict, nrows=4, list_of_indexes=None):
    if list_of_indexes is None:
        list_of_indexes = sample(population=range(len(dataset)), k=nrows)

    fig, axes = plt.subplots(nrows, 3, gridspec_kw={'width_ratios': [5, 5, 1]}, figsize=(15, 8))

    for iterator, data_index in enumerate(list_of_indexes):
        axes[iterator][0], axes[iterator][1] = visualize_single_data(dataset[data_index], grid,
                                                                     ax_freq=axes[iterator][0],
                                                                     ax_time=axes[iterator][1])

        axes[iterator][2] = data_description_ax(dataset[data_index], axes[iterator][2])


def test_data_multiple_figures(dataset, grid: dict, nrows=4, nfig=5, save_path=None):

    list_of_indexes = sample(population=range(len(dataset)), k=nrows*nfig)
    list_of_indexes = [[list_of_indexes[nrows*i + j] for j in range(nrows)] for i in range(nfig)]
    for figure_idx in range(nfig):
        test_data_single_figure(dataset, grid, nrows, list_of_indexes=list_of_indexes[figure_idx])

        plt.tight_layout()
        #todo
        if save_path is not None:
            if not os.path.exists(save_path):
                print(f'Path "{save_path}" doesn\'t exist. Creating one')
                os.mkdir(save_path)
            plt.savefig(f'{save_path}/Figure_{figure_idx}')


def time_plot(dataset, grid, time_range=None, ax=None):
        """Plotting intensity in linear scale in time domain.

        Parameters
        ----------
        solver : Solution
            Model outputs in the form of a ``Solution`` object.
        time_range : list, (2, )
            Time range. Set [min(``solver.t``), max(``solver.t``)] as default.
        ax : :class:`~matplotlib.axes.Axes`
            :class:`~matplotlib.axes.Axes` instance for plotting.
        norm : float
            Normalization factor for output spectrum. As default maximum of
            square absolute of ``solver.At`` variable is taken.

        Returns
        -------
        ax : :class:`~matplotlib.axes.Axes`
           Used :class:`~matplotlib.axes.Axes` instance.
        """
        dataset = dataset[0][0]
        dataset = np.array(dataset[int(len(dataset) / 2):])

        if ax is None:
            ax = plt.gca()

        grid = grid['time']

        if time_range is None:
            time_range = [np.min(grid), np.max(grid)]

        ax.plot(grid, dataset)

        ax.set_xlim(time_range)
        ax.set_xlabel("Delay [ps]")
        ax.set_ylabel("Power")
        return ax


def wavel_plot(dataset, grid, solver, frequency_range=None, ax=None):
    """Plotting chosen slices of intensity in linear scale in frequency domain.

    Parameters
    ----------
    solver : Solution
        Model outputs in the form of a ``Solution`` object.
    frequency_range : list, (2, )
        frequency range. Set [-150, 150] as default.
    ax : :class:`~matplotlib.axes.Axes`
        :class:`~matplotlib.axes.Axes` instance for plotting
    norm : float
        Normalization factor for output spectrum. As default maximum of
        square absolute of ``solver.AW`` variable is taken.

    Returns
    -------
    ax : :class:`~matplotlib.axes.Axes`
       Used :class:`~matplotlib.axes.Axes` instance.
    """

    if ax is None:
        ax = plt.gca()

    if frequency_range is None:
        frequency_range = [np.min((solver.W - solver.w_0) / 2 / np.pi),
                           np.max((solver.W - solver.w_0) / 2 / np.pi)]

    if norm is None:
        norm = np.max(np.abs(solver.AW)**2)

    IW = np.fliplr(
        np.abs(solver.AW)**2 / norm)

    # indices of interest if no z_slice positions were given
    if z_slice is None:
        iis = [0, -1]  # beginning, end
    # indices of interest nearest to given z_slice positions
    else:
        iis = [np.nonzero(
            np.min(np.abs(solver.Z - z)) == np.abs(solver.Z - z)
        )[0][0] for z in z_slice]

    for i in iis:
        label_i = "z = " + str(solver.Z[i]) + "m"
        ax.plot((solver.W - solver.w_0) / 2 / np.pi, IW[i][:], label=label_i)

    ax.set_xlim(frequency_range)
    ax.set_xlabel("Frequency [Thz]")
    ax.set_ylabel("Normalized Spectral Density")
    ax.legend()
    return ax

    pass

