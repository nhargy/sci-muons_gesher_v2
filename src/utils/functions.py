import numpy as np

def linear(x, m, c):
    return m * x + c


def gaussian(x, A, m, s):
    return A * np.exp(-((x-m)**2)/(2*(s**2)))


def decay(x, A, t):
    return A * np.exp(-x/t)


def remove_nans(data):
    valid_indices = ~np.isnan(data)
    data_clean    = data[valid_indices]
    return data_clean

def hist_to_scatter(data, bins, density = False):
    hist, bin_edges = np.histogram(data, bins = bins, density = density)
    bin_mids = bin_edges[:-1] + np.diff(bin_edges)/2
    return bin_mids, hist
