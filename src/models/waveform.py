import sys, os
import csv
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore")

# Add src directory to system path
project_path = os.getcwd().split('/src')[0]
sys.path.append(project_path)

try:
    from src.utils.functions import gaussian
except Exception as e:
    print("Failed to import local modules:")
    print(e)


class WaveForm:
    """
    <Description> 
    """
    def __init__(self, csvfile):
        """
        <Description>

        Args:

        Returns:
        """
        self.csvfile            = csvfile
        self.raw_data           = None
        self.processed_data     = None
        self.baseline           = None
        self.main_peak_idx      = None
        self.ingress_idx        = None


    """ ================== """
    """ Processing Methods """
    """ ================== """

    def read_from_csv(self):
        """
        <Description>

        Args:

        Returns:
        """
        try:
            with open(self.csvfile, 'r') as f:
                reader = csv.reader(f)
                data   = np.array(list(reader), dtype=float)
                data   = data.T
                data   = np.array(list(zip(data[0], data[1])))

                self.raw_data       = data
                self.processed_data = data

        except Exception as e:
            print(f"Failed to read csv file {self.csvfile}")
            print(e)


    def rescale(self, xfactor=1, yfactor=1):
        """
        <Description>

        Args:

        Returns:
        """
        processed_data = self.get_data(zipped=True, raw=False)
        rescaled_data  = [(x * xfactor, y * yfactor) for x, y in processed_data]
        self.processed_data = rescaled_data


    def calculate_baseline(self, bins, p0=[100,0,20]):
        """
        <Description>

        Args:

        Returns:
        """

        # Extract waveform y-axis data
        _, y = self.get_data(zipped=False)

        # Generate numpy histogram
        hist, bin_edges = np.histogram(y, bins)

        # Get mid-pounts of bin edges so as to make x and y arrays plottable
        bin_mids = bin_edges[:-1] + np.diff(bin_edges)/2

        # Catch all zero indices
        zero_indexes = []
        for idx, y_val in enumerate(hist):
            if y_val == 0:
                zero_indexes.append(idx)

        # Remove zeros
        hist = np.delete(hist, zero_indexes, axis=None)
        bin_mids = np.delete(bin_mids, zero_indexes, axis=None)

        # Fit to Gaussian
        try:
            popt, pcov = curve_fit(gaussian, bin_mids, hist, p0=p0)
        except Exception as e:
            print("Failed to fit baseline")
            print(e)
        
        baseline = popt[1] # the mean value of the fitted gaussian

        self.baseline = baseline


    def zero_baseline(self):
        """
        <Description>

        Args:

        Returns:
        """
        baseline = self.get_baseline()
        
        data = self.get_data()
        baseline_corrected_data = [(x, y-baseline) for x,y in data]

        self.processed_data = baseline_corrected_data


    def smooth(self, sigma=2):
        """
        <Description>

        Args:

        Returns:
        """
        x,y       = self.get_data(zipped=False)
        wf_smooth = gaussian_filter1d(y, sigma=sigma)
        
        smoothed_data = np.array(list(zip(x, wf_smooth)))
        self.processed_data = smoothed_data


    def detect_main_peak(self, ROI, height, width=6, distance=10, prominence=12):
        """
        <Description>

        Args:
            ROI (tuple[int,int]) : integer value of the left-most and right-most INDEX of the
                                   Region Of Interest.

        Returns:
        """
        a = int(ROI[0]); b = int(ROI[1])
        _,y = self.get_data(zipped=False)
        wf_cut = y[a:b]

        # Find peaks
        peaks, _ = find_peaks(wf_cut, height=height, width=width, distance=distance, prominence=prominence) 

        try:
            # Extract index of first peak
            peak_idx = a + peaks[0]
            self.main_peak_idx = peak_idx
        except:
            pass


    def identify_ingress(self, threshold, peak_idx, peak_val):
        """
        <Description>

        Args:

        Returns:
        """
        if peak_idx != None:
            _, y   = self.get_data(zipped=False)
            wf_cut = y[:peak_idx]

            ingress_idx = np.argwhere(wf_cut >= threshold)[0][0]
            self.ingress_idx = ingress_idx
        else:
            pass
        


    """ =========== """
    """ Get Methods """
    """ =========== """


    def get_data(self, zipped=True, raw=False):
        """
        <Description>

        Args:

        Returns:

        """
        if zipped == True:
            if raw == False:
                return self.processed_data
            else:
                return self.raw_data

        else:
            if raw == False:
                x,y = zip(*self.processed_data)
            else:
                x,y = zip(*self.raw_data)
            return np.array(x),np.array(y)


    def get_baseline(self):
        """
        <Description>

        Args:

        Returns:
        """
        return self.baseline


    def get_main_peak(self):
        """
        <Description>

        Args:

        Returns:
        """
        peak_idx = self.main_peak_idx

        if peak_idx != None:
            x, y = self.get_data(zipped=False)
            peak_val = y[peak_idx]

        else:
            peak_val = None

        return peak_idx, peak_val

    def get_ingress(self):
        """
        <Description>

        Args:

        Returns:

        """
        ingress_idx = self.ingress_idx

        x, _ = self.get_data(zipped=False)
        ingress_time_val = x[ingress_idx]

        return ingress_idx, ingress_time_val
