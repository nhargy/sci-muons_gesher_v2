import sys, os
import numpy as np
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# Add src directory to system path
project_path = os.getcwd().split('/src')[0]
sys.path.append(project_path)

try:
    from src.models.waveform import WaveForm
    from src.utils.functions import linear
except Exception as e:
    print("Failed to import local modules:")
    print(e)


class Event:
    """
    <Description>
    """
    def __init__(self, dirpath, segment):
        self.dirpath           = dirpath
        self.segment           = segment
        self.ROI               = None
        self.waveform_matrix   = None
        self.ingress_matrix    = None
        self.timestamp         = None
        self.delta_t_array     = None
        self.positions         = None
        self.linear_popt       = None
        self.peak_threshold    = None
        self.ingress_threshold = None
        self.angle             = None
        self.track_popt        = None
        self.hit_coordinates   = None


    """ ================== """
    """ Processing Methods """
    """ ================== """

    def read_timestamp(self):
        info_path = os.path.join(self.dirpath, 'scope-1_info.txt')
        with open(info_path, 'r') as f:
            lines = f.readlines()

            count = 0
            for line in lines:
                if 'Time Tags' in line:
                    count+=1
                    if count == self.segment:
                        timestamp = line.split(" = ")[-1]
                        timestamp = float(timestamp.split('\'')[1])
                        break

        self.timestamp = timestamp


    def process_waveform(self, waveform):
        waveform.rescale(1e9, -1e3)
        waveform.smooth()
        waveform.calculate_baseline()
        waveform.zero_baseline()


    def calculate_peak_and_ingress(self):
        try:
            for i in self.waveform_matrix:
                for wf in i:
                    wf.detect_main_peak((self.ROI[0], self.ROI[1]), self.peak_threshold)
                    wf.identify_ingress(self.ingress_threshold, (self.ROI[0], self.ROI[1]))
        except Exception as e:
            pass


    def gather_waveforms(self):
        def channel_path(scope, channel):
            path = os.path.join(self.dirpath, f'scope-{scope}-seg{self.segment}-ch{channel}.csv')
            return path

        def inst_and_process_waveform(scope, channel):
            try:
                path = channel_path(scope, channel)
                wf = WaveForm(path)
                self.process_waveform(wf)
                return wf
            except:
                return None

        wf1 = inst_and_process_waveform(1,1)
        wf2 = inst_and_process_waveform(1,2)
        wf3 = inst_and_process_waveform(1,3)
        wf4 = inst_and_process_waveform(1,4)
        wf5 = inst_and_process_waveform(2,1)
        wf6 = inst_and_process_waveform(2,2)
        wf7 = inst_and_process_waveform(2,3)
        wf8 = inst_and_process_waveform(2,4)

        self.waveform_matrix = [[wf1, wf2],[wf3, wf4],[wf5, wf6],[wf7, wf8]]


    def calculate_ingress_matrix(self):

        ingress_arr = []
        for i, plate in enumerate(self.waveform_matrix):
            for j, wf in enumerate(plate):
                try:
                    _, ingress_val = wf.get_ingress()
                    ingress_arr.append(ingress_val)
                except:
                    ingress_arr.append(np.nan)

        ingress_matrix = [[ingress_arr[0], ingress_arr[1]],[ingress_arr[2], ingress_arr[3]],[ingress_arr[4], ingress_arr[5]],[ingress_arr[6], ingress_arr[7]]]

        self.ingress_matrix = ingress_matrix


    def calculate_delta_t_array(self):
        delta_t_array = []
        for plate in self.ingress_matrix:
            dt = plate[0] - plate[1]
            delta_t_array.append(dt)

        self.delta_t_array = delta_t_array


    def calculate_track(self, err=25, min_hit=0, max_hit=144):

        positions, linear_popt = self.get_track_params()
        m = linear_popt[0]
        c = linear_popt[1]

        def inverse_linear(t, m, c):
            return (t-c)/m

        delta_t_array   = self.get_delta_t_array()
        hit_coordinates = inverse_linear(delta_t_array, m, c)

        for idx, hit in enumerate(hit_coordinates):
            if hit < min_hit:
                hit_coordinates[idx] = min_hit
            elif hit > max_hit:
                hit_coordinates[idx] = max_hit

        # Remove NaN values
        valid_indices = ~np.isnan(positions) & ~np.isnan(hit_coordinates)
        x_clean = positions[valid_indices]
        y_clean = hit_coordinates[valid_indices]

        # Fit track to linear function
        popt, pcov = curve_fit(linear, x_clean, y_clean)

        gradient = popt[0]
        angle    = -np.arctan(gradient) * 180/np.pi

        self.angle = angle
        self.track_popt = popt
        self.hit_coordinates = hit_coordinates


    """ =========== """
    """ Get Methods """
    """ =========== """

    def get_timestamp(self):
        timestamp = self.timestamp
        return timestamp


    def get_waveform_matrix(self):
        waveform_matrix = self.waveform_matrix
        return waveform_matrix


    def get_ingress_matrix(self):
        ingress_matrix = self.ingress_matrix
        return ingress_matrix


    def get_delta_t_array(self):
        delta_t_array = np.array(self.delta_t_array)
        return delta_t_array


    def get_track_params(self):
        positions   = np.array(self.positions)
        linear_popt = self.linear_popt
        return positions, linear_popt


    def get_angle(self):
        angle = self.angle
        return angle


    def get_hit_bools(self):
        hit_bools = []
        try:
            for hit in self.hit_coordinates:
                if 0 < hit < 144:
                    hit_bools.append(True)
                else:
                    hit_bools.append(False)
        except:
            pass
        return hit_bools


    """ =========== """
    """ SET METHODS """
    """ =========== """

    def set_peak_threshold(self, peak_threshold):
        self.peak_threshold = peak_threshold


    def set_ingress_threshold(self, ingress_threshold):
        self.ingress_threshold = ingress_threshold


    def set_ROI(self, ROI, index=False):

        if index == True:
            self.ROI = ROI

        elif index == False:
            # This is the case where the user enters ROI in nanoseconds
            try:
                x, _ = self.waveform_matrix[0][0].get_data(zipped=False)

                a = np.argmin(np.abs(x - ROI[0]))
                b = np.argmin(np.abs(x - ROI[1]))

                self.ROI = (a,b)
            except Exception as e:
                print(e)
                print("Could not find any waveform object in wavefrom_matrix.")


    def set_track_params(self, positions=None, linear_popt=None):
        self.positions   = positions
        self.linear_popt = linear_popt
