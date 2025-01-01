import sys, os
import json
import numpy as np
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# Add src directory to system path
project_path = os.getcwd().split("/src")[0]
sys.path.append(project_path)

try:
    from src.models.event import Event
    from src.utils.functions import gaussian
except Exception as e:
    print("Failed to import local modules:")
    print(e)

# Define important paths
lcd_path  = os.path.join(project_path, "lcd")
out_path  = os.path.join(project_path, "out")
plt_path  = os.path.join(project_path, "plt")


class Run:

    def __init__(self):
        self.data = []


    def check_segment_number(self, runpath):

        seg = 1
        for file in os.listdir(runpath):
            print(f"Current seg: {seg}")
            try:
                segment_number = int(file.split("seg")[1].split("-")[0])
                if segment_number > seg:
                    seg = segment_number
            except:
                pass

        return segment_number


    def event_processor(self, event, PEAK_THRESH=125, INGRESS_THRESH=25, T_MIN=-50, T_MAX=75, L=43):

        # timestamp
        event.read_timestamp()
        timestamp = event.get_timestamp()

        # set peak and ingress thresholds
        event.set_peak_threshold(PEAK_THRESH)
        event.set_ingress_threshold(INGRESS_THRESH)

        # gather waveforms of event
        event.gather_waveforms()

        # set Region Of Interest (ROI)
        event.set_ROI((T_MIN, T_MAX))

        # extract calibration.json popt
        json_path = os.path.join(out_path, "calibration.json")
        with open(json_path, "r") as f:
            content = json.load(f)
            linear_popt = content["popt"]

        # set track parameters
        positions=np.array([L*0, L*1, L*2, L*3])
        event.set_track_params(positions=positions, linear_popt=linear_popt)

        # calculate
        event.calculate_peak_and_ingress()
        event.calculate_ingress_matrix()
        event.calculate_delta_t_array()

        try:
            event.calculate_track()
        except Exception as e:
            print(e)
            pass


    def add_run(self, runpath):

        segment_number = self.check_segment_number(runpath)
        for segment in range(1, segment_number+1):
            try:
                print(segment)
                event = Event(runpath, segment)
                self.event_processor(event)

                timestamp = event.get_timestamp()
                angle     = event.get_angle()
                hits      = np.sum(event.get_hit_bools())

                if angle == None:
                    angle = np.nan
                if timestamp == None:
                    timestamp = np.nan

                tup       = (timestamp, angle, hits) #, hits)
                self.data.append(tup)

            except Exception as e:
                print(e)


    # == Get Methods == #

    def get_data(self):
        data = np.array(self.data)
        return data


# --------
# Testing
# --------

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Add src directory to system path
    project_path = os.getcwd().split("/src")[0]
    sys.path.append(project_path)

    # Define path to pdf
    pdf_path      = os.path.join(out_path, "run.pdf")

    # Initialise pdf
    pdf           = PdfPages(pdf_path)

    #run_num = sys.argv[1]

    #runs = [1,2,3,4,5,6,7,8,9,10]
    #runs = [11,12]
    runs = [13,14,15,16]

    run = Run()
    for run_num in runs:
        run_path = os.path.join(lcd_path, f"Run{run_num}")
        run.add_run(run_path)

    data = run.get_data()

    angle_vals = np.array([tup[1] for tup in data])

    # Remove NaN values
    valid_indices = ~np.isnan(angle_vals)
    angle_vals_clean = angle_vals[valid_indices]
    fig, ax = plt.subplots()
    bins = np.arange(-95,95,10)
    ax.hist(angle_vals_clean, bins = bins, edgecolor='black')

    pdf.savefig()

    # Close pdf
    pdf.close()
