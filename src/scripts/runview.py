# runview.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import json

# Add src directory to system path
project_path = os.getcwd().split('/src')[0]
sys.path.append(project_path)

try:
    from src.models.event import Event
    from src.utils.functions import gaussian
except Exception as e:
    print("Failed to import local modules:")
    print(e)

# Define important paths
lcd_path  = os.path.join(project_path, 'lcd')
out_path  = os.path.join(project_path, 'out')
plt_path  = os.path.join(project_path, 'plt')

# Define path to pdf
pdf_path      = os.path.join(out_path, 'runview.pdf')

# Initialise pdf
pdf           = PdfPages(pdf_path)


""" ========== """
""" == BODY == """
""" ========== """

def event_processor(run_path, seg):

    # define event
    event = Event(run_path, seg)

    # Event parameters
    PEAK_THRESH = 125
    INGRESS_THRESH = 25

    if run < 17:
        # Expected signal around zero
        T_MIN = -50
        T_MAX = 75
    else:
        # Expected signal after zero
        T_MIN = 1
        T_MAX = 85

    # timestamp
    event.read_timestamp()
    timestamp = event.get_timestamp()

    # set peak and ingress thresholds
    PEAK_THRESH = 125
    INGRESS_THRESH = 25
    event.set_peak_threshold(PEAK_THRESH)
    event.set_ingress_threshold(INGRESS_THRESH)

    # gather waveforms of event
    event.gather_waveforms()

    # set Region Of Interest (ROI)
    event.set_ROI((T_MIN, T_MAX))

    # extract calibration.json popt
    json_path = os.path.join(out_path, 'calibration.json')
    with open(json_path, 'r') as f:
        content = json.load(f)
        linear_popt = content["popt"]

    # set track parameters
    L = 43 #cm
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

    return event

# Take event parameters as input
if sys.argv[1] == '?':
    print("<Run>")

run = int(sys.argv[1])

# Path to Run
run_path = os.path.join(lcd_path, f'Run{run}')

# loop through all events in run and collect
events = []
for seg in range(1,1001):
    try:
        event = event_processor(run_path, seg)
        events.append(event)
    except:
        print('no event')
        events.append(np.nan)
    print(seg)

angles = []
timestamps = []
count=0
for event in events:
    try:
        angle = event.angle
        if angle != None:
            angles.append(angle)
    except:
        pass

    try:
        timestamp = event.get_timestamp()
        if timestamp != None:
            timestamps.append(timestamp)
    except:
        pass

bins = np.arange(-85,85, 10)
fig, ax = plt.subplots()
ax.hist(angles, bins=bins, edgecolor='black')

rate = np.round(len(timestamps) / timestamps[-1],5)
print(len(timestamps), timestamps[-1])

ax.set_title(f'Run{run}, Rate: {rate} events/sec')

fig.tight_layout()

pdf.savefig()
plt.close()


""" ========= """
""" == END == """
""" ========= """

pdf.close()
