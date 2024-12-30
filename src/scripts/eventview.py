# eventview.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# Add src directory to system path
project_path = os.getcwd().split('/src')[0]
sys.path.append(project_path)

try:
    from src.models.event import Event
    from src.utils.functions import linear
except Exception as e:
    print("Failed to import local modules:")
    print(e)

# Define important paths
lcd_path  = os.path.join(project_path, 'lcd')
out_path  = os.path.join(project_path, 'out')
plt_path  = os.path.join(project_path, 'plt')

# Define path to pdf
pdf_path      = os.path.join(out_path, 'eventview.pdf')

# Initialise pdf
pdf           = PdfPages(pdf_path)

""" ========== """
""" == BODY == """
""" ========== """

# Take event parameters as input
if sys.argv[1] == '?':
    print("<Run> <Segment>")

run = int(sys.argv[1])
seg = int(sys.argv[2])

# Path to Run
run_path = os.path.join(lcd_path, f'Run{run}')

# Initialise event
event = Event(run_path, seg)


""" == Event Processes == """

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

# calculate
event.calculate_peak_and_ingress()
event.calculate_ingress_matrix()
event.calculate_delta_t_array()

# get data
waveform_matrix = event.get_waveform_matrix()
ingress_matrix  = event.get_ingress_matrix()
delta_t_array   = np.round(event.get_delta_t_array(),2)


""" == Plot == """

wf_colors = [['indigo', 'firebrick'],
             ['forestgreen', 'magenta'],
             ['deepskyblue', 'sienna'],
             ['darkgreen', 'darkorange']
             ]

labels    = [['Scope 1; Ch1', 'Scope 1; Ch2'],
             ['Scope 1; Ch3', 'Scope 1; Ch4'],
             ['Scope 2; Ch1', 'Scope 2; Ch2'],
             ['Scope 2; Ch3', 'Scope 2; Ch4']
             ]


# Plot raw waveforms
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True, sharey=True)

axs_flat = axs.flatten()

for plate_num, plate in enumerate(waveform_matrix):
    for wf_num, wf in enumerate(plate):
        try:
            x, y = wf.get_data(zipped=False, raw=True)
            axs_flat[plate_num].plot(x,y, color = wf_colors[plate_num][wf_num], label = labels[plate_num][wf_num])
        except:
            pass

    axs_flat[plate_num].set_title(f"Plate {plate_num + 1}")
    axs_flat[plate_num].grid("on", linestyle = '--', alpha = 0.5)
    axs_flat[plate_num].legend()

fig.supxlabel("Time [sec]")
fig.supylabel("Volatge [V]")
fig.suptitle(f"RAW -> Run{run} seg{seg}; timestamp: {np.round(timestamp,3)} sec")
fig.tight_layout()

pdf.savefig()
plt.close()

# Plot processed waveforms
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True, sharey=True)

axs_flat = axs.flatten()

for plate_num, plate in enumerate(waveform_matrix):
    for wf_num, wf in enumerate(plate):
        try:
            x, y = wf.get_data(zipped=False, raw=False)
            axs_flat[plate_num].plot(x,y, color = wf_colors[plate_num][wf_num], label = labels[plate_num][wf_num])

            ingress = ingress_matrix[plate_num][wf_num]
            axs_flat[plate_num].axvline(ingress, color = wf_colors[plate_num][wf_num], linestyle = '--', alpha = 0.75, linewidth=1, zorder=2)
        except:
            pass

    axs_flat[plate_num].axvspan(T_MIN, T_MAX, color='lawngreen', alpha = 0.2, label = 'ROI')

    axs_flat[plate_num].set_title(rf"Plate {plate_num + 1}; $\Delta t$ = {delta_t_array[plate_num]}ns")
    axs_flat[plate_num].grid("on", linestyle = '--', alpha = 0.5)
    axs_flat[plate_num].legend()

fig.supxlabel("Time [ns]")
fig.supylabel("Volatge [mV]")
fig.suptitle(f"Processed -> Run{run} seg{seg}; timestamp: {np.round(timestamp,3)} sec")
fig.tight_layout()

pdf.savefig()
plt.close()


""" ========= """
""" == END == """
""" ========= """

# Close pdf
pdf.close()