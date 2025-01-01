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
pdf_path      = os.path.join(out_path, 'test-event.pdf')

# Initialise pdf
pdf           = PdfPages(pdf_path)


# ========= BODY ==========

run = sys.argv[1]
seg = int(sys.argv[2])

# Path to Run
run_path = os.path.join(lcd_path, f'Run{run}')

# Initialise event object
event    = Event(run_path, seg)

# Get timestamp
event.read_timestamp()

# Gather waveforms
event.gather_waveforms()

# Set ROI
ROI = (0,80)
event.set_ROI(ROI)

# Get waveform matrix
waveform_matrix = event.get_waveform_matrix()

# Further process to get ingress
event.calculate_peak_and_ingress()

# Calculate ingress matrix
event.calculate_ingress_matrix()

# Get ingress matrix
ingress_matrix = event.get_ingress_matrix()

# Calculate delta_t_array
event.calculate_delta_t_array()

# Get delta array
dt = event.get_delta_t_array()
dt = np.round(dt, 2)

fig, axs = plt.subplots(nrows=4, ncols=1)

for plate_num, plate in enumerate(waveform_matrix):

    try:
        x, y = plate[0].get_data(zipped=False)
        axs[plate_num].plot(x,y, color = 'black')
        ingress = ingress_matrix[plate_num][0]
        axs[plate_num].axvline(ingress, color='black', alpha=0.7)

        x,y = plate[1].get_data(zipped=False)
        axs[plate_num].plot(x,y, color='darkblue', label = rf'$\Delta$t = {dt[plate_num]}')
        ingress = ingress_matrix[plate_num][1]
        axs[plate_num].axvline(ingress, color='darkblue', alpha=0.7)

    except:
        pass

    axs[plate_num].legend()


fig.tight_layout()

pdf.savefig()
plt.close()

# ========END BODY ========


# Close pdf
pdf.close()
