# calibrate.py
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
    from src.utils.functions import gaussian
except Exception as e:
    print("Failed to import local modules:")
    print(e)

# Define important paths
lcd_path  = os.path.join(project_path, 'lcd')
out_path  = os.path.join(project_path, 'out')
plt_path  = os.path.join(project_path, 'plt')

# Define path to pdf
pdf_path      = os.path.join(out_path, 'calibrate.pdf')

# Initialise pdf
pdf           = PdfPages(pdf_path)

""" ========== """
""" == BODY == """
""" ========== """

# Event parameters
PEAK_THRESH = 125
INGRESS_THRESH = 25

# Plot parameters
LABELFONT = 14
TITLEFONT = 16

# Expected signal after zero
T_MIN = 1
T_MAX = 85

labels = ['L', 'CL', 'C', 'CR', 'R']
colors = ['purple', 'blue', 'green', 'darkorange', 'red']
p0s    = [[25, -5, 2],[25, -2, 2],[25, 0, 2],[25, 2, 2],[25, 5, 2]]

x_positions = [24, 48, 72, 96, 120]
pos_err     = 5 #cm

N = 120

def collect_dts(runs):
    dts = []
    for run in runs:
        print(f'RUN {run}'); print()
        run_path = os.path.join(lcd_path, f'Run{run}')
        for seg in range(1, N+1):
            try:
                event = Event(run_path, seg)

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

                # calculate
                event.calculate_peak_and_ingress()
                event.calculate_ingress_matrix()
                event.calculate_delta_t_array()

                # get data
                waveform_matrix = event.get_waveform_matrix()
                ingress_matrix  = event.get_ingress_matrix()
                delta_t_array   = np.round(event.get_delta_t_array(),2)

                dt = delta_t_array[1]
                dts.append(np.round(dt,2))

            except:
                pass

    dts = np.array(dts)
    dts = np.round(dts[~np.isnan(dts)],2)

    mean = np.mean(dts)
    std  = np.std(dts)
    z_scores = (dts - mean)/std
    threshold = 3
    dts = dts[np.abs(z_scores) < threshold]

    return dts

L  = collect_dts([17,18,19])
CL = collect_dts([20,21])
C  = collect_dts([22,23,24])
CR = collect_dts([25,26,27])
R  = collect_dts([28,29,30])

hist_arr = [L, CL, C, CR, R]
mean_arr = []
#sig_arr  = []

fig, ax = plt.subplots(figsize=(8,5))
bins = np.arange(-20.5,20.5,1)
for idx, h in enumerate(hist_arr):
    hist, bin_edges = np.histogram(h, bins=bins, density=True)
    bin_mids = bin_edges[:-1] + np.diff(bin_edges)/2
    popt, pcov = curve_fit(gaussian, bin_mids, hist, p0=p0s[idx])

    mean_arr.append(popt[1])
    #sig_arr.append(np.abs(popt[2])/np.sqrt(N))

    x_vals = np.linspace(-20,20,200)
    ax.plot(x_vals, gaussian(x_vals, *popt), label = labels[idx], color = colors[idx])
    ax.hist(h, bins = bins, color = colors[idx], alpha=0.15, density=True)
    ax.legend()

ax.set_xlabel(r"$\Delta t$ [ns]", fontsize = LABELFONT)
ax.set_ylabel("Relative Frequency", fontsize = LABELFONT)
ax.grid("on", linestyle='--', alpha=0.75)

plt.tight_layout()
pdf.savefig()

save_path = os.path.join(plt_path, "calibrate_gaussians.png")
plt.savefig(save_path)

plt.show()
plt.close()

#popt, pcov = curve_fit(linear, x_positions, mean_arr, sigma=sig_arr, p0 = [0.1, -10])
popt, pcov = curve_fit(linear, mean_arr, x_positions, sigma=pos_err, p0 = [0.1, -10])
fig, ax = plt.subplots(figsize=(8,5))
t_vals = np.linspace(-12,12)
ax.plot(t_vals, linear(t_vals, *popt), label = 'Linear Fit', color='black')
for idx, mean in enumerate(mean_arr):
    ax.errorbar(mean, x_positions[idx], yerr=pos_err, capsize=4, fmt='o', label = labels[idx], color=colors[idx])
    #ax.scatter(mean, x_positions[idx], label = labels[idx], color=colors[idx], zorder=2)

ax.set_yticks([0,24,48,72,96,120,144], labels=[0,24,48,72,96,120,144])
ax.set_ylabel("x [cm]", fontsize = LABELFONT)
ax.set_xlabel(r"$\Delta t$ [ns]", fontsize = LABELFONT)
ax.legend()
ax.grid("on", linestyle='--', alpha=0.75)

plt.tight_layout()
pdf.savefig()

save_path = os.path.join(plt_path, "calibrate_fit.png")
plt.savefig(save_path)

plt.show()
plt.close()


# =======================
# SAVE LINEAR FIT TO JSON
# =======================
json_path = os.path.join(out_path, 'calibration.json')
with open(json_path, 'w') as json:
    json.write('{\n')
    json.write(f'\"popt\": [{popt[0]}, {popt[1]}], \n')
    json.write(f'\"pcov\": [[{pcov[0][0]}, {pcov[0][1]}], [{pcov[1][0]}, {pcov[1][1]}]] \n')
    json.write('}')


""" ========= """
""" == END == """
""" ========= """

# Close pdf
pdf.close()
