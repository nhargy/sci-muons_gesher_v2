import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# Add src directory to system path
project_path = os.getcwd().split('/src')[0]
sys.path.append(project_path)

try:
    from src.models.waveform import WaveForm
    from src.utils.functions import gaussian
except Exception as e:
    print("Failed to import local modules:")
    print(e)

# Define important paths
lcd_path  = os.path.join(project_path, 'lcd')
out_path  = os.path.join(project_path, 'out')
plt_path  = os.path.join(project_path, 'plt')

# Define path to pdf
pdf_path      = os.path.join(out_path, 'test-waveform.pdf')

# Initialise pdf
pdf           = PdfPages(pdf_path)


# Waveform parameters
"""run = 5
scope = 1
seg = 17
ch = 1"""

# Waveform parameters (sys.argv)
if sys.argv[1] == '?':
    print("<Run> <scope> <channel> <segment>")
    exit()

run = sys.argv[1]
scope = sys.argv[2]
ch = sys.argv[3]
seg = sys.argv[4]

csvfile = os.path.join(lcd_path, f'Run{run}' ,f'scope-{scope}-seg{seg}-ch{ch}.csv')

# Initialise WaveForm object
wf = WaveForm(csvfile)
wf.read_from_csv()
wf.rescale(xfactor=1e9, yfactor=-1e3)

bins = np.arange(-49.5, 299.5,1)

wf.calculate_baseline(bins=bins)

wf.zero_baseline()
wf.calculate_baseline(bins=bins)

wf.smooth()

a = 150; b = 280; th = 25
wf.detect_main_peak((a,b), 140)
peak_idx, peak_val = wf.get_main_peak()

wf.identify_ingress(th)
ingress_idx, ingress_time_val = wf.get_ingress()

x,y = wf.get_data(zipped=False)

fig, ax = plt.subplots()

ax.plot(x,y)
ax.axvline(x[a], color='black')
ax.axvline(x[b], color='black')
ax.axvspan(x[a], x[b], color = 'lawngreen', alpha = 0.25, label = 'ROI')
ax.axhline(th, color = 'grey', linestyle = '--', label='Ingress threshold')
try:
    ax.scatter(x[peak_idx], y[peak_idx], color='red',s=25, label = f'Peak: {np.round(peak_val,2)}mV', zorder=2)
    ax.scatter(x[ingress_idx], y[ingress_idx], color='orange', s=25, label=f'Ingress: {np.round(ingress_time_val,2)}ns',zorder=2)
except:
    pass

ax.grid('on', linestyle='--', alpha=0.5)
ax.legend()

pdf.savefig()
plt.close()


# Close pdf
pdf.close()
