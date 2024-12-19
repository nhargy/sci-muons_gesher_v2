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
pdf_path      = os.path.join(out_path, 't-waveform.pdf')

# Initialise pdf
pdf           = PdfPages(pdf_path)


# Waveform parameters
"""run = 5
scope = 1
seg = 17
ch = 1"""

# Waveform parameters (sys.argv)
if sys.argv[1] == '?':
    print("<Run> <scope> <segment> <channel>")
    exit()

run = sys.argv[1]
scope = sys.argv[2]
seg = sys.argv[3]
ch = sys.argv[4]

csvfile = os.path.join(lcd_path, f'Run{run}' ,f'scope-{scope}-seg{seg}-ch{ch}.csv')

# Initialise WaveForm object
wf = WaveForm(csvfile)
wf.read_from_csv()
wf.rescale(xfactor=1e9, yfactor=-1e3)

bins = np.arange(-49.5, 299.5,1)

wf.calculate_baseline(bins=bins)
print(wf.get_baseline())

wf.zero_baseline()
wf.calculate_baseline(bins=bins)
print(wf.get_baseline())

wf.smooth()

x,y = wf.get_data(zipped=False)

fig, ax = plt.subplots()

ax.plot(x,y)

pdf.savefig()
plt.close()

"""
x,y = wf.get_data(zipped=False, raw=False)

bins = np.arange(-49.5, 299.5,1)


hist, bin_edges = np.histogram(y[:int(len(y)/1)], bins=bins)
bin_points      = bin_edges[:-1] + np.diff(bin_edges)/2

indexes = []
for idx, elm in enumerate(hist):
    if elm == 0:
        indexes.append(idx)

# remove zeros
hist = np.delete(hist, indexes, axis=None)
bin_points = np.delete(bin_points, indexes, axis=None)

# remove high values
cut = int(len(hist)*1)
hist = hist[:cut]
bin_points = bin_points[:cut]

print(bin_points, hist)

try:
    popt, pcov      = curve_fit(gaussian, bin_points, hist, p0=[100,0,10])
except:
    pass

fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,4))

ax1.plot(x,y)
ax1.set_ylim(-25,25)

ax2.hist(y, bins=bins)
ax2.scatter(bin_points, hist)

try:
    x_vals = np.linspace(-100,100,500)
    ax2.plot(x_vals, gaussian(x_vals, *popt), label=f'BL={np.round(popt[1],2)}')
    #ax2.set_xlim(-25,50)
except:
    pass

ax2.legend()

pdf.savefig()
plt.close()
"""

# Close pdf
pdf.close()

