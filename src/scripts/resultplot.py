import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# Add src directory to system path
project_path = os.getcwd().split('/src')[0]
sys.path.append(project_path)

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

try:
    #from src.models.run import Run
    from src.utils.functions import decay
    from src.utils.functions import hist_to_scatter
    from src.utils.functions import gaussian
except Exception as e:
    print("Failed to import local modules:")
    print(e)
    
# Define important paths
lcd_path  = os.path.join(project_path, 'lcd')
out_path  = os.path.join(project_path, 'out')
plt_path  = os.path.join(project_path, 'plt')

# Load processed data
timestamps1 = np.load(os.path.join(out_path, "a1.npy"))
diff1       = np.load(os.path.join(out_path, "b1.npy"))
angles1     = np.load(os.path.join(out_path, "c1.npy"))

timestamps2 = np.load(os.path.join(out_path, "a2.npy"))
diff2       = np.load(os.path.join(out_path, "b2.npy"))
angles2     = np.load(os.path.join(out_path, "c2.npy"))

timestamps3 = np.load(os.path.join(out_path, "a3.npy"))
diff3       = np.load(os.path.join(out_path, "b3.npy"))
angles3     = np.load(os.path.join(out_path, "c3.npy"))


# ======================

size = 20
bins = np.arange(-90 ,90+size*32,size)
off  = 200

fig, ax = plt.subplots(figsize=(15,8))

xticks1 = np.array([-45, 0, 45])

xticks = np.concatenate([xticks1, np.add(xticks1,off), np.add(xticks1,2*off)], axis = 0)
xlabels = np.concatenate([xticks1, xticks1, xticks1], axis=0)

ax.tick_params(axis='both', which='major', labelsize=25)

hist1, _,_ = ax.hist(angles1, bins=bins, density =True, edgecolor = 'black')
hist2, _,_ = ax.hist(angles2 + off, bins = bins, density =True, edgecolor = 'black', color='darkgreen')
hist3, _,_ = ax.hist(angles3 + 2*off, bins = bins, density =True, edgecolor = 'black', color='firebrick')

bin_mids = bins[:-1] + np.diff(bins)/2
popt, pcov = curve_fit(gaussian, bin_mids, hist1, p0=[0.1, 0, 20])
x_vals = np.linspace(-140,140, 250)
ax.plot(x_vals, gaussian(x_vals, *popt), color='blue', lw = 4, alpha = 0.75)
print(popt[1], pcov[1][1]**2)


popt, pcov = curve_fit(gaussian, bin_mids, hist2, p0=[0.1, off, 20])
x_vals = np.linspace(-140,140, 250) + off
ax.plot(x_vals, gaussian(x_vals, *popt), color='limegreen', lw = 4, alpha = 0.75)
print(popt[1] - off, pcov[1][1]**2)

popt, pcov = curve_fit(gaussian, bin_mids, hist3, p0=[0.1, 2*off, 20])
x_vals = np.linspace(-140,140, 250) + 2*off
ax.plot(x_vals, gaussian(x_vals, *popt), color = 'red', lw = 4, alpha = 0.75)
print(popt[1] - 2*off, pcov[1][1]**2)

ax.axvline(0*off, color = 'black', linestyle = '--', lw = 3, zorder = 2)
ax.axvline(1*off, color = 'black', linestyle = '--', lw = 3, zorder = 2)
ax.axvline(2*off, color = 'black', linestyle = '--', lw = 3, zorder = 2)

off_2 = int(off/2)
ax.axvspan(0-off_2-10, 0+off_2, color = 'cyan', alpha = 0.1, zorder=1)
ax.axvspan(off-off_2, off+off_2, color = 'green', alpha = 0.1, zorder=1)
ax.axvspan(2*off-off_2, 2*off+off_2+10, color = 'maroon', alpha = 0.1, zorder=1)

ax.set_xticks(xticks, labels = xlabels)
ax.set_yticks([])
ax.set_ylabel("Relative Frequency [A.U.]", fontsize = 25)

ax.text(-95, 0.0117, 'Sea-level', style='oblique', fontsize = 26)
ax.text(-95 + off, 0.0117, 'JS S-N', style='oblique', fontsize = 26)
ax.text(-95 + 2*off, 0.0117, 'JS W-E', style='oblique', fontsize = 26)

ax.set_xlabel("Incidence Angle [Degrees]", fontsize = 25, labelpad = 10)

ax.grid("on")

ax.set_xlim(-100, 100+2*off)

fig.tight_layout()

plt.show()

# ======================

fig, ax = plt.subplots(figsize=(12,10))

ax.tick_params(axis='both', which='major', labelsize=25)

time_bins = np.arange(0, 0.2, 0.0075)
x, y   = hist_to_scatter(diff1, bins = time_bins, density=True)
ax.scatter(x,y, s = 100)
popt, pcov   = curve_fit(decay, x, y, p0=[25, 0.05])
x_vals = np.linspace(x[0], x[-1], 1000)
ax.plot(x_vals, decay(x_vals, *popt), label = rf'Sea-level', lw = 3)

ax.grid("on")

ax.legend(fontsize = 25)
ax.set_xlabel(rf"$\Delta t$ [ns]", fontsize = 25, labelpad = 15)
ax.set_ylabel("Realtive Frequency [A.U.]", fontsize = 25, labelpad = 15)

ax.set_yticks([])

fig.tight_layout()

plt.show()

fig, ax = plt.subplots(figsize=(12,10))

ax.tick_params(axis='both', which='major', labelsize=25)

time_bins = np.arange(0, 600, 15)
x, y   = hist_to_scatter(diff2, bins = time_bins, density=True)
ax.scatter(x,y, s = 100, color = 'darkgreen')
popt, pcov   = curve_fit(decay, x, y, p0=[25, 100])
x_vals = np.linspace(x[0], x[-1], 1000)
ax.plot(x_vals, decay(x_vals, *popt), label = rf'JS S-N', lw = 3, color = 'darkgreen')

x, y   = hist_to_scatter(diff3, bins = time_bins, density=True)
ax.scatter(x,y, s = 100, color = 'maroon')
popt, pcov   = curve_fit(decay, x, y, p0=[25, 100])
x_vals = np.linspace(x[0], x[-1], 1000)
ax.plot(x_vals, decay(x_vals, *popt), label = rf'JS W-E', lw = 3, color = 'maroon')

ax.grid("on")

ax.set_yticks([])

ax.legend(fontsize = 25)
ax.set_xlabel(rf"$\Delta t$ [ns]", fontsize = 25, labelpad = 15)
ax.set_ylabel("Realtive Frequency [A.U.]", fontsize = 25, labelpad = 15)

fig.tight_layout()

plt.show()


max1  = np.argmax(hist1)
max2  = np.argmax(hist2)
max3  = np.argmax(hist3)

print(angles1)
print()
print(angles2)
print()
print(angles3)
