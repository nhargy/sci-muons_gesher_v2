# runview.py
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
    from src.models.run import Run
    from src.utils.functions import gaussian
    from src.utils.functions import decay
    from src.utils.functions import hist_to_scatter
    from src.utils.functions import remove_nans
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

# == Functions == #
def make_run_figs(runs, time_bins, time_p0, savefig = None, rate_array=None):

    #colors = ['blue', 'darkred', 'magenta']
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    run = Run()

    for run_num in runs:
        print(f" => Processing Run{run_num}")
        run_path = os.path.join(lcd_path, f"Run{run_num}")
        run.add_run(run_path)

    data = run.get_data()

    angles_all = remove_nans(np.array([tup[1] for tup in data]))
    angles_3   = remove_nans(np.array([tup[1] for tup in data if np.sum(tup[2])==3]))
    angles_4   = remove_nans(np.array([tup[1] for tup in data if np.sum(tup[2])==4]))

    timestamps_all = remove_nans(np.array([tup[0] for tup in data]))
    timestamps_3   = remove_nans(np.array([tup[0] for tup in data if np.sum(tup[2]==3)]))
    timestamps_4   = remove_nans(np.array([tup[0] for tup in data if np.sum(tup[2]==4)]))

    diff   = np.diff(timestamps_all)
    diff_3 = np.diff(timestamps_3)
    diff_4 = np.diff(timestamps_4)

    # Timestamps and Charactersitic Decay
    stretch = 1.6
    x, y   = hist_to_scatter(diff, bins = time_bins, density=True)
    x3, y3 = hist_to_scatter(diff_3, bins = time_bins*stretch, density=True)
    x4, y4 = hist_to_scatter(diff_4, bins = time_bins*stretch, density=True)

    # Fitting
    popt, pcov   = curve_fit(decay, x, y, p0=time_p0)
    #dtau         = np.round(pcov[1][1]**2,5)
    
    popt3, pcov3 = curve_fit(decay, x3, y3, p0=time_p0)
    #dtau3        = np.round(pcov3[1][1]**2,5)
    
    popt4, pcov4 = curve_fit(decay, x4, y4, p0=time_p0)
    #dtau4        = np.round(pcov4[1][1]**2,5)


    fontsize=14
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize = (8,10))

    ax1.scatter(x,y, color = colors[0])
    ax1.scatter(x3,y3, color = colors[1], alpha = 0.4)
    ax1.scatter(x4,y4, color = colors[2], alpha = 0.4)

    x_vals = np.linspace(x[0], x[-1], 1000)
    ax1.plot(x_vals, decay(x_vals, *popt), color = colors[0], label = rf'$\tau = ${np.round(popt[1],3)}')
    ax1.plot(x_vals*stretch, decay(x_vals*stretch, *popt3), color = colors[1], label = rf'$\tau_3 = ${np.round(popt3[1],3)}')
    ax1.plot(x_vals*stretch, decay(x_vals*stretch, *popt4), color = colors[2], label = rf'$\tau_4 = ${np.round(popt4[1],3)}')

    ax1.set_xlabel(r"$\Delta t$ [seconds]", fontsize=fontsize)
    ax1.set_ylabel("Relative Frequency", fontsize=fontsize)
    ax1.legend(fontsize=fontsize)
    ax1.grid("on", color="grey", linestyle = "--")

    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)

    ax1.set_title(r"Charactersitic Time ($\tau$) Between Events", fontsize=fontsize)

    m_all = np.round(np.mean(angles_all),2)
    std_all = np.round(np.std(angles_all),2)

    m_3 = np.round(np.mean(angles_3),2)
    std_3 = np.round(np.std(angles_3),2)

    m_4 = np.round(np.mean(angles_4),2)
    std_4 = np.round(np.std(angles_4),2)

    bins = np.arange(-97.5,97.5+15,15)
    ax2.hist(angles_all, bins = bins, density = False, label = rf'$\theta = ${m_all} $\pm$ {std_all}', color = colors[0], edgecolor="black")
    ax2.hist(angles_3, density = False, bins = bins, label = rf'$\theta_3 = ${m_3} $\pm$ {std_3}', color = colors[1], alpha = 1, zorder=2, edgecolor="black")
    ax2.hist(angles_4, density = False, bins = bins, label = rf'$\theta_4 = ${m_4} $\pm$ {std_4}', color = colors[2], alpha = 1, zorder=3, edgecolor="black")

    ax2.set_xlabel("Incidence Angle [Degrees]", fontsize=fontsize)
    ax2.set_ylabel("Frequency", fontsize=fontsize)
    ax2.legend(fontsize=15)
    ax2.grid("on", color="grey", linestyle = "--")

    ax2.set_title("Angular Distribution", fontsize=fontsize)

    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)

    fig.tight_layout()
    pdf.savefig()

    if savefig != None:
        save_path = os.path.join(plt_path, f"{savefig}.png")
        plt.savefig(save_path, dpi=350)

    plt.show()
    plt.close()

    if rate_array != None:
        rate, drate = run.get_rate()
        rate_array.append(rate)
        drate_array.append(drate)


    return timestamps_all, diff, angles_all



""" ========== """
""" == BODY == """
""" ========== """

rate_array = []
drate_array = []

#runs = [0]
#time_bins = np.arange(0, 0.2, 0.007)
#make_run_figs(runs, time_bins, [25, 0.05], rate_array=rate_array) #, savefig = "runview_TLV0")

runs = [0,1,2,3,4,5,6,7,8,9,10]
time_bins = np.arange(0, 0.2, 0.0075)
a1, b1, c1 = make_run_figs(runs, time_bins, [25, 0.05], rate_array=rate_array) #, savefig = "runview_TLV1")

runs = [11,12]
time_bins = np.arange(0, 600, 15)
a2, b2, c2 = make_run_figs(runs, time_bins, [25, 100], rate_array=rate_array) #, savefig = "runview_VOS0")

runs = [13,14,15,16]
time_bins = np.arange(0, 600, 15)
a3, b3, c3 = make_run_figs(runs, time_bins, [25, 100], rate_array=rate_array) #, savefig = "runview_VOS1")

np.save(os.path.join(out_path, "a1.npy"), a1)
np.save(os.path.join(out_path, "a2.npy"), a2)
np.save(os.path.join(out_path, "a3.npy"), a3)

np.save(os.path.join(out_path, "b1.npy"), b1)
np.save(os.path.join(out_path, "b2.npy"), b2)
np.save(os.path.join(out_path, "b3.npy"), b3)

np.save(os.path.join(out_path, "c1.npy"), c1)
np.save(os.path.join(out_path, "c2.npy"), c2)
np.save(os.path.join(out_path, "c3.npy"), c3)



""" ========= """
""" == END == """
""" ========= """

pdf.close()
