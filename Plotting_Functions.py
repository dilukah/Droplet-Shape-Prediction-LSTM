import matplotlib.pyplot as plt
import numpy as np
def ComparisonPolarPlot(theta, r_1, r_2, zero_location="SE"):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location(zero_location)
    ax.plot(theta, r_1, linestyle='-', marker='+', label = 'r_1', color = 'b')
    ax.plot(theta, r_2, linestyle='--', marker='x', label = 'r_2', color = 'r')
    ax.legend()


def MultipleComparisonPolarPlots(theta, r_1, r_2, first_plot_id = 0, nrows=3, ncols=8, wspace= 0.1, hspace=-0.7 , zero_location="SE", r_1_legend = "r1", r_2_legend = "r2"):
    """Plot multiple polar plots (subplots) with comparison of two plots in each.
        Max number of plots will be nrows*ncols, 
        Starting first plot can be freely choosen. 
        Make sure you have that required number rows in the input (theta, r_1, r_2)
        """
    fig_scale = 1.5
    fig, axs = plt.subplots(nrows, ncols,figsize=(fig_scale*ncols,fig_scale*nrows) ,subplot_kw={'projection': 'polar'})
    plot_id = first_plot_id
    for ax in axs.reshape(-1):
        ax.set_theta_zero_location(zero_location)
        ax.plot(theta, r_1[plot_id,:],'b--',label =r_1_legend)
        ax.plot(theta, r_2[plot_id,:],'r-.',label =r_2_legend)    
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.text(np.radians(0+10),ax.get_rmax()/4.,str(plot_id+1), rotation=0,ha='center',va='center')
        plot_id = plot_id+1

    plt.tight_layout()
    plt.subplots_adjust(wspace = wspace, hspace = hspace)
    ax.legend(bbox_to_anchor=(1.05, 1.05))
