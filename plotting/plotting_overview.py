import matplotlib.pyplot as plt
import matplotlib as mpl


def set_publication_style():
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        pass

    mpl.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 12,

        'font.family': 'Times New Roman',

        'mathtext.fontset': 'stix',
        'font.serif': ['Times New Roman'],
    })

    mpl.rcParams.update({
        'lines.linewidth': 1,
        'axes.linewidth': 0.5,
        'lines.markersize': 3.0,
        'grid.alpha': 0.6,
        'figure.autolayout': True,
        'figure.dpi': 150
    })


def save_for_paper(filename, dpi=600):
    # Added bbox_inches='tight' to prevent cutting off axis labels
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight', dpi=dpi)
    plt.savefig(f"{filename}.png", format='png', bbox_inches='tight', dpi=dpi)
    print(f"Saved {filename}.pdf")