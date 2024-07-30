import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


FIGS_DIR = "../figs"


def savefig(fig, name, prefix=FIGS_DIR, **kwargs):
    dst = os.path.join(prefix, name)
    fig.savefig(dst, bbox_inches="tight", transparent=True, **kwargs)
    return dst
