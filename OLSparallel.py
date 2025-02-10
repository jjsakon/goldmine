# General
import sys
import os.path as op
from time import time
from collections import OrderedDict as od
from glob import glob
import itertools
import warnings
from importlib import reload

# Scientific
import numpy as np
import pandas as pd
pd.options.display.max_rows = 200
pd.options.display.max_columns = 999

# Stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.utils.fixes import loguniform
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import patsy

# Plots
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['grid.linewidth'] = 0.1
mpl.rcParams['grid.alpha'] = 0.75
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
colors = ['1f77b4', 'd62728', '2ca02c', 'ff7f0e', '9467bd', 
          '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', colors)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.formatter.offset_threshold'] = 2
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelpad'] = 8
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['legend.loc'] = 'upper right'
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.figsize'] = (10, 4) 
mpl.rcParams['figure.subplot.wspace'] = 0.25 
mpl.rcParams['figure.subplot.hspace'] = 0.25 
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.fonttype'] = 42

# Personal
# sys.path.append('/home1/dscho/code/general')
# sys.path.append('/home1/dscho/code/projects/manning_replication')
# sys.path.append('/home1/dscho/code/projects')
sys.path.append('/home1/john/Goldmine/general')
sys.path.append('/home1/john/Goldmine/general/cluster_helper')
sys.path.append('/home1/john/Goldmine/time_cells_goldmine')
# %load_ext autoreload
# %autoreload
import data_io as dio
import array_operations as aop
from helper_funcs import *
from eeg_plotting import plot_trace, plot_trace2
import spike_sorting, spike_preproc, events_preproc, events_proc, time_bin_analysis, remapping, pop_decoding, time_cell_plots
from goldmine_replay import place_cells
from cluster_helper.cluster import cluster_view

data_dir = '/data7/goldmine'
proj_dir = '/home1/dscho/projects/time_cells'



# Get sessions.
sessions = np.unique([op.basename(f).split('-')[0] 
                      for f in glob(op.join(data_dir, 'analysis', 'events', '*-Events.pkl'))])
print('{} subjects, {} sessions'.format(len(np.unique([x.split('_')[0] for x in sessions])), len(sessions)))


# Get neurons to process.
fpath = op.join(proj_dir, 'analysis', 'unit_to_behav', '{}-Encoding_Retrieval-ols_model_pairs.pkl')
pop_spikes = pop_decoding.load_pop_spikes()
neurons = [neuron for neuron in pop_spikes.neurons if not op.exists(fpath.format(neuron))]
print('{} neurons to process'.format(len(neurons)))

task_id = int(sys.argv[1])

mod_pairs, ols_weights = time_bin_analysis.run_ols_nav(neurons[task_id],
                                                           n_perm=1000,
                                                           alpha=0.05,
                                                           save_output=True,
                                                           overwrite=False)
    
