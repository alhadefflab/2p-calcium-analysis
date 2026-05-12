from pipeline import Pipeline
pipeline = Pipeline("adm960-Mouse1")
pipeline.load_data()


from pipeline_funcs import select_subregions, show_subregions, custom_df_f_startend
from tifffile import imread
from caiman.source_extraction.cnmf import cnmf 
import numpy as np
from caiman.base.rois import com
import matplotlib.pyplot as plt

import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

img_path = pipeline.affcorr_filenames[pipeline.f_ch]
img = imread(img_path)
img = img.max(axis=0)

cnm = cnmf.load_CNMF(pipeline.cnmf_file)
center = com(cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1]) 

# guessing these numbers
# TODO put these into a function - returns stim1, stim2 and everything in a dataframe, and img
bline1_start, bline1_end = 0, 52
stim1_start, stim1_end = 52, 307
bline2_start, bline2_end = 307, 359
stim2_start, stim2_end = 359, 614

fl_acc1 = custom_df_f_startend(cnm, bline1_start, bline1_end, method='zscore', use_residuals=True)
fl_acc2 = custom_df_f_startend(cnm, bline2_start, bline2_end, method='zscore', use_residuals=True)

stim1 = fl_acc1[:, stim1_start:stim1_end]   # z-scored
stim2 = fl_acc2[:, stim2_start:stim2_end]

stim1 = fl_acc1[:, bline1_start:stim1_end]   # z-scored
stim2 = fl_acc2[:, bline2_start:stim2_end]

stim1_dict = [{'stim_id': 0, 
               'unit_id': i, 
               'stim_start': stim1_start - bline1_start, 
               'center': center[i,:],
               'response': unit_stim} for i, unit_stim in enumerate(stim1) ]
stim2_dict = [{'stim_id': 1, 
               'unit_id': i, 
               'stim_start': stim2_start - bline2_start, 
               'center': center[i,:],
               'response': unit_stim} for i, unit_stim in enumerate(stim2)]

stim_df = pd.DataFrame(stim1_dict + stim2_dict)



#### old KP stuff ===============
#1.
stim1 = fl_acc1[:, stim1_start:stim1_end]
stim2 = fl_acc2[:, stim2_start:stim2_end]

stim1_median = np.median(stim1, axis=1)
stim2_median = np.median(stim2, axis=1)

stim_idx = stim2_median - stim1_median # Index.cinacalcet is [Stim 2 - Stim 1]
stim_idx[np.maximum(stim1_median, stim2_median)<1.64] = np.nan

center = com(cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1]) 
plt.figure()
plt.imshow(img, cmap='gray',alpha=1)    
plt.scatter(center[:,1],center[:,0],c=stim_idx, cmap='bwr', vmin=-10, vmax=10, alpha=1, s=50, edgecolor='none',linewidth=0.75) # edgecolor='black' for lines around the circle
plt.title('cinacalcet_index')
plt.colorbar()

#2.
area_indices = select_subregions(img, cnm, ['Area 1', 'Area 2']) 
show_subregions(img, cnm, area_indices, ['red', 'yellow'])
stim1area1 = stim1[area_indices[0], :]
stim1area2 = stim2[area_indices[1], :]
### =============================

#4.
# Implement a smoothing algorithm option to the code to deal with artificial peaks 
# so we can determine if spontaneous neuron firing is adding noise to the signal when determining responsive neurons. 


def filter_savgol(unit):
    return savgol_filter(unit['response'], 10, 5)
    

stim_df['response_filtered_savgol'] = stim_df.apply(filter_savgol, axis=1)

### 1
#a.	Write python code for the following two methods to define responsive neurons 
#i.	KP’s method: Median Z-score across the stimulation period (in our case either fructose or glucose infusions) greater than 1.64 from the baseline (median) were responders 
#1.	In the meeting KP said the easiest method is to determine the median z score during the baseline period 1 (frames 1-52) and determine the median z score for the stimulation period 1 (frames 53-358) 
#2.	This will also need to be done for the neuron responses to the second stimulation where the baseline period 2 is frame 359-410 and stimulation period 2 is frame 411-717 
#3.	Then sort the neurons from most activated to least activated and neurons with a medium z -score>1.64 will be determined to be responsive and will be used for further statistical analysis 
#4.	When doing this method, it would be helpful retain individual neuron responses to both stimuli as that will address the central question for this part of the paper 
#ii.	Alan’s method: peaked z-score (calculated as the 10s mean z-scored activity surrounding the max z-scored activity) greater than 2.5 SD above the baseline mean were considered responders 



def is_responder_KP(unit):
    bline_response = unit['response'][:unit['stim_start']]
    stim_response = unit['response'][unit['stim_start']:]

    is_responder = np.median(stim_response) - np.median(bline_response) > 1.64
    return is_responder


def is_responder_Alan(unit, fs):
    bline_mean_response = unit['response'][:unit['stim_start']].mean()
    stim_response = unit['response'][unit['stim_start']:]
    argmax_stim_response = stim_response.argmax()

    win_len = int(5*fs)
    win_start = max(0, argmax_stim_response-win_len)
    win_end = min(len(stim_response), argmax_stim_response+win_len)
    
    peaked_zscore = stim_response[win_start:win_end].mean() 
    std = unit['response'].std()
    is_responder = peaked_zscore - bline_mean_response > 2.5*std
    return is_responder



stim_df['is_responder_KP'] = stim_df.apply(is_responder_KP, axis=1)

fs = 1
stim_df['is_responder_Alan'] = stim_df.apply(lambda unit : is_responder_Alan(unit, fs), axis=1)


###2.	
# Plotting functions to show mean neuron responses to a given stimuli and show within neuron responses to different stimuli 
#a.	Heatmaps to show the change in Z score activity across recording sessions to each stimuli 
#b.	Method to visualize example traces of individual neuron responds to different stimuli 
#c.	Represented images with neurons color coded based on their responses to different stimuli (this code is already written just need to determine how to properly use it)

def plot_response_heatmap(units):
    response = np.stack(units['response'])

    fig = plt.figure()
    ax = plt.gca()

    plt.imshow(response, cmap='plasma')
    ax.set_xlabel('frames')
    ax.set_ylabel('neurons')
    ax.set_aspect(4)
    fig.set_size_inches([7.26, 6.41])

    plt.colorbar(shrink=0.3)


def plot_response(units):
    response = np.stack(units['response'])
    
    plt.figure()
    plt.plot(response.T)
    plt.xlabel('frames')
    plt.ylabel('response')
    


def plot_rep_image(units, img):
    center = np.stack(units['center'])

    response = units.apply(lambda unit : np.median(unit['response'][unit['stim_start']:]), axis=1)
    
    plt.figure()
    plt.imshow(img, cmap='gray',alpha=1)    
    plt.scatter(center[:,1], center[:,0], c=response, cmap='plasma', alpha=1, s=50, edgecolor='none',linewidth=0.75)
    plt.colorbar()


condition = stim_df['stim_id']==0
units = stim_df.loc[condition]
plot_response_heatmap(units)


condition2 = (stim_df['stim_id']==0) & (stim_df['unit_id']==3)
units2 = stim_df.loc[condition2]
plot_response(units2)

plot_rep_image(units, img)


###3
#3.	Grouping neurons with similar temporal dynamics such as using a K mean clustering algorithm 
#a.	Some suggested features or parameters to group the neurons upon 
#i.	Intensity of activation 
#ii.	Binning the data over periods of time such as every 10 seconds and comparing the mean activation across the bins 

def bin_responses(unit, num_bins):
    bin_values = np.array_split( unit['response'][unit['stim_start']:], num_bins )
    bin_values_means = []
    for values in bin_values:
        bin_values_means.append(values.mean())

    return bin_values_means    


def clustering_kmeans(units):
    num_bins = 50 # None
    num_clusters = 5 # None
    response_binned = units.apply(lambda unit : bin_responses(unit, num_bins), axis=1)
    
    X = np.stack(response_binned)

    kmeans = KMeans(n_clusters=num_clusters).fit(X)

    return kmeans.labels_
    

condition = stim_df['stim_id']==0
units = stim_df.loc[condition]
clustering_kmeans(units)



