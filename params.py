from pipeline_utils import get_cycle, get_end

# general config
CONFIG_PARAMS = {
    "frame_function":  get_cycle, # the function for extracting the frame
    "z_function":  get_end, # the function for extracting the z level
    "z":  "z1", # the z level of interest for this analysis
    "f_ch":  "ch2", # the functional channel 
    "mc_ch":  "ch1", # the channel to use for motion correction
}


# settings for file loading - no defaults exist, so supply externally
LOAD_PARAMS = {
    #"multi_path" : ['./data/adm977 Mouse 3/20250327_adm977_fructose/20250327_adm977_fructose-170', './data/adm977 Mouse 3/20250327_adm977_glucose/20250327_adm977_glucose-172'], #['./data/adm960 Mouse 1/20241002 fructose-159', './data/adm960 Mouse 1/20241002 glucose-160'],
    "multi_path" : ['./data/adm960 Mouse 1/20241002 fructose-159', './data/adm960 Mouse 1/20241002 glucose-160'],
    "multi_crop" : True,
    "multi_crop_start" : [52, 52],
    "multi_crop_len"  : [307, 307]
} 


# settings for rigid motion correction
MC_PARAMS = {
    "max_shifts": (50, 50),  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    "niter_rig" :  3,          # number of times to perform rigid registration (i'd recommend doing at least 2 especially for these shorter recordings) # number of chunks for parallel processing ()
    "pw_rigid"  : True,       # flag for performing rigid or piecewise rigid motion correction (false for now as we want to asses rigid registration first)
    "shifts_opencv": True ,    # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    "border_nan"   : 'copy',
    "nonneg_movie" : True,
    "use_cuda"  : False,
    "strides"   : (64, 64), # size of patches for nonrigid motion correction in pixels
    "overlaps"  : (32, 32), # number of pixrls of overlap between patches
    "max_deviation_rigid"  : 3, # maximum allowed deviation of any individual patch's registered shift from the rigid shift
    "upsample_factor_grid" : 4
}


# settings for saving rigid motion correction video
MCVID_PARAMS = {
    "anchor": (50,50),  # where to anchor the stim bar    
    "wd" : 0, # width and height of the stim bar
    "ht": 0, 
    "t_stim_st" : 31, # time in seconds of the start of the stim
    "t_stim_end" : 211, # time in seconds of the end of the stim
    "ds_ratio":  0.5, # the fraction of frames to show in the video
    "speed_up":  20,  # how much to speed up the video by
    "frame_period":  0.585 # time in seconds between frames, get this from the xml file
}
MCVID_PARAMS['fr'] = 1/MCVID_PARAMS['frame_period'] 
MCVID_PARAMS['fps'] = MCVID_PARAMS['speed_up'] * MCVID_PARAMS['fr'] * MCVID_PARAMS['ds_ratio']


# settings for identifying rois
IDROI_PARAMS = {
    "method" :'max', 
    "filt" : True, # whether or not to spatially filter the resulting image
    "kern" : 1,      # kernel size of filter 
    "channels" : [[0,0]], 
    "flow_threshold" : 2, 
    "cellprob_threshold" : -1, 
    "diameter" : 15, 
    "model_type" : 'cyto',
    'show_figs' : True   # show figures
}


#set parameters for source extraction
CNMF_PARAMS = {
    'fr': MCVID_PARAMS['fr'],
    'decay_time': 1.8, # this should be set to 1.8 for GCamp6s, but can be changed to 0.4 for faster indicators
    'p': 2, # this must be 2 for GCamp6s, it can be set to 1 for faster indicators
    'nb': 2, # the number of background components, 
    'rf': None, #must be None for seeded mode
    'only_init':False, #must be false for seeded mode
    'min_SNR': 2.0,
    'use_cnn': False, # whether or not to use a convolutional neural network to classify rois as good or bad
    'use_cuda':True, # whether or not to use GPUs

    ## the rest of these are necessary parameters to set for that get ignored
    ## since we are seeding the algorithm for initialization
    'K': 300,  
    'gSig':[10,10]
}