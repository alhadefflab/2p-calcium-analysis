import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from bokeh.layouts import row, column


import bokeh.plotting as bpl
from bokeh.models import Button, CustomJS, ColumnDataSource, Range1d, LabelSet, LinearColorMapper
from bokeh.server.server import Server
from bokeh.io import curdoc

import matplotlib as mpl
import matplotlib.cm as cm

import caiman
from caiman.base.rois import com
from visualizationKB import get_contours

import glob
from caiman.source_extraction.cnmf import cnmf 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def kb_imshow(image, cmap='jet', title=None):
    """
    Interactive equivalent of imshow for ipython notebook
    """
    colormap = cm.get_cmap(cmap)  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    xr = Range1d(start=0, end=image.shape[1])
    yr = Range1d(start=image.shape[0], end=0)
    p = bpl.figure(x_range=xr, y_range=yr, tools="lasso_select", title=title)

    #p.image(image=[image[::-1, :]], x=0, y=image.shape[0],
    #        dw=image.shape[1], dh=image.shape[0], palette=grayp)
    
    # CHNG - copy from most recent version of plotting code
    p.image(image=[image], x=0, y=0, 
            dw=image.shape[1], dh=image.shape[0], palette=grayp)

    return p



def nb_pick_dots(title, image, A, d1, d2, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9,
                    face_color=None, line_color='cyan', dot_color = 'blue', alpha=0.4,
                    bg_brightness=0.5, line_width=2, coordinates=None, show=True, cmap='gray', **kwargs):
    

    isIPython = False
    try:
        if __PYTHON__: isIPython = True
    except:
        pass    
    
    server = None

    selected_indices = []

    old_cmap = plt.get_cmap(cmap)
    new_cmap = truncate_colormap(old_cmap, 0.0, bg_brightness)

    p = kb_imshow(image, cmap=new_cmap, title=title)
    p.width = 450
    p.height = 450 * d1 // d2
    center = com(A, d1, d2)

    s1 = ColumnDataSource(data=dict(x=center[:, 1], y=center[:, 0]))
    p.circle('x', 'y', source=s1, size=5, color=None,
             fill_color=dot_color, line_width=0, alpha=1)

    if coordinates is None:
        coors = get_contours(A, np.shape(image), thr, thr_method)
    else:
        coors = coordinates
    cc1 = [np.clip(cor['coordinates'][1:-1, 0], 0, d2) for cor in coors]
    cc2 = [np.clip(cor['coordinates'][1:-1, 1], 0, d1) for cor in coors]

    p2 = kb_imshow(image, cmap=new_cmap)
    p2.width = 450
    p2.height = 450 * d1 // d2

    p2.patches(cc1, cc2, alpha=1, color=face_color,
               line_color='yellow', line_width=line_width, **kwargs)

    s2 = ColumnDataSource(data=dict(x=[], y=[]))
    s3 = ColumnDataSource(data=dict(x=cc1, y=cc2))

    p2.patches('x', 'y', source=s2, alpha=1, color=face_color,
               line_color=line_color, line_width=line_width, **kwargs)

    s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2, s3=s3), code="""
                const inds = cb_obj.indices;
                const d1 = s1.data;
                const d2 = s2.data;
                const d3 = s3.data;
                d2['x'] = []
                d2['y'] = []
                for (let i = 0; i < inds.length; i++) {
                    d2['x'].push(d3['x'][inds[i]])
                    d2['y'].push(d3['y'][inds[i]])
                }
                s2.change.emit();
            """)
                             )
    

    button = Button(label="Save", button_type="success")

    def button_python_callback(e):
        selected_indices.extend(s1.selected.indices)
        curdoc().clear()
        server.stop()
        if not isIPython:
            curdoc().session_context.server_context.application_context.io_loop.stop()
        #sys.exit()

    button.on_click(button_python_callback)
    button.js_on_click(CustomJS(code="window.close();"))

    ##
    def bkapp(doc):
        doc.add_root(column(button, row(p, p2)))

    port = 5006
    keepTrying = True
    while keepTrying:
        try:
            server = Server({'/': bkapp}, num_procs=1, port=port)            
            server.start()
            keepTrying = False
        except:
            port -= 1    

    server.io_loop.add_callback(server.show, "/")
    if not isIPython:
        server.io_loop.start()

    #del s1, s2, 
    return selected_indices


def nb_show_work(image, area_indices, area_colors, A, d1, d2, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9,
                    face_color=None, line_color='cyan', dot_color = 'blue', alpha=0.4,
                    bg_brightness=0.5, line_width=1, coordinates=None, show=True, cmap='gray', **kwargs):

    old_cmap = plt.get_cmap(cmap)
    new_cmap = truncate_colormap(old_cmap, 0.0, bg_brightness)

    p = kb_imshow(image, cmap=new_cmap)
    p.width = 450
    p.height = 450 * d1 // d2

    if coordinates is None:
        coors = get_contours(A, np.shape(image), thr, thr_method)
    else:
        coors = coordinates
    cc1 = [np.clip(cor['coordinates'][1:-1, 0], 0, d2) for cor in coors]
    cc2 = [np.clip(cor['coordinates'][1:-1, 1], 0, d1) for cor in coors]

    p.patches(cc1, cc2, alpha=1, color=face_color,
              line_color=line_color, line_width=line_width, **kwargs)
    
    for indices, color in zip(area_indices, area_colors):
        coors = get_contours(A[:,indices], np.shape(image), thr=None, thr_method='max')
        cc1 = [np.clip(cor['coordinates'][1:-1, 0], 0, d2) for cor in coors]
        cc2 = [np.clip(cor['coordinates'][1:-1, 1], 0, d1) for cor in coors]
        p.patches(cc1, cc2, alpha=1, color=None, line_color=color, line_width=1.5)
    
    if show:
        bpl.show(p)


def select_subregions(img, cnm, area_names): # generalize this function: make a higher level function
    area_indices = []

    for name in area_names:
        #bg_brightness - 0.8 or 0.4?
        indices = nb_pick_dots(name, img, cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1], bg_brightness=0.8, line_width=1, show=True, dot_color='dodgerblue')
        area_indices.append(indices)
    
    return area_indices


def show_subregions(img, cnm, area_indices, area_colors):
    nb_show_work(img, area_indices, area_colors, cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1], bg_brightness=0.6, line_color='white', show=True)



def custom_df_f(c, baseline, quantileMin = 50, use_residuals = False): #TODO move this to utils
    """
    A custom version of the detrend_df_f function in the original CaImAn code.
    In this version, we compute the baseline fluorescence using only a specified number
    of frames at the beginning of the recording. By setting quantileMin to 50 we effectively
    normalize to the median of the baseline period.
    """
    A, C, b, f, YrA = c.estimates.A, c.estimates.C, c.estimates.b, c.estimates.f, c.estimates.YrA
    F = C + YrA if use_residuals else C
    F = F * np.sqrt((A.T @ A).diagonal()[:,None])
    B = b @ f
    f0 =  F + (A.T @ B)
    f0 = np.percentile(f0[:,:baseline], quantileMin, axis=1)
    fb = np.percentile(F[:,:baseline], quantileMin, axis=1)
    df_f = (F - fb[:,None])/f0[:,None]
    
    return df_f

def custom_df_f_startend(c, baseline_start, baseline_end, method = 'zscore', use_residuals = False):
    """
    A custom version of the detrend_df_f function in the original CaImAn code.
    In this version, we compute the baseline fluorescence using only a specified number
    of frames at the beginning of the recording. By setting quantileMin to 50 we effectively
    normalize to the median of the baseline period.
    """
    A, C, b, f, YrA = c.estimates.A, c.estimates.C, c.estimates.b, c.estimates.f, c.estimates.YrA
    F = C + YrA if use_residuals else C
    F = F * np.sqrt((A.T @ A).diagonal()[:,None])
    B = b @ f
    f0 =  F + (A.T @ B)
    if method == 'norm_to_median':
        f0 = np.percentile(f0[:,baseline_start:baseline_end], 50, axis=1)
        fb = np.percentile(F[:,baseline_start:baseline_end], 50, axis=1)
    elif method == 'zscore':
        fb = np.mean(F[:,baseline_start:baseline_end], axis=1)
        f0 = np.std(f0[:,baseline_start:baseline_end], axis=1)
    df_f = (F - fb[:,None])/f0[:,None]
    
    return df_f


def visualize_rigcorr_patch_sz(affcorr_file, stride, i=2, j=2):
    """
    visualize what different patch sizes will sample
    """

    raw2 = cm.load_movie_chain([affcorr_file])
    i,j = 2,2 
    x, y = stride

    plt.figure()
    plt.imshow(raw2[0, i*y:(i+1)*y + 1, j*x:(j + 1)*x + 1])

def why(provenance, cnm, center):
    z = 'z1'
    ch_dict = provenance['load_data']['args']['ch_dict']

    rigcorr_results_filenames = provenance['rigid_motion_correction'][z]['filenames']
    func_ch_file =  rigcorr_results_filenames[ch_dict['func_ch']] 
    
    Yr, dims, T = caiman.load_memmap(func_ch_file)
    f_ch_rigcorr = np.reshape(Yr.T, [T] + list(dims), order='F')
    image = f_ch_rigcorr.max(axis=0)

    #f_ch_rigcorr
    A = cnm.estimates.A
    d1, d2 = cnm.estimates.dims[0], cnm.estimates.dims[1]
    coors = get_contours(A, np.shape(image), None, 'max')

    cc1 = [np.clip(cor['coordinates'][1:-1, 0], 0, d2) for cor in coors]
    cc2 = [np.clip(cor['coordinates'][1:-1, 1], 0, d1) for cor in coors]

    # problem_neurons - vieing each C
    problem_neurons = [1, 2, 3, 5, 7, 10, 28, 73, 84, 97]
    plt.figure()
    plt.plot(cnm.estimates.C[problem_neurons, 719:].T)

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100) 
    ax.set_position([0, 0, 1, 1])   # axes fill the figure
    mv = []
    for frame in f_ch_rigcorr[719:]:
        ax.clear()
        ax.axis('off')
        ax.imshow(frame, cmap='binary_r')

        for j, (c1, c2) in enumerate([(cc1[i], cc2[i]) for i in problem_neurons]):
            ax.plot(c1, c2, c=f'C{j}', lw=0.5)

                
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        img = img[:,:,1:]
        
        mv.append(img)

    height, width = mv[0].shape[:2]
    fps = 2
    out = cv2.VideoWriter('debug.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    for img in mv:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release


def _session_window_indices(ses1_len, pre_f, base_f, stim_f):
    """Baseline/stim index windows for a 2-session concatenated recording.

    Session 2 begins at ses1_len — the actual frame count of session 1 in
    the concatenated stack, NOT a value derived from user-supplied stim_s.
    stim_f sets how many post-baseline frames to analyze per session.
    """
    return dict(
        bline1_start = pre_f,
        bline1_end   = pre_f + base_f,
        stim1_start  = pre_f + base_f,
        stim1_end    = pre_f + base_f + stim_f,

        bline2_start = ses1_len + pre_f,
        bline2_end   = ses1_len + pre_f + base_f,
        stim2_start  = ses1_len + pre_f + base_f,
        stim2_end    = ses1_len + pre_f + base_f + stim_f,
    )


def _session_lengths(provenance, z):
    """Per-session frame counts read from the affine-corrected TIFFs."""
    from tifffile import TiffFile
    aff_files = provenance['affine_motion_correction'][z]['filenames']
    ch_files = next(iter(aff_files.values()))  # any channel — counts match across channels
    lengths = []
    for i in sorted(ch_files):
        with TiffFile(str(ch_files[i])) as tf:
            lengths.append(len(tf.pages))
    return lengths


def get_stims1_stims2(provenance, frame_period=0.585, pre_discard_s=30, baseline_s=30, stim_s=360):
    stims1 = []
    stims2 = []
    z_ids = []

    zs = provenance['source_extraction'].keys()

    for i, z in enumerate(zs):
        cnm_file = provenance['source_extraction'][z]['filenames']['cnm_file']
        imgs_path = provenance['rigid_motion_correction'][z]['filenames']['ch2']

        Yr, dims, T = caiman.load_memmap(imgs_path)
        imgs = np.reshape(Yr.T, [T] + list(dims), order='F')
        img = imgs.max(axis=0)

        cnm = cnmf.load_CNMF(cnm_file)
        center = com(cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1])


        #why(provenance, cnm, center)

        pre_f  = round(pre_discard_s / frame_period)
        base_f = round(baseline_s    / frame_period)
        stim_f = round(stim_s        / frame_period)

        # Anchor session-2 windows to the *real* session-1 length, not a value
        # derived from stim_s. Mismatch here is the bug that made baselines
        # leak across sessions.
        ses1_len = _session_lengths(provenance, z)[0]

        if pre_f + base_f + stim_f > ses1_len:
            raise ValueError(
                f"Session-1 analysis window ({pre_f + base_f + stim_f} frames) "
                f"exceeds session-1 length ({ses1_len} frames). Reduce stim_s, "
                f"baseline_s, or pre_discard_s."
            )
        if ses1_len + pre_f + base_f + stim_f > T:
            raise ValueError(
                f"Session-2 analysis window ends at frame "
                f"{ses1_len + pre_f + base_f + stim_f} but the concatenated "
                f"recording has only {T} frames."
            )

        w = _session_window_indices(ses1_len, pre_f, base_f, stim_f)

        fl_acc1 = custom_df_f_startend(cnm, w['bline1_start'], w['bline1_end'], method='zscore', use_residuals=True)
        fl_acc2 = custom_df_f_startend(cnm, w['bline2_start'], w['bline2_end'], method='zscore', use_residuals=True)

        stim1 = fl_acc1[:, w['bline1_start']:w['stim1_end']]
        stim2 = fl_acc2[:, w['bline2_start']:w['stim2_end']]

        stims1.append(stim1)
        stims2.append(stim2)

        z_ids.extend([int(z.replace('z', ''))]*stim1.shape[0])


    stims1 = np.vstack(stims1)
    stims2 = np.vstack(stims2)

    return stims1, stims2, np.r_[z_ids]

def get_resp1_resp2(stims1, stims2, z_ids, stim_onset_idx=51, threshold=1.64):
    start, end = stim_onset_idx, stims1.shape[1]

    responder_stim1 = (np.median(stims1[:, start:end], axis=1) > threshold)
    responder_stim2 = (np.median(stims2[:, start:end], axis=1) > threshold)

    r_stim1only = responder_stim1 & ~responder_stim2
    r_stim2only = ~responder_stim1 & responder_stim2
    r_stim12 = responder_stim1 & responder_stim2




    s1_1 = sorted(stims1[r_stim1only], key=lambda x : np.median(x[start:end]))
    s1_1 = np.array(s1_1)[::-1]

    s1_12 = sorted(stims1[r_stim12], key=lambda x : np.median(x[start:end]))
    s1_12 = np.array(s1_12)[::-1]
    
    s1_2 = sorted(stims1[r_stim2only], key=lambda x : np.median(x[start:end]))
    s1_2 = np.array(s1_2)[::-1]


    s2_1 = sorted(stims2[r_stim1only], key=lambda x : np.median(x[start:end]))
    s2_1 = np.array(s2_1)[::-1]

    s2_12 = sorted(stims2[r_stim12], key=lambda x : np.median(x[start:end]))
    s2_12 = np.array(s2_12)[::-1]
    
    s2_2 = sorted(stims2[r_stim2only], key=lambda x : np.median(x[start:end]))
    s2_2 = np.array(s2_2)[::-1]
    
    arrays1 = [a for a in [s1_1, s1_12, s1_2] if a.shape[0] > 0]
    resp1 = np.vstack(arrays1) if arrays1 else np.empty((0, stims1.shape[1]))

    arrays2 = [a for a in [s2_1, s2_12, s2_2] if a.shape[0] > 0]
    resp2 = np.vstack(arrays2) if arrays2 else np.empty((0, stims2.shape[1]))

    nums = [r_stim1only.sum(), r_stim12.sum(), r_stim2only.sum()]

    z_ids_sel = [z_ids[r_stim1only], z_ids[r_stim12], z_ids[r_stim2only]]

    return resp1, resp2, nums, z_ids_sel


def get_region_labels(provenance, subregion_dir):
    """
    Return int array (N_neurons,) with neuron-to-region assignments:
      0 = Region A, 1 = Region B, -1 = unclassified / no sub-region file.
    Neuron ordering matches get_stims1_stims2 output for the same provenance.
    """
    from caiman.source_extraction.cnmf import cnmf as cnmf_module
    from caiman.base.rois import com
    from pathlib import Path

    labels_all = []
    for z in provenance['source_extraction'].keys():
        cnm_file = provenance['source_extraction'][z]['filenames']['cnm_file']
        cnm = cnmf_module.load_CNMF(cnm_file)
        K = cnm.estimates.A.shape[1]

        sreg_file = Path(subregion_dir) / z / f'subregion_masks_{z}.npy'
        if sreg_file.exists():
            sreg = np.load(sreg_file)           # shape (2, h, w)
            centers = com(cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1])
            h, w = sreg.shape[1], sreg.shape[2]
            lbl = np.full(K, -1, dtype=int)
            for k, (cy, cx) in enumerate(centers):
                r = min(max(int(round(cy)), 0), h - 1)
                c = min(max(int(round(cx)), 0), w - 1)
                if sreg[0, r, c]:
                    lbl[k] = 0
                elif sreg[1, r, c]:
                    lbl[k] = 1
            labels_all.append(lbl)
        else:
            labels_all.append(np.full(K, -1, dtype=int))

    return np.concatenate(labels_all)


def get_spatial_response_data(provenance, frame_period=0.585, pre_discard_s=30,
                              baseline_s=30, stim_s=360, subregion_dir=None):
    """Return per-z-plane data for spatial response map figures.

    For each z-plane: anatomy image (MC channel max projection), neuron
    centers of mass (row, col), per-neuron median z-scores over the stim
    window for stim1 and stim2, and optionally subregion masks.

    Returns a list of dicts with keys:
        z               – z-plane label (e.g. 'z1')
        anatomy         – 2-D float array (h, w), MC-channel max projection
        centers         – (K, 2) float array, center-of-mass (row, col)
        stim1_mdn       – (K,) float array, median z-score during stim1 window
        stim2_mdn       – (K,) float array, median z-score during stim2 window
        subregion_masks – (2, h, w) bool array or None if not defined
    """
    from caiman.source_extraction.cnmf import cnmf as cnmf_module
    from caiman.base.rois import com
    from pathlib import Path as _Path

    ch_dict = provenance['load_data']['args']['ch_dict']
    zs = provenance['source_extraction'].keys()
    result = []

    for z in zs:
        cnm_file = provenance['source_extraction'][z]['filenames']['cnm_file']
        mc_imgs_path = provenance['rigid_motion_correction'][z]['filenames'][ch_dict['mc_ch']]

        cnm = cnmf_module.load_CNMF(cnm_file)
        centers = com(cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1])

        Yr_mc, dims_mc, T_mc = caiman.load_memmap(mc_imgs_path)
        anatomy = np.reshape(Yr_mc.T, [T_mc] + list(dims_mc), order='F').max(axis=0)

        pre_f  = round(pre_discard_s / frame_period)
        base_f = round(baseline_s    / frame_period)
        stim_f = round(stim_s        / frame_period)
        ses1_len = _session_lengths(provenance, z)[0]
        w = _session_window_indices(ses1_len, pre_f, base_f, stim_f)

        fl1 = custom_df_f_startend(cnm, w['bline1_start'], w['bline1_end'], method='zscore', use_residuals=True)
        fl2 = custom_df_f_startend(cnm, w['bline2_start'], w['bline2_end'], method='zscore', use_residuals=True)

        stim1_mdn = np.median(fl1[:, w['stim1_start']:w['stim1_end']], axis=1)
        stim2_mdn = np.median(fl2[:, w['stim2_start']:w['stim2_end']], axis=1)

        sreg = None
        if subregion_dir is not None:
            sreg_file = _Path(subregion_dir) / z / f'subregion_masks_{z}.npy'
            if sreg_file.exists():
                sreg = np.load(sreg_file)

        result.append(dict(z=z, anatomy=anatomy, centers=centers,
                           stim1_mdn=stim1_mdn, stim2_mdn=stim2_mdn,
                           subregion_masks=sreg))
    return result