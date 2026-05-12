from pipeline import init, load_data, affine_motion_correction, rigid_motion_correction, source_extraction
import caiman as cm
from collections import _OrderedDictKeysView
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pipeline_funcs import get_stims1_stims2, get_resp1_resp2

#os.environ['CAIMAN_DATA'] = "../caiman_data"

if  __name__ == "__main__":
    
    
    if  True:
        provenance = init('ZH511')
        multi_path = ['data/ZH511_fructose', 'data/ZH511_fructose_2']
        zs = [ 'z3'] #['z1', 'z2', 'z3', 'z4', 'z5', 'z6']
        ch_dict = {'mc_ch': 'ch1',  'func_ch' : 'ch2'}

        for z in zs:
            provenance, data = load_data(provenance, multi_path, ch_dict, z)

            # #
            # data_array = np.zeros((len(z), len(multi_path), len(ch_dict), *data['ch1'][z[0]][0].shape))
            # for j, zz in enumerate(z):
            #     for k, ch in enumerate(ch_dict.values()):
            #         for i, _ in enumerate(multi_path):
            #             data_array[j, i, k, ...] = data[ch][zz][i]


            # max_proj = data_array.sum(axis=0) # sum across the first dimension - z level

            # #max_proj[0][1].view(cm.base.movies.movie).play(fr=30, magnification=2)   #first stim, 2nd channel

            # # put max proj back into the format the rest of the code expects
            # max_proj_dict = {ch : OrderedDict() for ch in ch_dict.values()}
            # for k, ch in enumerate(ch_dict.values()):
            #     for i, _ in enumerate(multi_path):
            #         max_proj_dict[ch][i] = max_proj[i, k, ...] # concatenate 


            provenance, affcorr_results = affine_motion_correction(provenance, z, data)   


            provenance, mc = rigid_motion_correction(provenance, z, affcorr_results, max_shifts=(100,100), max_deviation_rigid=20, pw_rigid=True)

            provenance = source_extraction(provenance, affcorr_results, z, mc)

    
        
    if False:
        stims1, stims2 = get_stims1_stims2(provenance)
        resp1, resp2, nums = get_resp1_resp2(stims1, stims2)
        fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
        im = ax1.imshow(resp1)
        ax1.set_aspect(8)
        im2 = ax2.imshow(resp2)
        ax2.set_aspect(8)

        ax1.set_xlabel('Time (frames)')
        ax1.set_ylabel('Neurons - responders sorted by stim1 responses')
        ax1.set_title('stim1')
        ax2.set_xlabel('Time (frames)')
        ax2.set_title('stim2')

        fig.colorbar(im, ax=[ax1, ax2], shrink=0.3)


    if True:
        from pipeline import _get_provenance

        #folders = ['ZH510_T','ZH417_T','ZH423','ZH505_T']
        folders = ['ZH511']
        _all_stims1 = []
        _all_stims2 = []
        _z_ids = [] 
        for folder in folders:
            provenance = _get_provenance(folder)

            stims1, stims2, z_ids = get_stims1_stims2(provenance)
            _all_stims1.append(stims1)
            _all_stims2.append(stims2)
            _z_ids.append(z_ids)


        all_stims1 = np.vstack(_all_stims1)
        all_stims2 = np.vstack(_all_stims2)  
        z_ids = np.concatenate(_z_ids)

        resp1, resp2, nums, z_ids_sel = get_resp1_resp2(all_stims1, all_stims2, z_ids) 
        
        stim1_responders = z_ids_sel[0]
        stim12_responders = z_ids_sel[1]
        stim2_responders = z_ids_sel[2]

        print(f'Z levels for Stim 1 responders\n{stim1_responders}')
        print(f'Z levels for Stim 1,2 responders\n{stim12_responders}')
        print(f'Z levels for Stim 2 responders\n{stim2_responders}')


        #############################
        from pipeline_funcs import select_subregions, show_subregions, custom_df_f_startend
        from tifffile import imread
        from caiman.source_extraction.cnmf import cnmf 
        import numpy as np
        from caiman.base.rois import com
        import matplotlib.pyplot as plt

        f_ch = provenance['load_data']['args']['ch_dict']['func_ch']
        img_path = provenance['affine_motion_correction'][f_ch]
        img = imread(img_path)
        img = img.max(axis=0)

        cnm = cnmf.load_CNMF(pipeline.cnmf_file)
        center = com(cnm.estimates.A, cnm.estimates.dims[0], cnm.estimates.dims[1]) 

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
        ############

        def make_figure(resp1, resp2, nums):
            # fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True, constrained_layout=True)
            # plt.subplots_adjust(left=0.2)
            fig = plt.figure(constrained_layout=True)
            gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 10, 10])
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2])

            ax0.xaxis.set_visible(False)
            ax0.set_ylim([0, resp1.shape[0]-1])
            ax0.invert_yaxis()
            ax0.axhspan(0, nums[0], facecolor='#4fa1ca')
            ax0.axhspan(nums[0], nums[0]+nums[1], facecolor='#bb70b6')
            ax0.axhspan(nums[0]+nums[1], nums[0]+nums[1]+nums[2], facecolor='#110979')
            ax0.set_yticks([0, resp1.shape[0]-1], [1, resp1.shape[0]])
            ax0.set_ylabel('Neuron #')

            vmin = 0#min([resp1.min(), resp2.min()])
            vmax = 8#max([resp1.max(), resp2.max()])

            im = ax1.imshow(resp1, aspect='auto', vmin=vmin, vmax=vmax)
            im2 = ax2.imshow(resp2, aspect='auto', vmin=vmin, vmax=vmax)

            #ax1.set_xticks([0, 103, 410], [0, 60, 240])
            #ax1.set_xticks([0, 52, 358], [0, 30, 210])
            ax1.set_xticks([0, 52, 154, 667], [0, 30, 90, 390])
            ax1.set_xlabel('Time (s)')
            ax1.yaxis.set_visible(False)
            #ax1.set_ylabel('Neuron #')
            ax1.set_title('Fructose - stim1')
            ax1.axvline(x=52, color='w', linewidth=0.5).set_dashes([10, 6])
            ax1.axvline(x=154, color='w', linewidth=0.5).set_dashes([10, 6])
            
            #ax2.set_xticks([0, 103, 410], [0, 60, 240])
            #ax2.set_xticks([0, 52, 358], [0, 30, 210])
            ax2.set_xticks([0, 52, 154, 667], [0, 30, 90, 390])
            ax2.set_xlabel('Time (s)')
            ax2.yaxis.set_visible(False)
            ax2.set_title('Fructose - stim2')
            ax2.axvline(x=52, color='w', linewidth=0.5).set_dashes([10, 6])
            ax2.axvline(x=154, color='w', linewidth=0.5).set_dashes([10, 6])

            fig.colorbar(im, ax=[ax1, ax2], shrink=0.5) 
            fig.set_size_inches([6.4, 4.8])

            return fig

        combined_fig = make_figure(resp1, resp2, nums)

        split_figs = []
        all_nums = []
        all_resp1 = []
        all_resp2 = []
        for i, f in enumerate(folders):
            _resp1, _resp2, _nums = get_resp1_resp2(_all_stims1[i], _all_stims2[i])
            all_resp1.append(resp1)
            all_resp2.append(resp2)
            all_nums.append(_nums)
            fig = make_figure(_resp1, _resp2, _nums)
            fig.suptitle(f)
            split_figs.append(fig)


        # % responsive neurons
        all_nums_arr = np.array(all_nums)
        pct_by_folder = (all_nums_arr / all_nums_arr.sum(axis=1, keepdims=True))*100
        pct = pct_by_folder.mean(axis=0)
        std = pct_by_folder.std(axis=0)

        pct = [pct[0], pct[2], pct[1]] # Fructose Glucose Both
        std = [std[0], std[2], std[1]]

        fig, ax = plt.subplots(1)
        ax.bar(['Fructose', 'Glucose', 'Both'], pct, color=['#4fa1ca', '#110979', '#bb70b6'])
        ax.errorbar(['Fructose', 'Glucose', 'Both'], pct, yerr=std, fmt='o', ecolor='black', elinewidth=0.5, capsize=5, capthick=0.5, markersize=0)

        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel('% responsive neurons')


        # Mean z score
        start, end = 103, 719 #103, 411
        # meanfrucResp2fruc = frucResp2fruc[:, start:end].mean(axis=1)
        # meanglucResp2gluc = glucResp2gluc[:, start:end].mean(axis=1)

        meansfruc = resp1[:, start:end].mean()
        stdsfruc = resp1[:, start:end].std()

        meansgluc = resp2[:, start:end].mean()
        stdsgluc = resp2[:, start:end].std()

        fig, ax = plt.subplots(1)
        ax.bar(['Fructose', 'Glucose'], [meansfruc, meansgluc], color=['#4fa1ca', '#110979'])
        ax.errorbar(['Fructose', 'Glucose'], [meansfruc, meansgluc], yerr=[stdsfruc, stdsgluc], fmt='o', ecolor='black', elinewidth=0.5, capsize=5, capthick=0.5, markersize=0)

        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel('Mean z-scoore')


        # time based
        frucResp2fruc = resp1[:nums[0]]
        frucResp2fruc_mean = frucResp2fruc.mean(axis=0)
        frucResp2fruc_std = frucResp2fruc.std(axis=0)

        frucResp2gluc = resp2[:nums[0]]
        frucResp2gluc_mean = frucResp2gluc.mean(axis=0)
        frucResp2gluc_std = frucResp2gluc.std(axis=0)

        glucResp2gluc = resp2[nums[0]+nums[1]:]
        glucResp2gluc_mean = glucResp2gluc.mean(axis=0)
        glucResp2gluc_std = glucResp2gluc.std(axis=0)

        glucResp2fruc = resp1[nums[0]+nums[1]:]
        glucResp2fruc_mean = glucResp2fruc.mean(axis=0)
        glucResp2fruc_std = glucResp2fruc.std(axis=0)

        fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(frucResp2fruc_mean, c='#4fa1ca')
        ax1.fill_between(np.arange(len(frucResp2fruc_mean)), 
                 frucResp2fruc_mean - frucResp2fruc_std, 
                 frucResp2fruc_mean + frucResp2fruc_std, color='#4fa1ca', alpha=0.2, edgecolor='none')
        ax1.axvline(x=52, color='k', lw=0.5).set_dashes([10, 6])
        ax1.set_xlabel('Time (s)')
        #ax1.set_xticks([0, 103, 411], [0, 60, 240])
        #ax1.set_xticks([0, 103,308], [0, 60, 180])
        ax1.set_xticks([0, 103,719], [0, 60, 420])


        ax2.plot(glucResp2fruc_mean, c='#4fa1ca')
        ax2.fill_between(np.arange(len(glucResp2fruc_mean)), 
                  glucResp2fruc_mean - glucResp2fruc_std, 
                  glucResp2fruc_mean + glucResp2fruc_std, color='#4fa1ca', alpha=0.2, edgecolor='none')
        ax2.axvline(x=52, color='k', lw=0.5).set_dashes([10, 6])
        ax2.axvline(x=52, color='k', lw=0.5).set_dashes([10, 6])
        ax2.set_xlabel('Time (s)')
        #ax2.set_xticks([0, 103, 411], [0, 60, 240])
        #ax2.set_xticks([0, 103,308], [0, 60, 180])
        ax2.set_xticks([0, 103,719], [0, 60, 420])

        fig1.set_size_inches([9.3, 4.7])

        fig2, (ax3, ax4) = plt.subplots(1, 2, sharey=True)
        ax3.plot(frucResp2gluc_mean, c='#110979')
        ax3.fill_between(np.arange(len(frucResp2gluc_mean)), 
                 frucResp2gluc_mean - frucResp2gluc_std, 
                 frucResp2gluc_mean + frucResp2gluc_std, color='#110979', alpha=0.2, edgecolor='none')
        ax3.axvline(x=52, color='k', lw=0.5).set_dashes([10, 6])
        ax3.set_xlabel('Time (s)')
        #ax3.set_xticks([0, 103, 411], [0, 60, 240])
        #ax3.set_xticks([0, 103,308], [0, 60, 180])
        ax3.set_xticks([0, 103,719], [0, 60, 420])

        ax4.plot(glucResp2gluc_mean, c='#110979')
        ax4.fill_between(np.arange(len(glucResp2gluc_mean)), 
                  glucResp2gluc_mean - glucResp2gluc_std, 
                  glucResp2gluc_mean + glucResp2gluc_std, color='#110979', alpha=0.2, edgecolor='none')
        ax4.axvline(x=52, color='k', lw=0.5).set_dashes([10, 6])
        ax4.set_xlabel('Time (s)')
        #ax4.set_xticks([0, 103, 411], [0, 60, 240])
        #ax1.set_xticks([0, 103,308], [0, 60, 180])
        ax4.set_xticks([0, 103,719], [0, 60, 420])
        fig2.set_size_inches([9.3, 4.7])

        # fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # ax1.plot(frucResp2fruc_mean, c='#4fa1ca')
        # ax1.fill_between(np.arange(len(frucResp2fruc_mean)), 
        #                  frucResp2fruc_mean - frucResp2fruc_std, 
        #                  frucResp2fruc_mean + frucResp2fruc_std, color='#4fa1ca', alpha=0.2, edgecolor='none')
        # ax1.axvline(x=103, color='k', lw=0.5).set_dashes([10, 6])
        # ax1.set_xlabel('Time (s)')
        # ax1.set_xticks([0, 103, 411], [0, 60, 240])

        # ax2.plot(glucResp2fruc_mean, c='#4fa1ca')
        # ax2.fill_between(np.arange(len(glucResp2fruc_mean)), 
        #                  glucResp2fruc_mean - glucResp2fruc_std, 
        #                  glucResp2fruc_mean + glucResp2fruc_std, color='#4fa1ca', alpha=0.2, edgecolor='none')
        # ax2.axvline(x=103, color='k', lw=0.5).set_dashes([10, 6])
        # ax2.axvline(x=103, color='k', lw=0.5).set_dashes([10, 6])
        # ax2.set_xlabel('Time (s)')
        # ax2.set_xticks([0, 103, 411], [0, 60, 240])

        # mean z score split
        meanfrucResp2fruc = frucResp2fruc[:, start:end].mean(axis=1)
        meanfrucResp2gluc = frucResp2gluc[:, start:end].mean(axis=1)
        meanglucResp2gluc = glucResp2gluc[:, start:end].mean(axis=1)
        meanglucResp2fruc = glucResp2fruc[:, start:end].mean(axis=1)

        fig3, ax5 = plt.subplots(1)
        meanmeanfrucResp2fruc = meanfrucResp2fruc.mean()
        stdmeanfrucResp2fruc = meanfrucResp2fruc.std()

        meanmeanglucResp2fruc = meanglucResp2fruc.mean()
        stdmeanglucResp2fruc = meanglucResp2fruc.std()

        ax5.bar(['Fructose', 'Glucose'], [meanmeanfrucResp2fruc, meanmeanglucResp2fruc], color=['#4fa1ca', '#110979'])
        ax5.errorbar(['Fructose', 'Glucose'], 
                    [meanmeanfrucResp2fruc, meanmeanglucResp2fruc], 
                    yerr=[stdmeanfrucResp2fruc, stdmeanglucResp2fruc], 
                    fmt='o', ecolor='black', elinewidth=0.5, capsize=5, capthick=0.5, markersize=0)
        ax5.spines[['top', 'right']].set_visible(False)
        ax5.spines['bottom'].set_position(('data', 0))
        miny = ax5.get_ylim()[0]
        for lab in ax5.get_xticklabels():
            lab.set_y(miny)
        ax5.set_ylabel('Mean z-score')
        ax5.set_title('Responses of fructose/glucose responders to fructose')

        fig4, ax6 = plt.subplots(1)
        meanmeanfrucResp2gluc = meanfrucResp2gluc.mean()
        stdmeanfrucResp2gluc = meanfrucResp2gluc.std()

        meanmeanglucResp2gluc = meanglucResp2gluc.mean()
        stdmeanglucResp2gluc = meanglucResp2gluc.std()

        ax6.bar(['Fructose', 'Glucose'], [meanmeanfrucResp2gluc, meanmeanglucResp2gluc], color=['#4fa1ca', '#110979'])
        ax6.errorbar(['Fructose', 'Glucose'], 
                    [meanmeanfrucResp2gluc, meanmeanglucResp2gluc], 
                    yerr=[stdmeanfrucResp2gluc, stdmeanglucResp2gluc], 
                    fmt='o', ecolor='black', elinewidth=0.5, capsize=5, capthick=0.5, markersize=0)
        ax6.spines[['top', 'right']].set_visible(False)
        ax6.spines['bottom'].set_position(('data', 0))
        miny = ax6.get_ylim()[0]
        for lab in ax6.get_xticklabels():
            lab.set_y(miny)
        ax6.set_ylabel('Mean z-score')
        ax6.set_title('Responses of fructose/glucose responders to glucose')

        pass
    


plt.show(block=True)





