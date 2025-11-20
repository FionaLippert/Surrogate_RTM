import argparse, glob, os
import os.path as osp
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sn
from rtm import utils
from rtm.datasets import remotap_retrieval_preprocessing


def load_retrieval_data(data_dir, iwin, npix=1000):
    ds_list = []
    for i in range(npix):
        fp = f'{data_dir}/pix_{i+1:>05}_win_{iwin:02}.nc'
        if not osp.exists(fp):
            continue
        ds_list.append(xr.load_dataset(fp))

    return xr.concat(ds_list, dim='npix')

def prepare_data(ret_data, max_chi2=10000):

    # get valid retrieval iterations
    chi2 = ret_data.chi2.values # (npix, nitr)
    valid = (chi2 > 0) & (chi2 < max_chi2)

    print(f'Number of valid retrieval iterations: {np.sum(valid)} ({np.sum(valid)/valid.size*100:.1f}%)')

    # get target derivatives for valid iterations
    dy_target = ret_data.kmat.values # (npix, nfeatures, nwave, nitr)
    dy_target = dy_target.transpose(0, 3, 1, 2) # (npix, nitr, nfeatures, nwave)
    dy_target = dy_target[valid] # (nvalid, nfeatures, nwave)

    # get proxy derivatives for valid iterations
    dy_proxy = ret_data.kmat_ns.values # (npix, nfeatures, nwave, nitr)
    dy_proxy = dy_proxy.transpose(0, 3, 1, 2) # (npix, nitr, nfeatures, nwave)
    dy_proxy = dy_proxy[valid] # (nvalid, nfeatures, nwave)

    # get NN derivates for valid iterations
    dy_nn = ret_data.kmat_nn.values # (npix, nfeatures, nwave, nitr)
    dy_nn = dy_nn.transpose(0, 3, 1, 2) # (npix, nitr, nfeatures, nwave)
    dy_nn = dy_nn[valid] # (nvalid, nfeatures, nwave)

    return dy_target, dy_proxy, dy_nn

def generate_figure(ret_data, dy_target, dy_nn, args):
    # iterate over features in x_in and plot dy_pred vs dy_target
    feature_names = ret_data.n_in_varnames.split(',')
    nfeatures = len(feature_names)
    
    gas_mapping = {
        1: 'h2o',
        2: 'co2',
        6: 'ch4'  
    }
    
    # make figure with 2x4 subplots, that I can index with i
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    axs = axs.flatten()
    
    i_ax = 0
    i_mol = 0
    
    for i, feature in enumerate(feature_names):

        if np.abs(dy_target[:, i, :]).max() == 0:
                continue

        ax = axs[i_ax]
        i_ax += 1

        if 'vmol' in feature:
                gas_id = int(feature.split('_')[1])
                feature = f'c_{gas_mapping[gas_id]}'
                i_mol += 1

        if '_mode1' in feature:
                feature = feature.replace('_mode1', '')
        if 'reff' in feature:
                feature = 'angstrom_exp'

        # set color to lowest viridis color
        c = '#440154'
        ax.scatter(dy_target[:, i, :].flatten(), (dy_nn[:, i, :]).flatten(), s=0.2, alpha=0.5, color='gray', zorder=0)

        # 2D histogram
        # sn.histplot(x=dy_target[:, i, :].flatten(), y=(dy_nn[:, i, :]).flatten(), 
        #                 bins=50, stat='proportion', cbar=False, cmap='viridis', ax=ax, pthresh=.05, zorder=100)
        lims = [min(dy_target[:, i, :].min(), (dy_nn[:, i, :]).min()),
                max(dy_target[:, i, :].max(), (dy_nn[:, i, :]).max())]

        # compute MSE
        mse = np.mean((dy_nn[:, i, :] - dy_target[:, i, :])**2)
        ax.text(0.05, 0.93, f'MSE = {mse:.2e}', transform=ax.transAxes, fontsize=12)

        # compute correlation coefficient
        corr_coef = np.corrcoef(dy_target[:, i, :].flatten(), (dy_nn[:, i, :]).flatten())[0, 1]
        ax.text(0.05, 0.88, f'R = {corr_coef:.2f}', transform=ax.transAxes, fontsize=12)
        
        # plot kde on top
        sn.kdeplot(x=dy_target[:, i, :].flatten(), y=(dy_nn[:, i, :]).flatten(), 
                   fill=True, thresh=0.05, levels=20, alpha=1.0, cmap='viridis', ax=ax)

        vmin = min(dy_target[:, i, :].min(), (dy_nn[:, i, :]).min())
        vmax = max(dy_target[:, i, :].max(), (dy_nn[:, i, :]).max())
        # lims = [dy_target[:, i, :].min(), dy_target[:, i, :].max()]
        lims = [vmin, vmax]

        # add 1:1 line
        ax.plot(lims, lims, 'r--', alpha=1.0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel(r'$dI/dx$ RemoTAP', fontsize=12)
        ax.set_ylabel(r'$dI/dx$ {args.nn_name}', fontsize=12)
        ax.set_title(f'x = {feature}', fontsize=12)

    # remove remaining empty subplots
    for j in range(i_ax, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.savefig(f"{args.figure_dir}/{args.prefix}_derivative_evaluation.png", bbox_inches='tight', dpi=400)
    plt.close(fig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--figure_dir', type=str, 
                        help='path to directory where the generated figure will be saved',
                        default='../output/figures')
    parser.add_argument('--data_dir', type=str, 
                        help='path to directory containing data collected during retrievals')
    parser.add_argument('--prefix', type=str, 
                        help='file prefix for saving figure')
    parser.add_argument('--nn_name', type=str,
                        help='name of the neural network model')
    parser.add_argument('--band', type=str, 
                        help='band name (NIR, SWIR1, or SWIR3)')
    parser.add_argument('--npix', type=int, default=1000,
                        help='maximum number of pixels to process')
    parser.add_argument('--max_chi2', type=float, default=1000.0,
                        help='maximum chi2 value to consider a pixel/iteration for training')
    args = parser.parse_args()


    iwin = utils.s5_band_to_iwin(args.band)

    ret_data = load_retrieval_data(args.data_dir, iwin=iwin, npix=args.npix)
    dy_target, dy_proxy, dy_nn = prepare_data(ret_data, max_chi2=args.max_chi2)

    generate_figure(ret_data, dy_target, dy_nn, args)
 
    print(f'Figure saved to {args.figure_dir} with prefix {args.prefix}')
