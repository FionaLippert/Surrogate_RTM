import argparse, glob, os
from rtm import utils
from rtm.datasets import remotap_retrieval_preprocessing

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, 
                        help='path to directory where the prepared data will be saved')
    parser.add_argument('--data_dir', type=str, 
                        help='path to directory containing NetCDF files (output of retrieval data collection)')
    parser.add_argument('--prefix', type=str, 
                        help='file prefix for saving prepared data')
    parser.add_argument('--band', type=str, 
                        help='band name (NIR, SWIR1, or SWIR3)')
    parser.add_argument('--npix', type=int, default=1000,
                        help='maximum number of pixels to process')
    parser.add_argument('--max_chi2', type=float, default=1000.0,
                        help='maximum chi2 value to consider a pixel/iteration for training')
    args = parser.parse_args()


    iwin = utils.s5_band_to_iwin(args.band)

    remotap_retrieval_preprocessing.prepare_data(
            args.target_dir, 
            args.data_dir, 
            prefix=args.prefix, 
            iwin=iwin, 
            npix=args.npix, 
            max_chi2=args.max_chi2,
    )
    print(f'Prepared data saved to {args.target_dir} with prefix {args.prefix}')
