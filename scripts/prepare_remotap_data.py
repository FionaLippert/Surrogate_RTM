import argparse, glob, os
import rtm
from rtm.datasets import remotap_preprocessing

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, 
                        help='path to directory where the prepared data will be saved')
    parser.add_argument('--data_dir', type=str, 
                        help='path to directory containing NetCDF files (output of synthetic data generation)')
    parser.add_argument('--prefix', type=str, 
                        help='file prefix')
    parser.add_argument('--band', type=str, 
                        help='band name (NIR, SWIR1, or SWIR3)')
    parser.add_argument('--n_chunks', type=int, default=1,
                        help='number of chunks to load and combine')
    args = parser.parse_args()

    if args.band == 'NIR':
        wl_range = (755, 773)
    elif args.band == 'SWIR1':
        wl_range = (1590, 1675)
    elif args.band == 'SWIR3':
        wl_range = (2305, 2385)


    aux_files = [f for f in os.listdir(args.data_dir) if f.startswith(args.prefix) and f.endswith('aux.nc')]
    aux_files.sort()

    aux_files = aux_files[:args.n_chunks]


    for f in aux_files:
        if args.n_chunks > 1:
            # find chunk number
            chunk = f.split('_')[-1].split('-aux')[0]
            prefix = f'{args.prefix}_{chunk}'
            print(f'Processing chunk {chunk}...')
        else:
            prefix = args.prefix

        remotap_preprocessing.prepare_data(args.target_dir, args.data_dir, 
                                        prefix=prefix, 
                                        wl_range=wl_range,
                                        gas_names=['co2', 'ch4'])

    if args.n_chunks > 1:
        remotap_preprocessing.combine_chunks(args.target_dir, 
                                            args.prefix, 
                                            n_chunks=args.n_chunks, 
                                            combined_dir=f'combined_nchunks={args.n_chunks}')
