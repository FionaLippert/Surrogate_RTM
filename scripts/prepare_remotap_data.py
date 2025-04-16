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
    parser.add_argument('--postfix', type=str, default='',
                        help='file postfix')
    parser.add_argument('--band', type=str, 
                        help='band name (NIR, NIR1, NIR2, SWIR1, or SWIR3)')
    parser.add_argument('--n_chunks', type=int, default=1,
                        help='number of chunks to load and combine')
    parser.add_argument('--instrument', type=str, default='instr2_noise0',
                        help='should be instr2_noise0 if both MAP and CO2I were simulated, otherwise instr1_noise0')
    parser.add_argument('--postfix_proxy', type=str, default='proxy',
                        help='postfix of the files containing proxy spectra')
    parser.add_argument('--instrument_proxy', type=str, default='instr1_noise0',
                        help='non-scattering simulations are always CO2I-only, so instr id=1')
    args = parser.parse_args()

    gas_names = ['co', 'co2', 'ch4'] # TODO: add o2 
    # gas_names = ['co2', 'ch4']
    wl_range = rtm.utils.s5_band_to_wl_range(args.band)

    if len(args.postfix) > 0:
        base_prefix = f'{args.prefix}_{args.postfix}'
    else:
        base_prefix = args.prefix

    aux_files = [f for f in os.listdir(args.data_dir) if f.startswith(base_prefix) and f.endswith('aux.nc')]
    aux_files.sort()

    aux_files = aux_files[:args.n_chunks]


    for f in aux_files:
        if args.n_chunks > 1:
            # find chunk number
            chunk = f.split(f'{base_prefix}_')[-1].split('-aux')[0]
            # chunk = f.split('_')[-1].split('-aux')[0]
            prefix = f'{base_prefix}_{chunk}'
            prefix_proxy = f'{args.prefix}_{args.postfix_proxy}_{chunk}'

            print(f'Processing chunk {chunk}...')
        else:
            prefix = base_prefix
            prefix_proxy = f'{args.prefix}_{args.postfix_proxy}'

        remotap_preprocessing.prepare_data(args.target_dir, args.data_dir, 
                                        prefix=prefix, 
                                        wl_range=wl_range,
                                        gas_names=gas_names,
                                        instrument=args.instrument,
                                        prefix_proxy=prefix_proxy,
                                        instrument_proxy=args.instrument_proxy)

    if args.n_chunks > 1:
        remotap_preprocessing.combine_chunks(args.target_dir, 
                                            args.prefix, 
                                            n_chunks=args.n_chunks, 
                                            combined_dir=f'combined_nchunks={args.n_chunks}')
