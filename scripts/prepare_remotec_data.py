import argparse

from rtm.datasets import remotec_preprocessing

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, 
                        help='path to directory containing MIPrep files (input & output of synthetic data generation)')
    parser.add_argument('--data_dir', type=str, 
                        help='path to directory where the prepared data will be saved')
    parser.add_argument('--prefix', type=str, 
                        help='MIPrep file prefix')
    parser.add_argument('--band', type=str, 
                        help='spectral band to select (BAND1, BAND2 or BAND3)')
    args = parser.parse_args()

    remotec_preprocessing.prepare_data(args.target_dir, args.data_dir, args.prefix, args.band)
