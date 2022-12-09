import argparse
import subprocess
from os import listdir
from os.path import isfile, join
from math import ceil

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Simulation Amount Analyser',
        description='A program to test the scalability of more simulations',
        epilog='Pully Porky'
    )
    
    # IO Options
    io_options = parser.add_argument_group(
        'IO Options'
    )

    io_options.add_argument(
        '-s',
        '--simulation',
        dest='simulation_file',
        help='Path to simulator',
        type=str,
        required=True
    )

    io_options.add_argument(
        '-m',
        '--model',
        dest='model_file',
        help='Path to model to run simulation on',
        type=str,
        required=True
    )

    io_options.add_argument(
        '-o',
        '--output',
        dest='output_path',
        help='Output File Path',
        type=str,
        required=True
    )

    io_options.add_argument(
        '-d',
        '--device',
        dest='device',
        help='What device to use [0 GPU 1 CPU]',
        type=int,
        default=0,
        required=False
    )

    #General Options
    general_options = parser.add_argument_group(
        'General Options'
    )

    general_options.add_argument(
        '-b',
        '--blocks',
        dest='blocks',
        help='Amount of blocks to run',
        type=int,
        default=1,
        required=False
    )

    general_options.add_argument(
        '-t',
        '--threads',
        dest='threads',
        help='Amount of threads to run per block',
        type=int,
        default=1,
        required=False
    )

    general_options.add_argument(
        '-c',
        '--cores',
        dest='cpu_cores',
        help='Amount of CPU cores to use',
        type=int,
        default=1,
        required=False
    )

    general_options.add_argument(
        '-a',
        '--max_amount',
        dest='max_amount',
        help='Max number of simulations to run UPPER BOUND',
        type=int,
        default=1000000,
        required=False
    )

    general_options.add_argument(
        '-i',
        '--interval',
        dest='interval',
        help='Interval of increasing simulations.',
        type=int,
        default=100000,
        required=False
    )

    args = parser.parse_args()
    return args

def run_simulations(args):
    for run_idx in range(1, args.max_amount + 1, args.interval):

        if args.device == 0:
            blocks = int(args.blocks)
            threads = int(args.threads)

            #amount = ceil(float(run_idx) / float(blocks * threads))

            file_path = f'{args.output_path}_GPU_{run_idx}'

            subprocess.run([
                args.simulation_file,
                '-m', args.model_file,
                '-b', f'{blocks},{1 if threads == 0 else threads}',
                '-n', f'{run_idx}',
                '-c', '1',
                '-d', '0',
                '-w', 'l',
                '-o', file_path,
                '-x', '100t',
                '-v', '0',
                '-s'
            ])
        
        if args.device == 1:
            cpu_core = args.cpu_cores
            #amount = ceil((float(run_idx) / float(cpu_core)))

            file_path = f'{args.output_path}_CPU_{run_idx}'

            subprocess.run([
                args.simulation_file,
                '-m', args.model_file,
                '-b', f'1,{cpu_core}',
                '-n', f'{run_idx}',
                '-c', f'{cpu_core}',
                '-d', '1',
                '-w', 'l',
                '-o', file_path,
                '-x', '100t',
                '-v', '0'
            ])


if __name__ == '__main__':
    args = parse_args()
    run_simulations(args)

    