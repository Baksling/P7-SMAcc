import argparse
import os
import subprocess
from os import listdir
from os.path import isfile, join, exists
from math import ceil
from typing import Dict, Tuple, Set, List

TEMP_FOLDER_NAME = './tmp_results'
POST_FIX_TMP_NAME = '_results.tsv'

PASSED = '\033[92m'
FAILED = '\033[93m'
WARNING = '\033[91m'
ENDC = '\033[0m'


class value_checker():
    def __init__(self, val, time):
        self.value = val
        self.time = time


def __parse_args():
    parser = argparse.ArgumentParser(
        prog="CUDA Analysis Program",
        description="Program is used to analyse different parameters for CUDA Simulation",
        epilog="Pully proky"
    )

    # IO Options
    io_options = parser.add_argument_group(
        'IO Options'
    )

    io_options.add_argument(
        '-s',
        '--simulation',
        dest='simulation_path',
        help='Path to simulation location',
        type=str,
        required=True
    )

    io_options.add_argument(
        '-f',
        '--folder',
        dest='folder_path',
        help='Path to folder with test XML',
        type=str,
        required=True
    )

    io_options.add_argument(
        '-op',
        '--output',
        dest='output_path',
        help='filepath to folder to output results',
        type=str,
        required=True
    )

    # General Options
    general_options = parser.add_argument_group(
        'General Options'
    )

    general_options.add_argument(
        '-y',
        '--use_time',
        dest='use_time',
        help='(0) Use steps (1) use time',
        type=int,
        default=1,
        required=False
    )

    general_options.add_argument(
        '-p',
        '--max_progression',
        dest='max_progression',
        help='Maximum number of steps / time to pass before terminating simulation can be configured with -y {0, 1}',
        type=int,
        default=100,
        required=False
    )

    general_options.add_argument(
        '-a',
        '--amount',
        dest='amount',
        help='Number of total simulations to run default = 1M',
        type=int,
        default=1000000,
        required=False
    )

    general_options.add_argument(
        '-v',
        '--variance',
        dest='variance',
        help='How much the results may variate from the facts [In percent default = 1.0]',
        type=float,
        default=1.0,
        required=False
    )

    general_options.add_argument(
        '-t',
        '--time_variance',
        dest='time_variance',
        help='How much the time results may variate form the facts [In ms default = 50] [0 to ignore]',
        type=int,
        default=0,
        required=False
    )

    general_options.add_argument(
        '-sm',
        '--shared_memory',
        dest='use_shared',
        help='Use shared Memory',
        type=int,
        default=0,
        required=False
    )

    general_options.add_argument(
        '-j',
        '--jit',
        dest='use_jit',
        help='Set it to 1 if you want to use jit',
        type=int,
        default=0,
        required=False
    )

    general_options.add_argument(
        '-alp',
        '--alpha',
        dest='alpha',
        help='Set value for alpha',
        type=float,
        default=0.0,
        required=False
    )

    general_options.add_argument(
        '-epi',
        '--epsilon',
        dest='epsilon',
        help='Set the value for epsilon',
        default=0.0,
        required=False
    )

    args = parser.parse_args()
    return args


def run_simulations(args_) -> tuple[set[str], list[str]]:
    not_run_set = set()
    simulator_path = args_.simulation_path
    folder_path = args_.folder_path
    output_path = args_.output_path

    only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for file in only_files:
        print("RUNNING", file)
        file_path = join(folder_path, file)
        # print(f' m: {file_path}\n n: {args_.amount}\n o: {join(TEMP_FOLDER_NAME, file.replace(".xml", ""))}\n x: {args_.max_progression}{"t" if args_.use_time else "s"}')
        # amount = int(ceil(float(args_.amount) / float(32 * 512)))

        parameters = [
            simulator_path,
            '-m', file_path,
            '-b', '40,256',
            '-c', '1',
            '-d', '0',
            '-w', 'r',
            '-o', f'{join(TEMP_FOLDER_NAME, file.replace(".xml", ""))}',
            '-x', f'{args_.max_progression}{"t" if args_.use_time else "s"}',
            '-v', '0'
            '-op', output_path 
             
        ]
        
        alpha = float(args_.alpha)
        epsilon = float(args_.epsilon)
        
        if float(alpha) > 0.0 and float(epsilon) > 0.0:
            parameters.append('-a')
            parameters.append(f'{alpha}')
            
            parameters.append('-e')
            parameters.append(f'{epsilon}')
        else:
            parameters.append('-n')
            parameters.append(f'{args_.amount}')

        if args.use_shared == 1:
            parameters.append('-s')

        if args.use_jit == 1:
            parameters.append('-j')

        subprocess.run(parameters)

        new_file_name = f'{file.replace(".xml", "")}_results.tsv'
        if not exists(join(TEMP_FOLDER_NAME, new_file_name)):
            not_run_set.add(new_file_name)

    return not_run_set, only_files


def check_simulation_results(args_, _expected_results, not_run_set, all_files) -> None:
    def within_range(expected, actual, variance) -> bool:
        return expected + variance > actual > expected - variance  # actual < expected + variance and actual > expected - variance <-- OLD BUT SECURE

    only_files = [f'{file.replace(".xml", "")}_results.tsv' for file in all_files]

    # print(only_files)
    with open(args_.output_path, 'w') as op:
        for file in only_files:
            file_path = join(TEMP_FOLDER_NAME, file)
            file_results = _expected_results[f'{file.replace(POST_FIX_TMP_NAME, "")}']
            expected_value = file_results.value
            expected_time = file_results.time
            actual_result = 0.0
            actual_time = 0.0
            if file not in not_run_set:
                try:
                    with open(file_path, 'r') as f:
                        data_line = f.readlines()[0].replace('\n', '')
                        tmp_split = data_line.split('\t')

                        actual_result = float(tmp_split[0])
                        actual_time = int(tmp_split[1])

                    output_str = f'{file} '
                    if within_range(expected_value, actual_result, args_.variance) and \
                            (args_.time_variance == 0 or within_range(expected_time, actual_time, args_.time_variance)):
                        output_str += f'{PASSED}PASSED{ENDC}'
                    else:
                        output_str += f'{FAILED}FAILED{ENDC}'
                    output_str += f' - Got: {actual_result} | Value variance: {expected_value - actual_result} percent | Time variance {expected_time - actual_time} [ms]'
                    op.write(f'{file}\t{actual_result}\t{expected_value}\n')
                    print(output_str)
                except: 
                    print(file, f'{FAILED}FAILED{ENDC} - EXCEPT')
            else:
                print(file, f"{WARNING}FAILED{ENDC} - DID NOT FINISH SIMULATION!")


def get_expected_simulation_results() -> dict[str, value_checker]:
    return {
        'clock_var': value_checker(100.0, 270),
        'dicebase': value_checker(16.6, 440),
        'dicebaseUnfair': value_checker(6.3, 1350),
        'random_test': value_checker(14.4, 122),
        'rare_events': value_checker(0.0001, 1000),
        'rate_test': value_checker(33.45, 155),
        'var_test': value_checker(47.4, 10300),
        'ifelse_test': value_checker(50.0, 225),
        'random_test_1': value_checker(37.45, 0),
        'clock_assign_1': value_checker(80.9, 0),
        'clock_assign_2': value_checker(68.9, 0),
        'clock_assign_3': value_checker(81.8, 0),
        'int_assign_1': value_checker(0.00, 0),
        'int_assign_2': value_checker(100.00, 0),
        'int_assign_3': value_checker(50.0, 0),
        'int_assign_4': value_checker(50.0, 0),
        'double_assign_1': value_checker(0.00, 0),
        'double_assign_2': value_checker(100.00, 0),
        'double_assign_3': value_checker(50.0, 0),
        'double_assign_4': value_checker(50.0, 0),
        'guardz_1': value_checker(63.0, 0),
        'guardz_2': value_checker(63.0, 0),
        'guardz_3': value_checker(63.0, 0),
        'guardz_4': value_checker(63.0, 0),
        'guardz_5': value_checker(63.0, 0),
        'update_1': value_checker(36.5, 0),
        'update_2': value_checker(36.5, 0),
        'update_3': value_checker(73.8, 0),
        'update_4': value_checker(36.5, 0),
        'update_5': value_checker(50.0, 0),
        'update_6': value_checker(50.0, 0),
        'update_7': value_checker(30.0, 0),
        'update_8': value_checker(30.0, 0),
        'update_var_inv_1': value_checker(50.0, 0),
        'invariant_1': value_checker(25.0, 0),
        'invariant_2': value_checker(50.0, 0),
        'invariant_3': value_checker(50.0, 0),
        'rates_1': value_checker(50.0, 0),
        'rates_2': value_checker(0.00, 0),
        'rates_3': value_checker(50.0, 0),
        'rates_4': value_checker(50.0, 0),
        'rates_5': value_checker(31.0, 0),
        'rates_6': value_checker(50.0, 0),
        'rates_7': value_checker(50.0, 0),
        'not_test': value_checker(50.0, 0)
    }


if __name__ == '__main__':
    args = __parse_args()
    expected_results = get_expected_simulation_results()

    try:
        os.mkdir(TEMP_FOLDER_NAME)
    except:
        pass

    try:
        not_run_set, all_files = run_simulations(args)
        check_simulation_results(args, expected_results, not_run_set, all_files)
    finally:
        pass #subprocess.run(['rm', '-r', TEMP_FOLDER_NAME])
