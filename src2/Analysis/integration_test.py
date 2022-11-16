import argparse
import os
import subprocess
from os import listdir
from os.path import isfile, join
from math import ceil
from typing import Dict

TEMP_FOLDER_NAME = './tmp_results'
POST_FIX_TMP_NAME = '_results.tsv'


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
        default=50,
        required=False
    )

    args = parser.parse_args()
    return args


def run_simulations(args_) -> None:
    simulator_path = args_.simulation_path
    folder_path = args_.folder_path

    only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for file in only_files:
        print("RUNNING", file)
        file_path = join(folder_path, file)
        amount = int(ceil(float(args_.amount) / float(32 * 512)))
        subprocess.run([
            simulator_path,
            '-m', file_path,
            '-b', str(32),
            '-t', str(512),
            '-a', str(amount),
            '-c', '1',
            '-d', '0',
            '-w', 'r',
            '-o', f'{join(TEMP_FOLDER_NAME, file.replace(".xml", ""))}',
            '-y', str(args_.use_time),
            '-s', str(args_.max_progression),
            '-p', str(args_.max_progression),
            '-v', '1'
        ])


def check_simulation_results(args_, _expected_results) -> None:
    def within_range(expected, actual, variance) -> bool:
        return expected + variance > actual > expected - variance  # actual < expected + variance and actual > expected - variance <-- OLD BUT SECURE

    only_files = [f for f in listdir(TEMP_FOLDER_NAME) if isfile(join(TEMP_FOLDER_NAME, f))]
    print(only_files)
    for file in only_files:
        file_path = join(TEMP_FOLDER_NAME, file)
        file_results = _expected_results[f'{file.replace(POST_FIX_TMP_NAME, "")}']
        expected_value = file_results.value
        expected_time = file_results.time
        actual_result = 0.0
        actual_time = 0.0
        try:
            with open(file_path, 'r') as f:
                data_line = f.readlines()[0].replace('\n', '')
                tmp_split = data_line.split('\t')

                actual_result = float(tmp_split[0])
                actual_time = int(tmp_split[1])

            output_str = f'{file} '
            if within_range(expected_value, actual_result, args_.variance) and \
                    (args_.time_variance == 0 or within_range(expected_time, actual_time, args_.time_variance)):
                output_str += f'PASSED'
            else:
                output_str += 'FAILED'
            output_str += f' - Value variance: {expected_value - actual_result} percent | Time variance {expected_time - actual_time} [ms]'
            print(output_str)
        except:
            print(file, 'FAILED - EXCEPT')


def get_expected_simulation_results() -> dict[str, value_checker]:
    return {
        'clock_var': value_checker(100.0, 270),
        'dicebase': value_checker(16.6, 440),
        'dicebaseUnfair': value_checker(6.3, 1350),
        'random_test': value_checker(14.4, 122),
        'rare_events': value_checker(0.0001, 1000),
        'rate_test': value_checker(33.45, 155),
        'var_test': value_checker(47.4, 10300)
    }


if __name__ == '__main__':
    args = __parse_args()
    expected_results = get_expected_simulation_results()

    try:
        os.mkdir(TEMP_FOLDER_NAME)
    except:
        pass

    try:
        run_simulations(args)
        check_simulation_results(args, expected_results)
    finally:
        pass  # subprocess.run(['rm', '-r', TEMP_FOLDER_NAME])
