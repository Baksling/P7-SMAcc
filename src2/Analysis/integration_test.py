import argparse
import os
import subprocess
from os import listdir
from os.path import isfile, join

TEMP_FOLDER_NAME = './tmp_results'
POST_FIX_TMP_NAME = '_results.csv'


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
        help='Number of total simulations to run default = 1B',
        type=int,
        default=1000000000,
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

    args = parser.parse_args()
    return args


def run_simulations(args_) -> None:
    simulator_path = args_.simulation_path
    folder_path = args_.folder_path

    only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for file in only_files:
        if file == 'random_test.xml': continue
        print("RUNNING", file)
        file_path = join(folder_path, file)
        amount = float(32 * 512) / float(args_.amount)

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


def check_simulation_results(args_, expected_results) -> None:
    def within_range(expected, actual, variance) -> bool:
        return expected + variance > actual > expected - variance and 100.0 >= actual >= 0.0  # actual < expected + variance and actual > expected - variance <-- OLD BUT SECURE

    only_files = [f for f in listdir(TEMP_FOLDER_NAME) if isfile(join(TEMP_FOLDER_NAME, f))]

    for file in only_files:
        file_path = join(TEMP_FOLDER_NAME, file)
        expected_result = expected_results[f'{file.replace(POST_FIX_TMP_NAME, "")}']
        actual_result = 0.0

        with open(file_path, 'r') as f:
            value = float(f.readlines()[0].replace('\n', ''))

            actual_result = value

        if within_range(expected_result, actual_result, args_.variance):
            print(file, 'PASSED')
        else:
            print(file, 'FAILED')


def get_expected_simulation_results() -> dict[str, float]:
    return {
        'clock_var': 100.0,
        'dicebase': 16.6,
        'dicebaseUnfair': 6.3,
        'random_test': 14.4,
        'rare_events': 0.0001,
        'rate_test': 33.45,
        'var_test': 39.0
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
        pass# subprocess.run(['rm', '-r', TEMP_FOLDER_NAME])