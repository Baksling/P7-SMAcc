import argparse
import subprocess
import time

# Workflow
# Make temp files for each configuration! or edit them
# Run UPPAAL thingy
# Capture time

TEMP_FILE_PATH = './dont_touch_this.xml'


def make_temp_file(args: argparse.ArgumentParser, formula: str) -> None:
    # Makes a copy with new configuration of UPPAAL File and returns path/filename of copied file
    in_query = False
    wrote_query = False
    with open(args.model_path, 'r') as model:
        with open(TEMP_FILE_PATH, 'w') as tmp_file:
            for line in model.readlines():
                if '<queries>' in line:
                    in_query = True
                    tmp_file.write(line)
                if not in_query:
                    tmp_file.write(line)
                else:
                    query = f'<query>\n<formula>{formula}</formula>\n<comment></comment>\n</query>\n'
                    if not wrote_query:
                        tmp_file.write(query)
                        wrote_query = True
                if '</queries>' in line:
                    tmp_file.write(line)
                    in_query = False


def run_uppaal_file(args: argparse.ArgumentParser) -> float:
    # Runs uppaal and returns time it took
    max_simulations = int(args.max_amount)
    interval = int(args.interval)
    formula = str(args.formula)

    with open(args.output_path, 'w') as out_f:
        for amount in range(1, max_simulations + 2, interval):
            new_formula = formula.replace('%', str(amount))
            make_temp_file(args, new_formula)

            parameters = [
                'bash', 'verifyta', TEMP_FILE_PATH, '-q', '-s'
            ]

            start_time = time.time()
            subprocess.run(parameters)
            out_f.write(f'{amount}\t{time.time() - start_time}\n')
            out_f.flush()

            subprocess.run(['rm', TEMP_FILE_PATH])


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Amount tester UPPAAL',
        description='A program to test the scalability of UPPAAL',
        epilog='PULLY PORKY!'
    )

    io_options = parser.add_argument_group(
        'IO Options'
    )

    io_options.add_argument(
        '-m',
        '--model',
        dest='model_path',
        help='Path to model file',
        type=str,
        required=True
    )

    io_options.add_argument(
        '-o',
        '--output',
        dest='output_path',
        help='path to put output file',
        type=str,
        required=True
    )

    general_options = parser.add_argument_group(
        'General Options'
    )

    general_options.add_argument(
        '-f',
        '--formula',
        dest='formula',
        help='The formula to be tested USE % as placeholder for simulation amount placement',
        type=str,
        required=True
    )

    general_options.add_argument(
        '-a',
        '--amount',
        dest='max_amount',
        help='Max amount of simulations to run!',
        type=int,
        required=True
    )

    general_options.add_argument(
        '-i',
        '--interval',
        dest='interval',
        help='Interval of simulations',
        type=int,
        required=True
    )

    _ = parser.parse_args()
    return _


if __name__ == '__main__':
    args = parse_args()
    run_uppaal_file(args)
