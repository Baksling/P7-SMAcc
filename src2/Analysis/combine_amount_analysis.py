import argparse
from os import listdir
from os.path import isfile, join

def arg_parse():
    parser = argparse.ArgumentParser(
        prog='Simulation Amount Post Process',
        description='Used to make a single tsv file, with all the data',
        epilog='Pully Porky'
    )

    # IO Options
    io_options = parser.add_argument_group(
        'IO Options'
    )

    io_options.add_argument(
        '-f',
        '--folder_path',
        dest='folder_path',
        help='Path to folder with files from simulation_amount_analysis.py',
        type=str,
        required=True
    )

    io_options.add_argument(
        '-o',
        '--output',
        dest='output_file',
        help='Path to Output file',
        type=str,
        required=True
    )

    args = parser.parse_args()
    return args

def combine_files(args):
    only_files = [f for f in listdir(args.folder_path) if isfile(join(args.folder_path, f))]

    tmp_data = []

    for file in only_files:
        file_split = file.split('_')
        device = file_split[len(file_split) - 4]
        run_idx = int(file_split[len(file_split) - 3])


        file_path = join(args.folder_path, file)
        with open(file_path, 'r') as f:
            data = int(f.readlines()[0].replace('\n', ''))
            tmp_data.append((run_idx, data))
        
    tmp_data.sort(key=lambda x: x[0])

    with open(args.output_file, 'w') as f:
        for run_idx, time_ms in tmp_data:
            f.write(f'{run_idx}\t{time_ms}\n')



if __name__ == '__main__':
    args = arg_parse()
    combine_files(args)
