import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

BUFFER_SIZE = 10


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Data Performance Analysis",
        description="Program that is used to tell what permutation of blocks is optimal",
        epilog="Pully proky"
    )

    # IO GROUP ARGUMENTS
    io_parser = parser.add_argument_group(
        "IO parser"
    )

    io_parser.add_argument(
        '-f',
        '--folder',
        dest='folder_path',
        help='Path to where the results is stored',
        type=str,
        required=True
    )

    general_parser = parser.add_argument_group(
        'General Options'
    )

    general_parser.add_argument(
        '-mb',
        '--min_blocks',
        dest='blocks_start',
        help='Number of blocks to start from',
        type=int,
        default=1,
        required=False
    )

    general_parser.add_argument(
        '-b',
        '--blocks',
        dest='blocks_end',
        help='Upperbound of blocks to analyse',
        type=int,
        default=1024,
        required=False
    )

    general_parser.add_argument(
        '-mt',
        '--min_threads',
        dest='threads_start',
        help='Number of threads to start from',
        type=int,
        default=1,
        required=False
    )

    general_parser.add_argument(
        '-t',
        '--threads',
        dest='threads_end',
        help='Upperbound of threads to analyse',
        type=int,
        default=1024,
        required=False
    )

    args = parser.parse_args()
    return args


class data_wrapper():
    def __init__(self):
        self.min_time = 2_147_483_647
        self.min_time_file = 'Not defined'

        self.min_time_arr = [2_147_483_647 for _ in range(BUFFER_SIZE)]
        self.min_time_arr_str = ['' for _ in range(BUFFER_SIZE)]

        self.average_dict = {}
        self.min_average_time = 2_147_483_647.0
        self.lowest_average_block = 'Not defined'

    def __str__(self):
        buffer_str = f'{BUFFER_SIZE} lowest time combinations:\n'
        for i in range(BUFFER_SIZE):
            buffer_str += f'{i}: {self.min_time_arr_str[i]}\n'
        return f'Minimum time: {self.min_time_file} | Minimum Average Block: {self.lowest_average_block} \n{buffer_str}'


def analyse_data(args):
    result = data_wrapper()
    only_files = [f for f in listdir(args.folder_path) if isfile(join(args.folder_path, f))]

    for file in tqdm(only_files):

        # Guard to skip data we dont want
        file_split = file.split('_')
        thread = file_split[len(file_split) - 3]
        block = file_split[len(file_split) - 4]

        if block != 'CPU' and (int(block) < args.blocks_start or int(block) > args.blocks_end): continue
        if int(thread) < args.threads_start or int(thread) > args.threads_end: continue

        path = join(args.folder_path, file)
        with open(path, 'r') as f:
            data = int(f.readlines()[0].replace('\n', ''))
            block_str = f'Blocks: {block}, Threads: {thread}'
            if data < result.min_time:
                result.min_time = data
                result.min_time_file = block_str

            result.average_dict[block] = result.average_dict.get(block, []) + [data]

            for i in range(BUFFER_SIZE):
                if result.min_time_arr[i] > data and block_str != result.min_time_arr_str[i]:
                    result.min_time_arr[i] = data
                    result.min_time_arr_str[i] = block_str
                    break

    for key, val_lst in result.average_dict.items():
        value = 0 if len(val_lst) == 0 else float(sum(val_lst)) / float(len(val_lst))
        if value < result.min_average_time:
            result.min_average_time = value
            result.lowest_average_block = f'Block: {key}'
    return result


if __name__ == '__main__':
    args = parse_args()
    data = analyse_data(args)
    print(data)
