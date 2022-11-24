# 70000032_CPU_96_lite_summary.txt

import argparse
from os import listdir, rename
from os.path import isfile, join

def parse_args():
    parser = argparse.ArgumentParser(
        prog='FIX DA FUCKUP!',
        description='Program to fix name fuckup!',
        epilog='Pully Porky'
    )

    parser.add_argument(
        '-f',
        '--folder',
        dest='folder_path',
        help='Path to folder with fuckup',
        type=str,
        required=True
    )

    args = parser.parse_args()
    return args
def main(args):
    only_files = [f for f in listdir(args.folder_path) if isfile(join(args.folder_path, f))]
    
    for file in only_files:
        split_file = file.split('_')
        new_file_name = f'{split_file[2]}_{split_file[1]}_{split_file[0]}_lite_summary.txt'
        rename(join(args.folder_path, file), join(args.folder_path, new_file_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)
