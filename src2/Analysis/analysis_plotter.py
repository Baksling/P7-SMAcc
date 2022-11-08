import matplotlib.pyplot as plt
from typing import List, Tuple
from os import listdir
from os.path import isfile, join
import argparse

COLORS = ['r', 'g', 'b', 'k']
COLORS_LEN = len(COLORS)


def plot_points(p_list: List[Tuple[int, int, int]], dims: Tuple[int, int, int]) -> None:
    ax = plt.figure().add_subplot(projection='3d')
    block_dim, thread_dim, time_dim = dims

    x, y, z, color = [], [], [], []

    for p in range(len(p_list)):
        block, thread, time = p_list[p]
        
        if block > block_dim: continue
        if thread > thread_dim: continue
        if time_dim > time_dim: continue
        
        color.append(COLORS[block % COLORS_LEN])

        x.append(thread)
        y.append(block)
        z.append(time)

    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x, y) points are plotted on the x and z axes.
    ax.scatter(x, y, zs=z, zdir='z', c=color, label='points in (x, z)')

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlim(0, thread_dim)
    ax.set_ylim(0, block_dim)
    ax.set_zlim(0, time_dim)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Blocks')
    ax.set_zlabel('Time')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35, roll=0)

    plt.show()


def get_data(folder_path: str, args, filter_str: str = None) -> Tuple[int, List[Tuple[int, int, int]]]:
    result: List[Tuple[int, int, int]] = []
    max_time = 0
    only_files = [f for f in listdir(folder_path) if
                  isfile(join(folder_path, f)) and (filter_str is None or filter_str == "" or f.endswith(filter_str))]

    for file in only_files:
        tmp = file.split('_')
        tmp_len = len(tmp)
        is_gpu = tmp[tmp_len - 5] == "GPU"
        block, thread, time = tmp[tmp_len - 4], tmp[tmp_len - 3].split('.')[0], 0

        with open(join(folder_path, file)) as f:
            lines = f.readlines()
            time = int(lines[0].replace('\n', ''))
            
        max_time = max(time, max_time)

        mode = args.mode
        if       is_gpu and (mode == 0 or mode == 2): result.append((int(block), int(thread), time))
        elif not is_gpu and (mode == 1 or mode == 2): result.append((0, int(thread), time))

    return max_time, result


def get_args():
    parser = argparse.ArgumentParser(
        prog="Data Analyser",
        description="Program is used to show data from Simulations!",
        epilog="Pully Porky!"
    )

    # IO Arguments
    io_parser = parser.add_argument_group(
        "IO Arguments"
    )

    io_parser.add_argument(
        '-f',
        '--folder_path',
        dest="folder_path",
        help="Path to folder with data",
        type=str,
        required=True
    )

    # Option Arguments
    option_parser = parser.add_argument_group(
        "Option Arguments"
    )

    option_parser.add_argument(
        '-m',
        '--mode',
        dest='mode',
        help='What to analyse: (0, GPU), (1, CPU), (2, BOTH)',
        type=int,
        default=0,
        required=False
    )
    
    option_parser.add_argument(
        '-b',
        '--blocks',
        dest='block_dim',
        help='Number of blocks shown',
        type=int,
        default=32,
        required=False
    )
    
    option_parser.add_argument(
        '-t',
        '-threads',
        dest='thread_dim',
        help='Number of threads shown',
        type=int,
        default=32,
        required=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # dummy_data = [(1, 1, 10), (1, 2, 5)]

    args = get_args()
    max_time, data = get_data(args.folder_path, args, '.txt')
    dims = (args.block_dim, args.thread_dim, max_time)

    plot_points(data, dims)
