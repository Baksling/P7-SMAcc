import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from typing import List, Tuple
from os import listdir
from os.path import isfile, join
import argparse

COLORS = ['r', 'g', 'b', 'k']
COLORS_LEN = len(COLORS)

def plot_points(p_list: List[Tuple[int, int, int]], dims: Tuple[int, int, int], args_) -> None:
    ax = plt.figure().add_subplot(projection='3d')
    block_dim, thread_dim, time_dim = dims

    x, y, z, color = [], [], [], []

    for p in range(len(p_list)):
        block, thread, time = p_list[p]
        if block % args_.interval != 0: continue

        color.append(COLORS[block % COLORS_LEN])

        x.append(block)
        y.append(thread)
        z.append(time)

    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x, y) points are plotted on the x and z axes.
    ax.scatter(x, y, zs=z, zdir='z', c=color, label='points in (x, z)')

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlim(0, block_dim)
    ax.set_ylim(1, thread_dim)
    ax.set_zlim(0, time_dim)
    ax.set_xlabel('Blocks')
    ax.set_ylabel('Threads')
    ax.set_zlabel('Time (ms)')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35, roll=0)

    plt.show()


def plot_lines(p_list: List[Tuple[int, int, int]], dims: Tuple[int, int, int], args_) -> None:
    block_dim, thread_dim, time_dim = dims[0] - args_.min_blocks, dims[1] - args_.min_threads, dims[2]

    v_lst = [[] for i in range(block_dim + 1)]

    for p in range(len(p_list)):
        block, thread, time = p_list[p]

        block_idx = block - args_.min_blocks
        v_lst[block_idx].append((thread, time))

    for lst_idx in range(len(v_lst)):
        lst = v_lst[lst_idx]
        v_lst[lst_idx] = sorted(lst, key=lambda t: t[0])

    lines = [[] for i in range(thread_dim + 1)]

    labels_start = ["CPU"] if args_.min_blocks <= 0 else []
    labels = labels_start + [f"#Block {block + args_.interval}" for block in range(args_.min_blocks, block_dim) if (args_.interval > 0 and (block % args_.interval == 0) and block + args_.interval <= block_dim)]

    for thread_idx in range(thread_dim):
        for block_idx in range(block_dim + 1):
            if (block_idx % args_.interval != 0): continue

            if thread_idx == 0:  # This is needed for plot thingy!
                lines[thread_idx].append(v_lst[block_idx][thread_idx][1])
            data = v_lst[block_idx][thread_idx][1]
            lines[thread_idx + 1].append(data)

    for x in lines:
        print(x)
    for x in labels:
        print(x)

    if args_.show_label > 0: plt.plot(lines, label=labels)
    else:                    plt.plot(lines)

    # Make legend, set axes limits and labels
    plt.legend()
    plt.xlim(1, thread_dim)
    plt.ylim(0, time_dim)
    plt.xlabel('Threads')
    plt.ylabel('Time (ms)')

    plt.show()


def get_data(folder_path: str, args_, dims, filter_str: str = None) -> Tuple[int, List[Tuple[int, int, int]]]:
    result: List[Tuple[int, int, int]] = []
    max_time_ = 0
    only_files = [f for f in listdir(folder_path) if
                  isfile(join(folder_path, f)) and (filter_str is None or filter_str == "" or f.endswith(filter_str))]

    block_dim, thread_dim, time_dim = dims

    for file in only_files:
        tmp = file.split('_')
        tmp_len = len(tmp)
        is_gpu = tmp[tmp_len - 5] == "GPU"
        block, thread, time = int(tmp[tmp_len - 4]) if is_gpu else 0, int(tmp[tmp_len - 3].split('.')[0]), 0

        if block > block_dim or block < args_.min_blocks: continue
        if thread > thread_dim or thread < args_.min_threads: continue
        if time > time_dim or time < args_.min_time: continue

        with open(join(folder_path, file)) as f:
            lines = f.readlines()
            time = int(lines[0].replace('\n', ''))

        max_time_ = max(time, max_time_)

        mode = args_.mode
        if is_gpu and (mode == 0 or mode == 2):
            result.append((block, thread, time))
        elif not is_gpu and (mode == 1 or mode == 2):
            result.append((0, thread, time))

    return max_time_, result


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
        '-mb',
        '--min_blocks',
        dest='min_blocks',
        help='Minimum number of blocks',
        type=int,
        default=0,
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

    option_parser.add_argument(
        '-mt',
        '--min_threads',
        dest='min_threads',
        help='Minimum number of threads',
        type=int,
        default=1,
        required=False
    )

    option_parser.add_argument(
        '-p',
        '--max_time',
        dest='max_time',
        help='Max time cutoff',
        type=int,
        default=-1,
        required=False
    )

    option_parser.add_argument(
        '-mp',
        '--min_time',
        dest='min_time',
        help='Minimum time',
        type=int,
        default=0,
        required=False
    )
    
    option_parser.add_argument(
        '-l',
        '--label',
        dest='show_label',
        help='Set to 1 if you want to see labels',
        type=int,
        default=0,
        required=False
    )

    option_parser.add_argument(
        '-i',
        '--interval',
        dest='interval',
        help='Interval amount',
        type=int,
        default=1,
        required=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # dummy_data = [(1, 2, 5), (1, 1, 10), (1, 3, 4), (2, 3, 9), (2, 1, 5), (2, 2, 3)]

    args = get_args()
    dims = (args.block_dim, args.thread_dim, args.max_time)
    max_time, data = get_data(args.folder_path, args, dims, '.txt')
    dims = (dims[0], dims[1], max_time)
    # test_dims = (args.block_dim, args.thread_dim, 15)

    plot_points(data, dims, args)

    plot_lines(data, dims, args)
