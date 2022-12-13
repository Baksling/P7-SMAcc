import argparse
import subprocess
import math
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="CUDA Analysis Program",
        description="Program is used to analyse different parameters for CUDA Simulation",
        epilog="Pully proky"
    )

    # IO GROUP ARGUMENTS
    io_parser = parser.add_argument_group(
        "IO parser"
    )

    io_parser.add_argument(
        '-s',
        '--simulation',
        dest='simulation_file',
        help='Path to simulation file',
        type=str,
        required=True
    )

    io_parser.add_argument(
        '-m',
        '--model',
        dest='model_file',
        help='Path to XML file to simulate',
        type=str,
        required=False
    )

    io_parser.add_argument(
        '-o',
        '--output',
        dest='output_file',
        help='Path to where the results should be stored',
        type=str,
        required=True
    )

    # OPTION GROUP ARGUMENTS
    option_parser = parser.add_argument_group(
        "Option Parser"
    )

    option_parser.add_argument(
        '-p',
        '--mode',
        dest='mode',
        help='0: GPU 1: CPU 2: BOTH',
        type=int,
        default=0,
        required=False
    )

    option_parser.add_argument(
        '-mb',
        '--min_block',
        dest='min_block',
        help='Minimum number of blocks to use',
        type=int,
        default=1,
        required=False
    )

    option_parser.add_argument(
        '-b',
        '--max_block',
        dest='max_block',
        help='Number of Max blocks',
        type=int,
        default=1,
        required=False
    )

    option_parser.add_argument(
        '-mt',
        '--min_threads',
        dest='min_threads',
        help='Minimum number of threads to use',
        type=int,
        default=1,
        required=False
    )

    option_parser.add_argument(
        '-t',
        '--max_threads',
        dest='max_threads',
        help='Number of Max threads',
        type=int,
        default=1,
        required=False
    )

    option_parser.add_argument(
        '-c',
        '--max_gpu_compute',
        dest='max_gpu_compute',
        help='Max number of threads that can be called by GPU',
        type=int,
        default=(128 * 512),
        required=False
    )

    option_parser.add_argument(
        '-mco',
        '--min_cpu_threads',
        dest='min_cpu_threads',
        help='Mimumnum number of threads to use',
        type=int,
        default=1,
        required=False
    )

    option_parser.add_argument(
        '-co',
        '--cpu_threads',
        dest='cpu_threads',
        help='Number of CPU threads to test',
        type=int,
        default=1,
        required=False
    )

    option_parser.add_argument(
        '-a',
        '--simulation_amount',
        dest='simulation_amount',
        help='Number of simulations to run each time',
        type=int,
        default=1,
        required=True
    )

    option_parser.add_argument(
        '-sm',
        '--shared_memory',
        dest='use_shared',
        help='Use shared Memory',
        type=int,
        default=0,
        required=False
    )
    
    option_parser.add_argument(
        '-j',
        '--jit',
        dest='use_jit',
        help='Use Jit',
        type=int,
        default=0,
        required=False
    )

    option_parser.add_argument(
        '-j',
        '--jit',
        dest='use_jit',
        help='Set it to 1 if you want to use jit',
        type=int,
        default=0,
        required=False
    )

    args = parser.parse_args()
    return args


def analyse_performance(args) -> None:
    min_blocks, blocks = args.min_block, args.max_block
    min_threads, block_threads = args.min_threads, args.max_threads
    min_cpu_count, cpu_count = args.min_cpu_threads, args.cpu_threads
    max_gpu_compute: int = args.max_gpu_compute

    generator = ((block, thread) for block in range(min_blocks, blocks + 1) for thread in
                 range(min_threads, block_threads + 1, 32) if
                 (block * thread <= max_gpu_compute))

    if args.mode == 0 or args.mode == 2:
        for block, thread in generator:
            file_path = f'{args.output_file}_GPU_{1 if block == 0 else block}_{thread}'
            # amount = math.ceil((args.simulation_amount / (block * thread)))

            # subprocess.run([
            #     args.simulation_file,
            #     '-m', args.model_file,
            #     '-b', str(block),
            #     '-t', str(thread),
            #     '-a', str(amount),
            #     '-c', '1',
            #     '-d', '0',
            #     '-w', 'l',
            #     '-o', file_path,
            #     '-y', '0',
            #     '-v', '1'
            # ])

            parameters = [
                args.simulation_file,
                '-m', args.model_file,
                '-b', f'{block},{1 if thread == 0 else thread}',
                '-n', f'{args.simulation_amount}',
                '-c', '1',
                '-d', '0',
                '-w', 'l',
                '-o', file_path,
                '-x', '100t',
                '-v', '0'
            ]
            if args.use_shared == 1:
                parameters.append('-s')

            if args.use_jit == 1:
                parameters.append('-j')

            subprocess.run(parameters)

    if args.mode == 1 or args.mode == 2:
        for cpu_core in range(min_cpu_count, cpu_count + 1):
            file_path = f'{args.output_file}_CPU_{cpu_core}'

            # amount = math.ceil((args.simulation_amount / cpu_core))

            # subprocess.run([
            #     args.simulation_file,
            #     '-m', args.model_file,
            #     '-b', str(1),
            #     '-t', str(cpu_core),
            #     '-a', str(amount),
            #     '-c', '1',
            #     '-d', '1',
            #     '-w', 'l',
            #     '-o', file_path,
            #     '-y', '0',
            #     '-v', '1',
            #     '-u', str(cpu_core)
            # ])

            subprocess.run([
                args.simulation_file,
                '-m', args.model_file,
                '-b', f'1,{cpu_core}',
                '-n', f'{args.simulation_amount}',
                '-c', f'{cpu_core}',
                '-d', '1',
                '-w', 'l',
                '-o', file_path,
                '-x', '100t',
                '-v', '0'
            ])


if __name__ == "__main__":
    arguments = parse_arguments()
    analyse_performance(arguments)
