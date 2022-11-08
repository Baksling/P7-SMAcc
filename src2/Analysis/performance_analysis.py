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
        '-b',
        '--max_block',
        dest='max_block',
        help='Number of Max blocks',
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

    args = parser.parse_args()
    return args


def analyse_performance(args) -> None:
    blocks: int = args.max_block
    block_threads: int = args.max_threads
    cpu_count: int = args.cpu_threads
    max_gpu_compute: int = args.max_gpu_compute 

    generator = ((block, thread) for block in range(1, blocks + 1) for thread in range(1, block_threads + 1) if
                 (block * thread <= max_gpu_compute))

    for block, thread in generator:
        file_path = f'{args.output_file}_GPU_{block}_{thread}'
        amount = math.ceil((args.simulation_amount / (block * thread)))

        subprocess.run([
            args.simulation_file,
            '-m', args.model_file,
            '-b', str(block),
            '-t', str(thread),
            '-a', str(amount),
            '-c', '1',
            '-d', '0',
            '-w', str(3),
            '-o', file_path,
            '-y', '0',
            '-v', '1'
        ])

    for cpu_core in range(1, cpu_count + 1):
        file_path = f'{args.output_file}_CPU_{cpu_core}'

        amount = math.ceil((args.simulation_amount / cpu_count))

        subprocess.run([
            args.simulation_file,
            '-m', args.model_file,
            '-b', str(1),
            '-t', str(cpu_count),
            '-a', str(amount),
            '-c', '1',
            '-d', '1',
            '-w', str(3),
            '-o', file_path,
            '-y', '0',
            '-v', '1',
            '-u', str(cpu_count)
        ])


if __name__ == "__main__":
    arguments = parse_arguments()
    analyse_performance(arguments)
