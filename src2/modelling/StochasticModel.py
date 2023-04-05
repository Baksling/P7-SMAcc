import sys
import argparse
from SMAccDomain import *
from SMAccBuilder import SMAccBuilder


class StochasticModel:
    def setup_model(self) -> Network:
        raise NotImplementedError(
            "method 'setup_model' required to be overwritten, and return a object of type network")

    def additional_args(self, cli_args: argparse.ArgumentParser):
        return cli_args

    def parse_args(self):
        cli_parser = argparse.ArgumentParser()

        cli_parser.add_argument("--exe", type=str, dest='exe', required=True,
                                help="Path to engine executable. Required.")
        cli_parser.add_argument("--epsilon", "-e", type=float, dest='epsilon', required=True,
                                help="margin of error as percentage (e.g. 5%% = 0.05).")
        cli_parser.add_argument("--alpha", "-a", type=float, dest="alpha", required=True,
                                help="degree of confidence as percentage (e.g. 5%% = 0.05)")
        cli_parser.add_argument("--device", "-d", choices=['CPU', 'GPU'], type=str, dest='device', required=False,
                                help="Device to use for model checking (GPU or CPU)")
        cli_parser.add_argument("--units", "-x", type=str, dest='units', required=True,
                                help="Maximum number of steps or time to simulate (e.g. 100t = 100 time units / 100s = 100 steps")
        cli_parser.add_argument("--blocks", "-b", type=int, dest='blocks', required=True,
                                help="Number of blocks to use on GPU.")
        cli_parser.add_argument("--threads", "-t", type=int, dest='threads', required=True,
                                help="Number of threads to use. If on GPU, parameter is per block. If on CPU, parameter is total threads.")
        cli_parser.add_argument("--jit", '-j', action='store_true', dest='jit', default=False, required=False,
                                help="Utilise JIT compilation of expression on GPU (does nothing on CPU)")
        cli_parser.add_argument("-shared", '-s', action='store_true', dest='sm', default=False, required=False,
                                help="store model in shared memory on GPU (does nothing on CPU), only possible for small/medium models.")
        if len(sys.argv) <= 1 or "-h" in sys.argv:
            cli_parser.print_help()
            sys.exit(0)

        return cli_parser.parse_args()

    def run_cli(self):
        """Run SMC using args loaded through CLI"""
        args = self.parse_args()
        self.run(
            args.exe,
            epsilon=args.epsilon,
            alpha=args.alpha,
            blocks=args.blocks,
            threads=args.threads,
            use_gpu=args.device == "GPU",
            units=args.units,
            use_jit=args.jit,
            use_sm=args.sm,
            cpu_threads=args.threads
        )

    def run(self, executable_path: str, *,
            epsilon=0.005, alpha=0.1, blocks=40, threads=256,
            use_gpu=True, units='100t', use_jit=False, use_sm=False,
            write_mode='c', silent=False, cpu_threads=1, output_name="output"):
        """Run SMC with args through method"""
        model = self.setup_model()
        builder = SMAccBuilder(executable_path)

        builder.build_and_run(model,
                              epsilon=epsilon,
                              alpha=alpha,
                              blocks=blocks,
                              threads=threads,
                              use_gpu=use_gpu,
                              end_criteria=units,
                              use_jit=use_jit,
                              use_sm=use_sm,
                              write_mode=write_mode,
                              silent=silent,
                              output_name=output_name,
                              cpu_threads=cpu_threads)
