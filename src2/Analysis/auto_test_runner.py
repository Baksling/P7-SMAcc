import sys
import argparse
import os
import os.path as path
import tempfile as temp
import subprocess as cmd
import multiprocessing as mp
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import csv

CUDA_PARALLEL = "40,256"
CPU_CORES = str(mp.cpu_count())
THREAD_JOBS = str(mp.cpu_count() * 10)
CPU_PARALLEL = "1," + CPU_CORES
ALL = "ALL"
DEVICE_CHOICES = \
    {
        "GPU": ["-d", "0", "-b", CUDA_PARALLEL],
        "CPU": ["-d", "1", "-c", THREAD_JOBS, "-b", CPU_PARALLEL],
        "JIT": ["-d", "0", "-j", "-b", CUDA_PARALLEL],
        "SM": ["-d", "0", "-s", "-b", CUDA_PARALLEL],
        "PN": ["-d", "0", "-z", "-b", CUDA_PARALLEL],
        "PN-CPU": ["-d", "1", "-c", THREAD_JOBS, "-z", "-b", CPU_PARALLEL]
    }
ADDITIONAL_CHOICES = {ALL, "ALL-CPU", "ALL-GPU"}


def load_choice(choice) -> List[Tuple[str, list]]:
    if choice == ALL:
        return [(x, y) for x, y in DEVICE_CHOICES.items()]
    elif choice == "ALL-GPU":
        return [("GPU", DEVICE_CHOICES["GPU"]), ("JIT", DEVICE_CHOICES["JIT"]),
                ("SM", DEVICE_CHOICES["SM"]), ("PN", DEVICE_CHOICES["PN"])]
    elif choice == "ALL-CPU":
        return [("CPU", DEVICE_CHOICES["CPU"]), ("PN-CPU", DEVICE_CHOICES["PN-CPU"])]
    return [(choice, DEVICE_CHOICES[choice])]


# assuming lite summary
def load_time(filepath):
    with open(filepath, 'r') as f:
        return float(f.readline())


def build_args(folder: str, name: str, time: str, number: int, upscale: int, use_scale: bool,
               query: str | None) -> list:
    return ["-m", path.join(folder, name), "-x", time, "-n", str(number)] \
        + (["-u", str(upscale)] if use_scale else []) \
        + (["-q", query] if query is not None else [])


def run_model(default_args, settings_name: str, d_args, time: str, folder, cache_dir, file_name, numb,
              upscale, use_scale: bool = True, query: str = None) -> float:
    output_name = f"{path.join(str(cache_dir), file_name)}_U{upscale}_{settings_name}"
    print(f"Running: {file_name} w. {upscale}")
    cmd.call(default_args
             + d_args
             + build_args(folder, file_name, time, numb, upscale, use_scale, query)
             + ["-o", output_name])
    return load_time(output_name + "_lite_summary.txt")


def test_uppaal(uppal_binary: str, models) -> Dict[str, Dict[int, float]]:
    cmd.run([uppal_binary, ])


def test_smacc(binary: str, device, models, cache_dir, args) -> \
        Tuple[Dict[Tuple[str, str], Dict[int, float]], Dict[Tuple[str, str], float]]:
    device_args = load_choice(device)
    default_args = [binary, "-w", "l", "-v", "0"]
    result_dct: Dict[Tuple[str, str], Dict[int, float]] = {}
    single_dct: Dict[Tuple[str, str], float] = {}

    for settings, d_args in device_args:
        print(f"\nInitialising tests with {settings}:")
        # aloha
        aloha = {}
        aloha[2] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 2)
        aloha[5] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 5)
        aloha[10] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 10)
        aloha[25] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 25)
        aloha[50] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 50)
        aloha[100] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 100)
        aloha[250] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 250)
        # aloha[500] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 500)
        # aloha[1000] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "AlohaSingle.xml", 10240, 1000)
        result_dct[("aloha", settings)] = aloha

        # agant covid
        agent_covid = {}
        agent_covid[100] = run_model(default_args, settings, d_args, "100t", models, cache_dir,
                                     "agentBaseCovid_100_1.0.xml",
                                     10240, 100)
        agent_covid[500] = run_model(default_args, settings, d_args, "100t", models, cache_dir,
                                     "agentBaseCovid_500_5.0.xml",
                                     10240, 500)
        agent_covid[1000] = run_model(default_args, settings, d_args, "100t", models, cache_dir,
                                      "agentBaseCovid_1000_10.0.xml", 10240, 1000)
        agent_covid[5000] = run_model(default_args, settings, d_args, "100t", models, cache_dir,
                                      "agentBaseCovid_5000_50.0.xml", 10240, 5000)
        agent_covid[10000] = run_model(default_args, settings, d_args, "100t", models, cache_dir,
                                       "agentBaseCovid_10000_100.0.xml", 10240, 10000)
        # agent_covid[50000] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "agentBaseCovid_50000_500.0.xml", 10240, 50000)
        # agent_covid[100000] = run_model(default_args, settings, d_args, "100t", models, cache_dir, "agentBaseCovid_100000_1000.0.xml", 10240, 1000000)
        result_dct[("agent_covid", settings)] = agent_covid

        if settings == "SM":
            continue

        # csma
        csma = {}
        csma[2] = run_model(default_args, settings, d_args, "2000t", models, cache_dir, "CSMA_2.xml", 10240, 2,
                            use_scale=False)
        csma[5] = run_model(default_args, settings, d_args, "2000t", models, cache_dir, "CSMA_5.xml", 10240, 5,
                            use_scale=False)
        csma[10] = run_model(default_args, settings, d_args, "2000t", models, cache_dir, "CSMA_10.xml", 10240, 10,
                             use_scale=False)
        csma[25] = run_model(default_args, settings, d_args, "2000t", models, cache_dir, "CSMA_25.xml", 10240, 25,
                             use_scale=False)
        csma[50] = run_model(default_args, settings, d_args, "2000t", models, cache_dir, "CSMA_50.xml", 10240, 50,
                             use_scale=False)
        csma[100] = run_model(default_args, settings, d_args, "2000t", models, cache_dir, "CSMA_100.xml", 10240, 100,
                              use_scale=False)
        csma[250] = run_model(default_args, settings, d_args, "2000t", models, cache_dir, "CSMA_250.xml", 10240, 250,
                              use_scale=False)
        # csma[500] = run_model(default_args, d_args, "2000t", models, cache_dir, "CSMA_500.xml", 10240, 500, use_scale=False)
        # csma[1000] = run_model(default_args, d_args, "2000t", models, cache_dir, "CSMA_1000.xml", 10240, 1000, use_scale=False)
        result_dct[("csma", settings)] = csma

        fischer = {}
        fischer[2] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_2_2.xml", 10240, 2,
                               use_scale=False)
        fischer[5] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_5_2.xml", 10240, 5,
                               use_scale=False)
        fischer[10] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_10_2.xml", 10240,
                                10, use_scale=False)
        fischer[25] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_25_2.xml", 10240,
                                25, use_scale=False)
        fischer[50] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_50_2.xml", 10240,
                                50, use_scale=False)
        fischer[100] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_100_2.xml", 10240,
                                 100, use_scale=False)
        fischer[250] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_250_2.xml", 10240,
                                 250, use_scale=False)
        # fischer[500] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_500_2.xml", 10240, 500, use_scale=False)
        # fischer[1000] = run_model(default_args, settings, d_args, "300t", models, cache_dir, "fischer_500_2.xml", 10240, 1000, use_scale=False)
        result_dct[("fischer", settings)] = fischer

        # reaction_covid
        single_dct[("covid", settings)] = \
            run_model(default_args, settings, d_args, "100t", models, cache_dir, "covidmodelQueryI.xml", 10240, 1,
                      query="Template4.Query")

        single_dct[("bluetooth", settings)] = \
            run_model(default_args, settings, d_args, "5000t", models, cache_dir, "bluetoothNoParaSimas.cav.xml", 10240,
                      1,
                      query="Receiver.Reply")

        single_dct[("firewire", settings)] = \
            run_model(default_args, settings, d_args, "1000t", models, cache_dir, "firewireGoal.cav.xml", 10240, 1,
                      query="Node0.s5")

    return result_dct, single_dct


def print_output(filepath):
    results: Dict[Tuple[str, str, int], float] = {}
    with open(filepath, 'r') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            results[(row["system"], row["device"], int(row["scale"]))] = float(row["time"])

    data: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {}
    for (system, settings, scale), time in results.items():
        dct = data[system] = data.get(system, {})
        xs, ys = dct.get(settings, ([], []))
        data[system][settings] = (xs + [scale], ys + [time])

    for system, rs in data.items():
        plt.title(system)
        plt.xlabel("#processes")
        plt.ylabel("time [ms]")
        for settings, (xs, ys) in rs.items():
            plt.plot(xs, ys, label=settings)
        plt.legend()
        plt.show()


def write_output(results: Dict[Tuple[str, str], Dict[int, float]], output_file, show) -> None:
    # file
    with open(output_file, 'w') as f:
        f.write("system\tdevice\tscale\tms\n")
        for (system, dtype), rs in results.items():
            for scale, time in rs.items():
                f.write(f"{system}\t{dtype}\t{scale}\t{time}\n")

    if show:
        print_output(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--program", type=str, required=False, dest="program", help="Path to program binary")
    parser.add_argument("-d", "--device", type=str, choices=list(DEVICE_CHOICES.keys()) + list(ADDITIONAL_CHOICES), default=ALL,
                        required=False,
                        help="Method to run test using")
    parser.add_argument("-m", "--model", type=str, required=False, default=None, help="path to models to test")
    parser.add_argument("-c", "--cache", type=str, required=False, default=None, dest="temp",
                        help="Folder to store cache files (defualt = create temporary folder)")
    parser.add_argument("-o", "--output", type=str, required=False, default=None,
                        dest="output", help="path to store output")
    parser.add_argument("-u", "--uppaal", type=str, default=None, required=False,
                        dest="uppaal", help="Path to uppaal binary. If supplied, runs uppaal tests on uppaal too.")
    parser.add_argument("-g", "--graph", type=str, required=False, default=None,
                        dest="graph", help=
                        "Path to existing data to generate plots. If this option is supplied, no tests will be run.")
    parser.add_argument("-s", "--show", required=False, action='store_true', dest="show",
                        default=False, help="Whether to show plots or not. Default is not show")

    if len(sys.argv) < 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.graph is not None:
        if not path.exists(args.graph):
            raise argparse.ArgumentError(None, "Path to existing data source doesnt exist.")
        print_output(args.graph)
        return

    if args.program is None or not path.exists(args.program):
        raise argparse.ArgumentError(None, "Cannot find program binary")

    if args.device not in DEVICE_CHOICES and args.device not in ADDITIONAL_CHOICES:
        raise argparse.ArgumentError(None, "device type is not recognised device type. Should be in "
                                     + str(DEVICE_CHOICES) + " or a batch run, using one of the following: " 
                                     + str(ADDITIONAL_CHOICES) )

    if args.model is None or not path.exists(args.model):
        raise argparse.ArgumentError(None, "Cannot find models to test")

    if args.temp is None:
        args.temp_dir = temp.TemporaryDirectory(dir=os.getcwd(), ignore_cleanup_errors=True)
        args.temp = args.temp_dir.name
    else:
        args.temp_dir = None

    if args.output is None or path.exists(args.output):
        raise argparse.ArgumentError(None, "No output path supplied or file already exists")

    results, table_res = test_smacc(args.program, args.device, args.model, args.temp, args)
    if args.uppaal is not None:
        uppaal_dct = test_uppaal(args.uppaal, args.model)
        for system, dct in uppaal_dct.items():
            results[(system, "uppaal")] = dct
    write_output(results, args.output, args.show)

    if args.temp_dir:
        args.temp_dir.cleanup()
    # delete folder


if __name__ == "__main__":
    main()
