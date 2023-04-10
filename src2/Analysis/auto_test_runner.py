import sys
import argparse
import os
import os.path as path
import tempfile as temp
import subprocess as cmd

ALL = "ALL"
DEVICE_CHOICES = \
    {
        "GPU": ["-d", "0"],
        "CPU": ["-d", "1", "-c", "0"],
        "JIT": ["-d", "0", "-j"],
        "SM": ["-d", "0", "-s"],
        "PN": ["-d", "0", "-z"],
        "PN-CPU": ["-d", "0", "-c", "0", "-z"],
        ALL: []
    }


def run_smacc(binary: str, device, models, cache_dir, output, args):
    device_args = list(DEVICE_CHOICES.values()) if device == ALL else [DEVICE_CHOICES[device]]
    default_args = [binary, "-w", "wfl"]
    
    for args in device_args:
        #agent covid
        cmd.call(default_args + args + ["-m", path.join(models, "agentCovid.xml")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--program", type=str, required=False, dest="program", help="Path to program binary")
    parser.add_argument("-d", "--device", type=str, choices=DEVICE_CHOICES, default="ALL", required=False,
                        help="Method to run test using")
    parser.add_argument("-m", "--model", type=str, required=False, default=None, help="path to models to test")
    parser.add_argument("-c", "--cache", type=str, required=False, default=None,
                        help="Folder to store cache files (defualt = create temporary folder)")
    parser.add_argument("-o", "--output", type=str, required=False, default=None, help="path to store output")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.program is None or not path.exists(args.program):
        raise argparse.ArgumentError("--program", "Cannot find program binary")

    if args.device not in DEVICE_CHOICES:
        raise argparse.ArgumentError("--decive", "device type is not recognised device type. Should be in "
                                     + str(DEVICE_CHOICES))

    if args.model is None or not path.exists(args.model):
        raise argparse.ArgumentError("--model", "Cannot find models to test")

    if args.temp is None:
        args.is_temp_output = True
        args.temp = temp.TemporaryDirectory(ignore_cleanup_errors=True)
    else:
        args.is_temp_output = False

    if args.output is None or not path.exists(args.output):
        raise argparse.ArgumentError("--output", "No output path supplied")

    run(args.program, args.device, args.model, args.temp, args.output, args)


if __name__ == "__main__":
    main()
