import sys
import argparse
import os
import os.path as path
import tempfile as temp
import subprocess as cmd
import multiprocessing as mp
# import matplotlib.pyplot as plt
import csv
import time as timer

N_SAMPLES = 10
CUDA_PARALLEL = "40,256"
GPU_POWER = 250
CPU_POWER = 450
CACUTS_PLOT_CUTOFF = 2
CPU_CORES = str(mp.cpu_count())
THREAD_JOBS = str(mp.cpu_count() * 10)
CPU_PARALLEL = "1," + CPU_CORES
ALL = "ALL"
FULL_OUTPUT_FILENAME = "all_results.tsv"
BASELINE = "BASELINE"
TOTAL_SIMS = 10240
DEFAULT_UPPAAL_EXPERIMENT_FOLDERNAME = "UPPAALexperiments"
DEVICE_CHOICES = \
    {
        "BASELINE": ["-d", "1", "-b", CPU_PARALLEL, "-c", "1"],
        "GPU": ["-d", "0", "-b", CUDA_PARALLEL],
        "CPU": ["-d", "1", "-b", CPU_PARALLEL],
        "JIT": ["-d", "0", "-j", "-b", CUDA_PARALLEL],
        "SM": ["-d", "0", "-s", "-b", CUDA_PARALLEL],
        "PN": ["-d", "0", "-z", "-b", CUDA_PARALLEL],
        "PN-CPU": ["-d", "1", "-z", "-b", CPU_PARALLEL]
    }
ADDITIONAL_CHOICES = {ALL, "ALL-CPU", "ALL-GPU"}
REF_DEVICE = "CPU"
CPU_CHOICES = {"CPU", "PN-CPU"}
GPU_CHOICES = ["GPU", "JIT", "PN", "SM"]

def avg(lst):
    return sum(lst) / len(lst)


def power_ratio(device, time, comp_time):
    my_power = GPU_POWER if device in GPU_CHOICES else CPU_POWER
    if device == "uppaal":
        my_power = CPU_POWER / mp.cpu_count()
    return (time * my_power) / (comp_time * CPU_POWER)


class TestResults:
    def __init__(self, run_time, total_time, hit):
        self.run_time = run_time
        self.total_time = total_time
        self.hit = hit

    def __iter__(self):
        yield self.run_time
        yield self.total_time
        yield self.hit

class TimeoutReference:
    def __init__(self, v = False) -> None:
        self._time_out = v
    
    def is_timed_out(self):
        return self._time_out
    
    def set_timeout(self):
        self._time_out = True

def load_choice(choice):
    if choice == ALL:
        return [(x, y) for x, y in DEVICE_CHOICES.items()]
    elif choice == "ALL-GPU":
        return [("GPU", DEVICE_CHOICES["GPU"]), ("JIT", DEVICE_CHOICES["JIT"]),
                ("SM", DEVICE_CHOICES["SM"]), ("PN", DEVICE_CHOICES["PN"])]
    elif choice == "ALL-CPU":
        return [("BASELINE", DEVICE_CHOICES["BASELINE"]),
                ("CPU", DEVICE_CHOICES["CPU"]),
                ("PN-CPU", DEVICE_CHOICES["PN-CPU"])]
    return [(choice, DEVICE_CHOICES[choice])]


def threads(settings, thread_count):
    return ["-c", str(thread_count)] if settings != BASELINE else ["-c", str(1)]


def build_args(folder, name, time_arg, number, upscale, use_scale,
               query):
    return ["-m", path.join(folder, name), "-x", time_arg, "-n", str(number)] \
        + (["-u", str(upscale)] if use_scale else []) \
        + (["-q", query] if query is not None else [])


def run_model(default_args, settings_name, d_args, time_arg, args, file_name, numb,
              upscale, use_scale=True, query=None, q_index=0, timeout_p: TimeoutReference = TimeoutReference(False)):
    if timeout_p.is_timed_out():
        return None
    folder, cache_dir = args.model, args.temp
    output_name = f"{path.join(str(cache_dir), file_name)}_U{upscale}_{settings_name}"
    print(f"Running: {file_name} w. {upscale} (at. {timer.strftime('%H:%M:%S', timer.localtime())})")

    # assuming lite summary
    def load_time(filepath):
        with open(filepath, 'r') as f:
            return float(f.readline()) / float(1000)  # convert ms to s

    def load_reach(file, index):
        with open(file, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.replace("\n", "")
                p, hit = line.split('\t')
                if int(p) == index:
                    return float(hit)
            return 0.0

    time_lst, out_lst, reach_lst = [], [], []
    try:
        
        for i in range(args.n_samples):
            print(f"\t running test #{i}")
            call_args = args.additional_args \
                    + default_args \
                    + d_args \
                    + threads(settings_name, args.threads) \
                    + build_args(folder, file_name, time_arg, numb, upscale, use_scale, query) \
                    + ["-o", output_name]
            start = timer.time()
            cmd.run(call_args,
                    timeout=args.timeout, check=True, stderr=cmd.STDOUT)
            out_time = (timer.time() - start)
            time_file = output_name + "_lite_summary.txt"
            reach_file = output_name + "_reach.tsv"
            
            time_lst.append(time_file)
            out_lst.append(out_time)
            reach_lst.append(load_reach(reach_file, q_index))
        
    except cmd.TimeoutExpired:
        print("timed out...")
        timeout_p.set_timeout()
        return None
    
    return TestResults(avg(time_lst), avg(out_lst), avg(reach_lst))



def test_uppaal(binary, args):
    # -> Tuple[
    #     Dict[str, Dict[int, test_result | None]],
    #     Dict[str, test_result | None]
    # ]:
    result_dct = {}
    single_dct = {}
    print(f"\nInitialising tests with uppaal:")

    def run_uppaal(model, timeout_p: TimeoutReference = TimeoutReference(False)):
        if timeout_p.is_timed_out():
            return None
        print("running uppaal on: " + model)
        time_lst, out_lst, reach_lst = [], [], []
        try:
            for i in range(args.n_samples):            
                print(f"\t running test #{i}")
                call_args = args.additional_args + [binary, path.join(args.model, model), "-q", "-s"]
                start = timer.time()
                cmd.run(call_args,
                        capture_output=True, text=True, check=True, timeout=args.timeout)
                total = (timer.time() - start)
                
                time_lst.append(total)
                out_lst.append(total)
                reach_lst.append(0.0) #ignore
            
        except cmd.TimeoutExpired:
            print("timed out...")
            timeout_p.set_timeout()
            return None
            
        return TestResults(avg(time_lst), avg(out_lst), avg(reach_lst))


    aloha_timeout = TimeoutReference()
    aloha = {}
    aloha[2] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_2.xml"), aloha_timeout)
    aloha[5] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_5.xml"), aloha_timeout)
    aloha[10] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_10.xml"), aloha_timeout)
    aloha[25] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_25.xml"), aloha_timeout)
    aloha[50] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_50.xml"), aloha_timeout)
    aloha[100] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_100.xml"), aloha_timeout)
    aloha[250] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_250.xml"), aloha_timeout)
    aloha[500] = run_uppaal(path.join(args.uppaal_test_folder, "AlohaSingle_500.xml"), aloha_timeout)
    result_dct["aloha"] = aloha

    # agent covid
    agent_timeout = TimeoutReference()
    agent_covid = {}
    agent_covid[100] = run_uppaal(path.join(args.uppaal_test_folder, "AgentBasedCovid_100.xml"), agent_timeout)
    agent_covid[500] = run_uppaal(path.join(args.uppaal_test_folder, "AgentBasedCovid_500.xml"), agent_timeout)
    agent_covid[1000] = run_uppaal(path.join(args.uppaal_test_folder, "AgentBasedCovid_1000.xml"), agent_timeout)
    agent_covid[5000] = run_uppaal(path.join(args.uppaal_test_folder, "AgentBasedCovid_5000.xml"), agent_timeout)
    agent_covid[10000] = run_uppaal(path.join(args.uppaal_test_folder, "AgentBasedCovid_10000.xml"), agent_timeout)
    # agent_covid[50000] = run_uppaal("/UPPAALexperiments/AgentBasedCovid_50000.xml", agent_timeout)
    # agent_covid[100000] = run_uppaal("/UPPAALexperiments/AgentBasedCovid_100000.xml", agent_timeout)
    result_dct["agent_covid"] = agent_covid

    # reaction covid
    
    single_dct["covid"] = run_uppaal(path.join(args.uppaal_test_folder, "covidmodelQueryUPPAAL.xml"))
    single_dct["bluetooth"] = run_uppaal(path.join(args.uppaal_test_folder, "bluetoothNoParaSimas.cav.xml"))
    single_dct["firewire"] = run_uppaal(path.join(args.uppaal_test_folder, "firewireGoal.cav.xml"))

    # csma
    csma_timeout = TimeoutReference()
    csma = {}
    csma[2] = run_uppaal(path.join(args.uppaal_test_folder, "CSMA_2.xml"), csma_timeout)
    csma[5] = run_uppaal(path.join(args.uppaal_test_folder, "CSMA_5.xml"), csma_timeout)
    csma[10] = run_uppaal(path.join(args.uppaal_test_folder, "CSMA_10.xml"), csma_timeout)
    csma[25] = run_uppaal(path.join(args.uppaal_test_folder, "CSMA_25.xml"), csma_timeout)
    csma[50] = run_uppaal(path.join(args.uppaal_test_folder, "CSMA_50.xml"), csma_timeout)
    csma[100] = run_uppaal(path.join(args.uppaal_test_folder, "CSMA_100.xml"), csma_timeout)
    # csma[250] = run_uppaal("UPPAALexperiments/CSMA_250.xml")
    # csma[500] = run_uppaal("UPPAALexperiments/CSMA_500.xml")
    # csma[1000] = run_uppaal("UPPAALexperiments/CSMA_1000.xml")
    result_dct["csma"] = csma

    # fischer
    fischer_timeout = TimeoutReference()
    fischer = {}
    fischer[2] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_2.xml"), fischer_timeout)
    fischer[5] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_5.xml"), fischer_timeout)
    fischer[10] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_10.xml"), fischer_timeout)
    fischer[25] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_25.xml"), fischer_timeout)
    fischer[50] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_50.xml"), fischer_timeout)
    fischer[100] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_100.xml"), fischer_timeout)
    fischer[250] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_250.xml"), fischer_timeout)
    fischer[500] = run_uppaal(path.join(args.uppaal_test_folder, "fischer_500.xml"), fischer_timeout)
    # fischer[1000] = run_uppaal("UPPAALexperiments/fischer_1000.xml")
    result_dct["fischer"] = fischer

    return result_dct, single_dct


def test_smacc(binary, device, args):  # -> \
    # Tuple[
    #     Dict[Tuple[str, str], Dict[int, test_result | None]],
    #     Dict[Tuple[str, str], test_result | None]
    # ]:
    is_lite = bool(args.lite)
    device_args = load_choice(device)
    default_args = [binary, "-w", "lq", "-v", "0"]
    result_dct = {}
    single_dct = {}

    for settings, d_args in device_args:
        print(f"\nInitialising tests with {settings}:")

        # aloha
        aloha_timeout = TimeoutReference()
        aloha = {}
        aloha[2] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 2, timeout_p=aloha_timeout)
        aloha[5] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 5, timeout_p=aloha_timeout)
        if not is_lite:
            aloha[10] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 10, timeout_p=aloha_timeout)
            aloha[25] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 25, timeout_p=aloha_timeout)
            aloha[50] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 50, timeout_p=aloha_timeout)
            aloha[100] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 100, timeout_p=aloha_timeout)
            aloha[250] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 250, timeout_p=aloha_timeout)
            aloha[500] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 500, timeout_p=aloha_timeout)
        # aloha[1000] = run_model(default_args, settings, d_args, "100t", args, "AlohaSingle.xml", TOTAL_SIMS, 1000)
        result_dct[("aloha", settings)] = aloha

        # agant covid
        agent_covid = {}
        agent_timeout = TimeoutReference()
        agent_covid[100] = run_model(default_args, settings, d_args, "100t", args,
                                     "agentBaseCovid_100_1.0.xml", TOTAL_SIMS, 100, timeout_p=agent_timeout)
        if not is_lite:
            agent_covid[500] = run_model(default_args, settings, d_args, "100t", args,
                                         "agentBaseCovid_500_5.0.xml", TOTAL_SIMS, 500, timeout_p=agent_timeout)
            agent_covid[1000] = run_model(default_args, settings, d_args, "100t", args,
                                          "agentBaseCovid_1000_10.0.xml", TOTAL_SIMS, 1000, timeout_p=agent_timeout)
            agent_covid[5000] = run_model(default_args, settings, d_args, "100t", args,
                                          "agentBaseCovid_5000_50.0.xml", TOTAL_SIMS, 5000, timeout_p=agent_timeout)
            agent_covid[10000] = run_model(default_args, settings, d_args, "100t", args,
                                           "agentBaseCovid_10000_100.0.xml", TOTAL_SIMS, 10000, timeout_p=agent_timeout)
        # agent_covid[50000] = run_model(default_args, settings, d_args, "100t", args, "agentBaseCovid_50000_500.0.xml", TOTAL_SIMS, 50000)
        # agent_covid[100000] = run_model(default_args, settings, d_args, "100t", args, "agentBaseCovid_100000_1000.0.xml", TOTAL_SIMS, 1000000)
        result_dct[("agent_covid", settings)] = agent_covid

        # reaction_covid
        single_dct[("covid", settings)] = \
            run_model(default_args, settings, d_args, "100t", args, "covidmodelQueryI.xml", TOTAL_SIMS, 1,
                      query="Template4.Query")

        if settings == "SM":
            continue

        single_dct[("bluetooth", settings)] = \
            run_model(default_args, settings, d_args, "5000t", args, "bluetoothNoParaSimas.cav.xml", TOTAL_SIMS,
                      1,
                      query="Receiver.Reply")

        single_dct[("firewire", settings)] = \
            run_model(default_args, settings, d_args, "1000t", args, "firewireGoal.cav.xml", TOTAL_SIMS, 1,
                      query="Node0.s5")

        # csma
        csma_timeout = TimeoutReference()
        csma = {}
        csma[2] = run_model(default_args, settings, d_args, "2000t", args, "CSMA_2.xml", TOTAL_SIMS, 2,
                            use_scale=False, query="Process0.SUCCESS", timeout_p=csma_timeout)
        csma[5] = run_model(default_args, settings, d_args, "2000t", args, "CSMA_5.xml", TOTAL_SIMS, 5,
                            use_scale=False, query="Process0.SUCCESS", timeout_p=csma_timeout)
        if not is_lite:
            csma[10] = run_model(default_args, settings, d_args, "2000t", args, "CSMA_10.xml", TOTAL_SIMS, 10,
                                 use_scale=False, query="Process0.SUCCESS", timeout_p=csma_timeout)
            csma[25] = run_model(default_args, settings, d_args, "2000t", args, "CSMA_25.xml", TOTAL_SIMS, 25,
                                 use_scale=False, query="Process0.SUCCESS", timeout_p=csma_timeout)
            csma[50] = run_model(default_args, settings, d_args, "2000t", args, "CSMA_50.xml", TOTAL_SIMS, 50,
                                 use_scale=False, query="Process0.SUCCESS", timeout_p=csma_timeout)
            csma[100] = run_model(default_args, settings, d_args, "2000t", args, "CSMA_100.xml", TOTAL_SIMS, 100,
                                  use_scale=False, query="Process0.SUCCESS", timeout_p=csma_timeout)
        # csma[250] = run_model(default_args, settings, d_args, "2000t", args, "CSMA_250.xml", TOTAL_SIMS, 250,
        #                      use_scale=False, query="Process0.SUCCESS")
        # csma[500] = run_model(default_args, d_args, "2000t", args, "CSMA_500.xml", TOTAL_SIMS, 500, use_scale=False)
        # csma[1000] = run_model(default_args, d_args, "2000t", args, "CSMA_1000.xml", TOTAL_SIMS, 1000, use_scale=False)
        result_dct[("csma", settings)] = csma

        fischer_timeout = TimeoutReference()
        fischer = {}
        fischer[2] = run_model(default_args, settings, d_args, "300t", args, "fischer_2_29.xml", TOTAL_SIMS, 2,
                               use_scale=False, timeout_p=fischer_timeout)
        fischer[5] = run_model(default_args, settings, d_args, "300t", args, "fischer_5_29.xml", TOTAL_SIMS, 5,
                               use_scale=False, timeout_p=fischer_timeout)
        fischer[10] = run_model(default_args, settings, d_args, "300t", args, "fischer_10_29.xml", TOTAL_SIMS,
                                10, use_scale=False, timeout_p=fischer_timeout)
        if not is_lite:
            fischer[25] = run_model(default_args, settings, d_args, "300t", args, "fischer_25_29.xml", TOTAL_SIMS,
                                    25, use_scale=False, timeout_p=fischer_timeout)
            fischer[50] = run_model(default_args, settings, d_args, "300t", args, "fischer_50_29.xml", TOTAL_SIMS,
                                    50, use_scale=False, timeout_p=fischer_timeout)
            fischer[100] = run_model(default_args, settings, d_args, "300t", args, "fischer_100_29.xml", TOTAL_SIMS,
                                     100, use_scale=False, timeout_p=fischer_timeout)
            fischer[250] = run_model(default_args, settings, d_args, "300t", args, "fischer_250_29.xml", TOTAL_SIMS,
                                     250, use_scale=False, timeout_p=fischer_timeout)
            fischer[500] = run_model(default_args, settings, d_args, "300t", args, "fischer_500_29.xml", TOTAL_SIMS,
                                     500, use_scale=False, timeout_p=fischer_timeout)
        # fischer[1000] = run_model(default_args, settings, d_args, "300t", args, "fischer_500_29.xml", TOTAL_SIMS, 1000, use_scale=False)
        result_dct[("fischer", settings)] = fischer

    return result_dct, single_dct


def print_output(filepath, args):
    import matplotlib.markers as mplmarkers
    import matplotlib.pyplot as plt

    def load_time(row):
        try:
            return float(row["total_time"])
        except ValueError:
            return None

    def load_scale(row):
        try:
            return int(row["scale"])
        except ValueError:
            return None

    def get_comparison_table(data, scale_systems, single_systems, devices):
        if not (any(devices.intersection(CPU_CHOICES)) and any(devices.intersection(GPU_CHOICES))):
            return dict()

        def iterate_scales(choices, system):
            for row in (r for r in data if
                        r["system"] == system and r["scale"] != "single" and load_time(r) is not None):
                for device in choices:
                    if row["device"] == device:
                        yield int(row["scale"])

        def find_time(choices, system, scale):
            smallest = None
            for device in choices:
                for row in (r for r in data
                            if r["system"] == system and
                               r["device"] == device and
                               r["scale"] == str(scale)):
                    current = load_time(row)
                    if current is None: continue
                    smallest = min(smallest, current) if smallest is not None else current
            return smallest

        scale_dct = {}
        for system in scale_systems:
            scale_dct[system] = min(max(iterate_scales(CPU_CHOICES, system)), max(iterate_scales(GPU_CHOICES, system)))

        cpu_dct, gpu_dct = dict(), dict()
        for system in scale_systems:
            cpu_dct[system] = find_time(CPU_CHOICES, system, scale_dct[system])
            gpu_dct[system] = find_time(GPU_CHOICES, system, scale_dct[system])
        for system in single_systems:
            cpu_dct[system] = find_time(CPU_CHOICES, system, "single")
            gpu_dct[system] = find_time(GPU_CHOICES, system, "single")
        return cpu_dct, gpu_dct

    def print_table(columns, data, name):
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        ax.table(colLabels=columns, cellText=data)
        fig.tight_layout()

        if args.plot_dest is not None:
            plt.savefig(path.join(args.plot_dest, name), dpi=1024)
        if args.show:
            plt.show()
        plt.clf()

    def print_cactus_plot():
        nonlocal args, device_dct
        reference_d = {(system, scale): time for (system, scale, time) in device_dct[REF_DEVICE]}
        #problems which can be solved in less time than CACTUS_PLOT_CUTOFF in any config
        small_problems = set(((system, scale)
                              for lst in device_dct.values()
                              for (system, scale, time) in lst if time < CACUTS_PLOT_CUTOFF))
        ref_speed = dict()
        for system, scale, time in device_dct[REF_DEVICE]:
            ref_speed[(system, scale)] = min(ref_speed.get((system, scale), time), time)
        speedup_dct = {device: sorted([(system, scale, reference_d[(system, scale)] / time)
                                       for (system, scale, time) in lst
                                       if (system, scale) not in small_problems and (system, scale) in reference_d],
                                      key=lambda x: x[2])
                       for device, lst in device_dct.items()}

        plt.title("speedup cactus plt")
        plt.xlabel("problem instance")
        plt.ylabel("speedup over CPU")

        markers = list(mplmarkers.MarkerStyle.markers.values())
        # maxtime, maxratio = 0, 0
        for i, (device, lst) in enumerate(speedup_dct.items()):
            if len(lst) == 0: continue
            xs, ys = list(range(len(lst))), [speedup for (_, _, speedup) in lst]
            plt.plot(xs, ys, linewidth=2.0, label=device)
        plt.legend()
        if args.plot_dest is not None:
            plt.savefig(path.join(args.plot_dest, "cactus_speedup_plot.png"))
        if args.show:
            plt.show()
        plt.clf()

        power_dct = {device: sorted([(system, scale, power_ratio(device, time, ref_speed[(system, scale)]))
                                     for (system, scale, time) in lst
                                     if (system, scale) not in small_problems and (system, scale) in ref_speed],
                                    key=lambda x: x[2])
                     for device, lst in device_dct.items() if device != BASELINE}

        plt.title("power cactus plt")
        plt.xlabel("problem instance")
        plt.ylabel("power usage over CPU")
        for i, (device, lst) in enumerate(power_dct.items()):
            if len(lst) == 0: continue
            xs, ys = list(range(len(lst))), [power for (_, _, power) in lst]
            plt.plot(xs, ys, linewidth=2.0, label=device)
        plt.legend()
        if args.plot_dest is not None:
            plt.savefig(path.join(args.plot_dest, "cactus_power_plot.png"))
        if args.show:
            plt.show()
        plt.clf()


    systems, single_systems = set(), set()
    results = dict()
    single_results = dict()
    device_dct = dict()  # Dict[str, List[Tuple[str, int or None, float]]]
    cpu_dct, gpu_dct = None, None
    with open(filepath, 'r') as f:
        csv_reader = list(csv.DictReader(f, delimiter='\t'))
        for row in csv_reader:
            s = single_systems if row["scale"] == "single" else systems
            s.add(row["system"])
            device_dct[row["device"]] = \
                (device_dct.get(row["device"], [])
                 + [(row["system"], load_scale(row), load_time(row))])
            time = load_time(row)
            if row["scale"] == "single":
                single_results[row["system"]] = single_results.get(row["system"], []) + [(row["device"], time)]
            else:
                system, settings, scale = row["system"], row["device"], row["scale"]
                dct = results[system] = results.get(system, {})
                xs, ys = dct.get(settings, ([], []))
                results[system][settings] = (xs + [scale], ys + [time])
        cpu_dct, gpu_dct = get_comparison_table(csv_reader, systems, single_systems, set(device_dct.keys()))
    systems = systems.union(single_systems)
    for system, rs in results.items():
        # fig, ax = plt.subplots()
        plt.title(system)
        plt.xlabel("#components")
        plt.ylabel("time [sec]")
        for settings, (xs, ys) in rs.items():
            plt.plot(xs, ys, label=settings)
        plt.legend()
        if args.plot_dest is not None:
            plt.savefig(path.join(args.plot_dest, system + ".png"))
        if args.show:
            plt.show()
        plt.clf()

    columns = ["system", "BASELINE", "UPPAAL", "CPU", "PN-CPU", "GPU", "JIT", "PN", "SM"]
    index_map = {"BASELINE": 1, "UPPAAL": 2, "CPU": 3, "PN-CPU": 4, "GPU": 5, "JIT": 6, "PN": 7, "SM": 8}
    data = list()
    for system, rs in single_results.items():
        current = [system] + ['-' for _ in range(len(index_map))]
        for device, time in rs:
            if device not in index_map: continue
            current[index_map[device]] = "{:.{}f}".format(time, 2)
        data.append(current)
    print_table(columns, data, "nonscalable_table.png")

    if not (any(cpu_dct) and any(gpu_dct)):
        return

    systems = list(systems)
    columns = ["system", "CPU", "GPU", "speedup"]
    data = list()

    for system in systems:
        cpu, gpu = cpu_dct.get(system), gpu_dct.get(system)
        ratio = (cpu / gpu)
        # equiv = ratio * mp.cpu_count()
        data.append([system, cpu, gpu, ratio * 100])
    avg_ratio = sum([x[1] for x in data]) / sum([x[2] for x in data])
    data.append(["average",
                 sum([x[1] for x in data]),
                 sum([x[2] for x in data]),
                 avg_ratio])
    print_table(columns, data, "comparison_table.png")
    if REF_DEVICE in device_dct:
        print_cactus_plot()


DID_NOT_FINISH = 'DNF'


def write_output(
        results,
        single_results, output_path):
    def time_convert(t):
        return t if t is not None else DID_NOT_FINISH

    # file
    with open(path.join(output_path, FULL_OUTPUT_FILENAME), 'w') as f:
        f.write("system\tdevice\tscale\trun_time\ttotal_time\thit\n")
        for (system, device), (scale, result) in ((x, y) for x, rs in results.items() for y in rs.items()):
            r_time = result.run_time if result is not None else None
            t_time = result.total_time if result is not None else None
            hit = result.hit if result is not None else None
            f.write(f"{system}\t{device}\t{scale}\t{time_convert(r_time)}\t{time_convert(t_time)}\t{hit}\n")
        for (system, device), result in single_results.items():
            r_time = result.run_time if result is not None else None
            t_time = result.total_time if result is not None else None
            hit = result.hit if result is not None else None
            f.write(f"{system}\t{device}\tsingle\t{time_convert(r_time)}\t{time_convert(t_time)}\t{hit}\n")

    for (system, device), rs in results.items():
        with open(path.join(output_path, system + "_" + device + ".tsv"), 'x') as f:
            for scale, o in rs.items():
                if o is None: continue
                f.write(f"{scale}\t{o.total_time}\n")

    with open(path.join(output_path, "singles.tsv"), 'x') as f:
        for (system, device), o in single_results.items():
            f.write(f"{system}\t{device}\t{time_convert(o.total_time if o is not None else None)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--program", type=str, required=False, dest="program", help="Path to program binary")
    parser.add_argument("-d", "--device", type=str,
                        choices=list(DEVICE_CHOICES.keys()) + list(ADDITIONAL_CHOICES),
                        default=ALL, required=False, help="Method to run test using")
    parser.add_argument("-m", "--model", type=str, required=False, default=None,
                        dest="model", help="path to models to test")
    parser.add_argument("-c", "--cache", type=str, required=False, default=None, dest="temp",
                        help="Folder to store cache files (defualt = create temporary folder)")
    parser.add_argument("-o", "--output", type=str, required=False, default=None,
                        dest="output", help="path to folder store output")
    parser.add_argument("-t", "--threads", type=str, required=False, default=THREAD_JOBS, dest="threads",
                        help="# of jobs to utilise in CPU computations")
    parser.add_argument("--timeout", type=int, required=False, default=3600, dest="timeout",
                        help="seconds to run simulation, before timing out. Default = 3600")
    parser.add_argument("--UPPAAL", type=str, required=False, default=None, dest="uppaal",
                        help="path to uppaal executable to run (e.g. /verifyta), to run tests on uppaal.")
    parser.add_argument("--up", type=str, required=False, default=DEFAULT_UPPAAL_EXPERIMENT_FOLDERNAME,
                        dest='uppaal_test_folder',
                        help='path extension on m, which to use for uppaal experiments (not required).')
    parser.add_argument("-b", "--blocks", type=str, required=False, default=CUDA_PARALLEL,
                        help="blocks and threads configuration to use on the GPU. should be [blocks],[threads], "
                             "e.g. 40,256. default=40,256")
    parser.add_argument("-n", "--n_samples", type=int, required=False, dest='n_samples', default=None, #adjusted later in function depending on mode
                        help="Number of samples per test. (default=10)")
    # parser.add_argument("-u", "--uppaal", type=str, default=None, required=False,
    #                     dest="uppaal", help="Path to uppaal binary. If supplied, runs uppaal tests on uppaal too.")
    parser.add_argument("-g", "--graph", type=str, required=False, default=None,
                        dest="graph", help=
                        "Path to existing data to generate plots. If this option is supplied, no tests will be run.")
    parser.add_argument("-s", "--show", required=False, action='store_true', dest="show",
                        default=False, help="Whether to show plots or not. Default is not show")
    parser.add_argument("--saveplots", required=False, dest="plot_dest",
                        default=None, help="Path to save plots as pngs. If not supplied, plots wont be seved as file")
    parser.add_argument("--lite", required=False, default=0, dest="lite", type=int,
                        help="Whether to run lite or non-lite version. (0 = full suite; 1 = lite suite; default = 0)")
    parser.add_argument("--args", required=False, nargs="+", dest="additional_args", default=[],
                        help="Additional arguments for running simulation. Added before all other args.")

    if len(sys.argv) < 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    
    if args.n_samples is None:
        args.n_samples = N_SAMPLES if args.lite == 0 else 1

    if args.graph is not None:
        if not path.exists(args.graph):
            raise argparse.ArgumentError(None, "Path to existing data source doesnt exist.")
        print_output(args.graph, args)
        return

    if args.device not in DEVICE_CHOICES and args.device not in ADDITIONAL_CHOICES:
        raise argparse.ArgumentError(None, "device type is not recognised device type. Should be in "
                                     + str(DEVICE_CHOICES) + " or a batch run, using one of the following: "
                                     + str(ADDITIONAL_CHOICES))

    if args.model is None or not path.exists(args.model):
        raise argparse.ArgumentError(None, "Cannot find models to test")

    if args.temp is None:
        args.temp_dir = temp.TemporaryDirectory(dir=os.getcwd(), ignore_cleanup_errors=True)
        args.temp = args.temp_dir.name
    else:
        args.temp_dir = None

    if args.output is None or not path.isdir(args.output):
        raise argparse.ArgumentError(None, "No output folder supplied (or path is not folder).")
    elif len(os.listdir(args.output)) > 0:
        raise argparse.ArgumentError(None, "Output folder is not empty. Please make sure folder is empty.")

    if (args.program is None or not path.exists(args.program)) and args.uppaal is None:
        raise argparse.ArgumentError(None, "Cannot find program binary")

    global TOTAL_SIMS
    blocks, threads = str.split(args.blocks, ',')
    TOTAL_SIMS = int(blocks) * int(threads)
    results, table_res = dict(), dict()

    if args.program is not None:
        results, table_res = test_smacc(args.program, args.device, args)

    if args.uppaal is not None:
        uppaal_res, uppaal_table_res = test_uppaal(args.uppaal, args)
        for system, rs in uppaal_res.items():
            results[(system, "uppaal")] = rs
        for system, result in uppaal_table_res.items():
            table_res[(system, "uppaal")] = result

    write_output(results, table_res, args.output)
    if args.show or args.plot_dest is not None:
        print_output(path.join(args.output, FULL_OUTPUT_FILENAME), args)

    if args.temp_dir:
        args.temp_dir.cleanup()
    # delete folder


if __name__ == "__main__":
    print(sys.argv)
    main()
