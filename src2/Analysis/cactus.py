import sys
import os.path as path
import os
from typing import Dict, List, Tuple
import csv

BASE_CONFIG = "CPU"
CPU_CONFIG = {"PN-CPU", "BASELINE", "uppaal"}
GPU_CONFIG = {"GPU", "JIT", "PN", "SM"}
UPPAAL = "uppaal"
MODEL = 0
DEVICE = 1
SCALE = 2
TOTAL_TIME = 4

def main():
    print(sys.argv)
    file = path.join(os.getcwd(), sys.argv[1])
    gpu_power, cpu_power = float(sys.argv[2]), float(sys.argv[3])
    
    models : Dict[Tuple[str, int], List[Tuple[str, float]]] = dict()
    with open(file, 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            model, scale, device, time = row[MODEL], row[SCALE], row [DEVICE], row[TOTAL_TIME]
            if time == "DNF" or scale is None or time is None: continue
            time = float(time)
            scale = 1 if (scale == "single") else int(scale)
            models[(model, scale)] = models.get((model, scale), []) + [(device, time)]

    remove_set = set()

    for system in models.keys():
        for (device, time) in models[system]:
            if (time < 2.0):
                remove_set.add(system)
                break

    for key in remove_set:
        models.pop(key)

    cactus : Dict[Tuple[str, int, str], float] = dict()

    for (model, scale), values in models.items():
        for (device, time) in values:
            cactus[(model, scale, device)] = time
    
    speed_dct : Dict[str, List[float]] = dict()
    power_dct : Dict[str, List[float]] = dict()
    
    for (model, scale, device), time in cactus.items():
        if (device == BASE_CONFIG): continue
        base_speed = cactus[model, scale, BASE_CONFIG]
        use_power = gpu_power if (device in GPU_CONFIG) else cpu_power
        speed = base_speed / time
        power = (time * use_power) / (base_speed * cpu_power)
        speed_dct[device] = speed_dct.get(device, []) + [speed]
        power_dct[device] = power_dct.get(device, []) + [power]

    speed_sorted : Dict[str, List[float]] = dict()
    power_sorted : Dict[str, List[float]] = dict()

    for device, speed in speed_dct.items():
        newSpeed = sorted(speed, key = float)
        speed_sorted[device] = newSpeed

    for device, power in power_dct.items():
        newPower = sorted(power, key = float)
        power_sorted[device] = newPower

    configs = {device for device in speed_dct.keys()}

    for device in configs:
        with open(path.join(os.getcwd(), "cactus_power_" + device + ".tsv"), 'x') as p:
            with open(path.join(os.getcwd(), "cactus_speedup_" + device + ".tsv"), 'x') as s:
                i = 1
                for speed in speed_sorted[device]:
                    s.write(f"{i}\t{speed}\n")
                    i += 1
                i = 1
                for power in power_sorted[device]:
                    p.write(f"{i}\t{power}\n")
                    i += 1

    
if __name__ == "__main__":
    main()