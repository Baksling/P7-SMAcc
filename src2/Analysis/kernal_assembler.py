import sys
import os.path as path

#This is what the JIT compiler looks for.
OUTPUT_FILENAME = "kernal.cu"

def header_file_lst(source_path: str) -> list[str]: 
    return [path.join(source_path, 'common/macro.h',)]

def file_lst(source_path: str) -> list[str]:
    return [path.join(source_path, x) for x in [
        'common/sim_config.h',
        'common/my_stack.h',
        'engine/Domain.h',
        'engine/Domain.cu',
        'engine/model_oracle.h',
        'engine/model_oracle.cu',
        'results/result_store.h',
        'engine/automata_engine.cu'
    ]]

def combine_kernal(output_path: str):
    with open(path.join(output_path, OUTPUT_FILENAME) , 'w', encoding="utf8") as fout:
        fout.write('#include <cmath>\n')
        fout.write('#include <string>\n')
        fout.write('#include <curand.h>\n')
        fout.write('#include <curand_kernel.h>\n')
        for file in header_file_lst(output_path):
            with open(file, 'r') as f:
                for line in f.readlines():
                    tmp = line.replace('\ufeff', '').replace('ï»¿', '')
                    if "#define QUALIFIERS" in tmp or "#undef QUALIFIERS" in tmp: continue
                    if "#include" not in tmp:
                        fout.write(tmp)

        for file in file_lst(output_path):
            with open(file, 'r') as f:
                for line in  f.readlines():
                    tmp = line.replace('\ufeff', '').replace('ï»¿', '') #remove weird character
                    if "#include" not in tmp:
                        fout.write(tmp)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please provide only 1 argument, namely the location of the source")
        sys.exit(1)
    combine_kernal(sys.argv[1])
