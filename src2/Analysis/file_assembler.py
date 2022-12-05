import argparse

def file_lst() -> list[str]:
    return [
        './common/macro.h',
        './common/sim_config.h',
        './common/my_stack.h',
        './engine/Domain.h',
        './engine/Domain.cu',
        './engine/model_oracle.h',
        './engine/model_oracle.cu',
        './results/result_store.h',
        './engine/automata_engine.cu'
    ]


def do_da_one_file_inator_and_yeet_da_includes(file_lst: list[str]):
    with open('./one_file_inator_output_without_includes.cu', 'w') as fout:
        for file in file_lst:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    tmp = line.replace('ï»¿', '')
                    if not tmp.startswith('#include') or file == file_lst[0]:
                        fout.write(tmp)


if __name__ == '__main__':
    do_da_one_file_inator_and_yeet_da_includes(file_lst())
