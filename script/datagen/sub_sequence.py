'''
Pending
'''

import argparse
import subprocess

def get_argument():
    parser = argparse.ArgumentParser(prog='sub_orca_sequence')
    parser.add_argument('nprocess')
    parser.add_argument('replicate')
    args = parser.parse_args()
    return args

def submit(index, previous=None):
    command = f'sbatch --nodes=1 --ntasks-per-node=8 --mem=6G --time=1-00:00:00 '
    command += f'--account=def-crowley-ab --job-name=compute_{index} --output={index}.log'
    if previous is not None:
        command += f' --dependency=afterany:{previous}'
    command += f" --wrap=\' module load StdEnv/2020 gcc/10.3.0 openmpi/4.1.1 orca/5.0.4; "
    command += f" source ~/working/bin/activate ; "
    command += f" python datagen_compute.py -s scratch -i {index} -c control -t 1000 \'"
    output = subprocess.getoutput(command)
    return output

def main():
    args = get_argument()
    nprocess = int(args.nprocess)
    n_replicate = int(args.replicate)
    for index in range(nprocess):
        output = submit(index)
        previous = output.split()[-1]
        print(f'Submit: {previous}')
        for i in range(1, n_replicate):
            output = submit(index, previous)
            previous = output.split()[-1]
            print(f'Submit: {previous}')

if __name__ == '__main__':
    main()
