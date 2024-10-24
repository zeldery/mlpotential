'''
DATA GENERATION
Generate the data and run computational for that
Support format:
 + orca
 + ...
'''

import argparse
import os
import numpy as np

ATOMIC_NUMBERS_TO_SPECIES = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
SPECIES_TO_ATOMIC_NUMBERS = {value: key for key, value in ATOMIC_NUMBERS_TO_SPECIES.items()}

def get_argument():
    parser = argparse.ArgumentParser('datagen_compute',
                                     description='')
    parser.add_argument('-s', '--scratch', default='scratch')
    parser.add_argument('-i', '--index')
    parser.add_argument('-c', '--control', default='control')
    parser.add_argument('-t', '--tarsize', default='1000')
    args = parser.parse_args()
    return args

def run_orca(name, atomic_numbers, positions):
    f = open(f'{name}.inp', 'w')
    f.write('!B3LYP def2-TZVP ENGRAD PAL8\n\n* xyz 0 1\n')
    n = atomic_numbers.shape[0]
    for i in range(n):
        f.write(f'{ATOMIC_NUMBERS_TO_SPECIES[atomic_numbers[i]]} {positions[i,0]} {positions[i,1]} {positions[i,2]} \n')
    f.write('*\n')
    f.close()
    os.system(f'$EBROOTORCA/orca {name}.inp >{name}.out')

def main():
    args = get_argument()
    index = int(args.index)
    tarsize = int(args.tarsize)
    path_control = os.path.join(args.control, f'{index}.index')
    n_complete = 0
    if os.path.exists(path_control):
        f = open(path_control, 'r')
        line = f.readline()
        n_complete = int(line.split()[0])
        f.close()
    else:
        n_complete = 0
    f_data = open(os.path.join(args.control, f'{index}.txt'), 'r')
    i = 0
    line = f_data.readline()
    while line != '' and i < n_complete:
        line = f_data.readline()
        i += 1
    if line == '':
        f_data.close()
        print('Complete')
        return
    while line != '':
        atomic_, pos_ = line[:-1].split(',')
        atomic_numbers = atomic_.split(' ')
        positions = pos_.split(' ')
        atomic_numbers = [int(x) for x in atomic_numbers]
        atomic_numbers = np.array(atomic_numbers)
        positions = [float(x) for x in positions]
        positions = np.array(positions).reshape(-1, 3)
        os.chdir(args.scratch)
        run_orca(f'{index}_{n_complete}', atomic_numbers, positions)
        os.chdir('..')
        n_complete += 1
        f = open(path_control, 'w')
        f.write(f'{n_complete}\n')
        f.close()
        if n_complete % tarsize == 0:
            os.system(f'tar -cvf {args.control}/{index}_{n_complete//tarsize}.tar {args.scratch}/{index}_*')
            os.system(f'rm {args.scratch}/{index}_*')
        line = f_data.readline()
    f_data.close()
    
if __name__ == '__main__':
    main()