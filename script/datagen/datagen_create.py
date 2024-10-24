'''
DATA GENERATION
Generate the data structure for running
Support format:
 + orca
 + ...
'''

import argparse
import os
import pandas as pd
from mlpotential.dataloader import H5PyScanner

def get_argument():
    parser = argparse.ArgumentParser('datagen_create',
                                     description='Create the structure to run')
    parser.add_argument('-d', '--datalist', nargs='*')
    parser.add_argument('-n', '--nprocess')
    parser.add_argument('-c', '--control', default='control')
    parser.add_argument('-s', '--scratch', default='scratch')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    scanner = H5PyScanner(['atomic_numbers', 'coordinates'], 'atomic_numbers')
    n_max = 0
    for dat in scanner.scan_individual(args.datalist):
        n = dat['atomic_numbers'].sum()
        if n > n_max:
            n_max = n
        path = os.path.join(args.scratch, f'{n}.txt')
        if os.path.exists(path):
            f = open(path, 'a')
            atomic = ' '.join([str(x) for x in dat['atomic_numbers']])
            pos = ' '.join([str(x) for x in dat['coordinates'].reshape(-1)])
            f.write(f'{atomic},{pos}\n')
            f.close()
        else:
            f = open(path, 'w')
            atomic = ' '.join([str(x) for x in dat['atomic_numbers']])
            pos = ' '.join([str(x) for x in dat['coordinates'].reshape(-1)])
            f.write(f'{atomic},{pos}\n')
            f.close()
    # Split into 
    n_process = int(args.nprocess)
    for i in range(n_process):
        path = os.path.join(args.control, f'{i}.txt')
        f = open(path, 'w')
        f.close()
    current = 0
    n_current = 1
    while n_current <= n_max:
        while not os.path.exists(os.path.join(args.scratch, f'{n_current}.txt')) and n_current <= n_max:
            n_current += 1
        if n_current > n_max:
            continue
        origin_path = os.path.join(args.scratch, f'{n_current}.txt')
        f = open(origin_path, 'r')
        line = f.readline()
        while line != '':
            g = open(os.path.join(args.control, f'{current}.txt'), 'a')
            g.write(line)
            g.close()
            current += 1
            if current >= n_process:
                current = 0
            line = f.readline()
        f.close()
        n_current += 1

if __name__ == '__main__':
    main()