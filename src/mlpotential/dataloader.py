'''
Handle the hdf5 data type, create ready-to-use data for pytorch
'''

import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.linear_model import LinearRegression

class H5PyScanner:
    '''
    Scanner for the list of h5py files based on the field name
    The H5 file can have as many layers as possible, as long as it has the named fields in the list
    The exception is used for atomic_numbers or species in the case of 1 common value for all data in that batch
    '''
    def __init__(self, field_list, exception=None):
        self.field_list = field_list.copy()
        self.exception = exception
    
    def scan_(self, sub):
        '''
        Internal use for scan the h5py directory
        '''
        lst_key = list(sub.keys())
        if len(lst_key) > 0:
            if isinstance(sub[lst_key[0]], h5py.Group):
                for key in lst_key:
                    yield from self.scan_(sub[key])
            else:
                for field in self.field_list:
                    if field not in lst_key:
                        raise KeyError(f'{field} not in dataset')
                tmp = {}
                for field in self.field_list:
                    tmp[field] = np.array(sub[field])
                yield tmp

    def scan(self, list_file_name):
        '''
        Real scan for list of file name here
        '''
        for name in list_file_name:
            file = h5py.File(name)
            yield from self.scan_(file)
            file.close()

    def scan_individual(self, list_file_name):
        '''
        Similar to scan, but instead of all the field, slice through individual 
        data point in the batch
        The batch dimension is set to be the first dimension by default
        '''
        for dat in self.scan(list_file_name):
            n_dims = {}
            n_batch = -1
            for field in self.field_list:
                if field != self.exception:
                    n_batch = dat[field].shape[0]
                    break
            for field in self.field_list:
                n_dims[field] = dat[field].ndim
            for i in range(n_batch):
                tmp = {}
                for field in self.field_list:
                    if field == self.exception:
                        tmp[field] = dat[field]
                    else:
                        if n_dims[field] == 1:
                            tmp[field] = dat[field][i]
                        else:
                            tmp[field] = dat[field][i,...]
                yield tmp

    def stack_(self, data_list, pad_value):
        '''
        Stack the list of data into single array (the batch dimension is the first)
        The value is pad in the first dimension with the pad value
        '''
        if isinstance(data_list[0], np.ndarray):
            n_dim = data_list[0].ndim
            n_mask = -1
            for dat in data_list:
                if dat.shape[0] > n_mask:
                    n_mask = dat.shape[0]
            new_list = []
            for dat in data_list:
                pad_width = [[0, 0] for i in range(n_dim)]
                pad_width[0][1] = n_mask - dat.shape[0]
                new_list.append(np.pad(dat, pad_width, constant_values=pad_value))
            return np.stack(new_list, axis=0)
        else:
            return np.array(data_list)

    def scan_stack(self, list_file_name, n_batch):
        '''
        Main scanner with n_batch data stacked
        '''
        storage = {}
        for field in self.field_list:
            storage[field] = []
        n_count = 0
        for dat in self.scan_individual(list_file_name):
            n_count += 1
            for field in self.field_list:
                storage[field].append(dat[field])
            if n_count == n_batch:
                result = {}
                for field in self.field_list:
                    pad_value = 0.0
                    if isinstance(storage[field], np.ndarray):
                        if storage[field].dtype == np.int32 or storage[field].dtype == np.int64:
                            pad_value = -1
                    result[field] = self.stack_(storage[field], pad_value)
                    storage[field] = []
                yield result
                n_count = 0
        if n_count > 0:
            result = {}
            for field in self.field_list:
                pad_value = 0.0
                if isinstance(storage[field], np.ndarray):
                    if storage[field].dtype == np.int32 or storage[field].dtype == np.int64:
                        pad_value = -1
                result[field] = self.stack_(storage[field], pad_value)
                storage[field] = []
            yield result
                
    def generate_iterator(self, list_file_name, n_batch, output_name, testratio):
        file = h5py.File(output_name, 'w')
        file.create_group('train')
        file.create_group('test')
        n_train = 0
        n_test = 0
        for dat in self.scan_stack(list_file_name, n_batch):
            if random.random() < testratio:
                file.create_group(f'test/{n_test}')
                for field in self.field_list:
                    file.create_dataset(f'test/{n_test}/{field}', data=dat[field])
                n_test += 1
            else:
                file.create_group(f'train/{n_train}')
                for field in self.field_list:
                    file.create_dataset(f'train/{n_train}/{field}', data=dat[field])
                n_train += 1
        file.close()

class DataIterator(Dataset):
    '''
    The data iterator for pre-defined hdf5 structure that split in batch and split train/test
    Always load to CPU following the guideline
    '''
    def __init__(self, destination, property_list):
        self.file = h5py.File(destination, 'r')
        self.property_list = property_list.copy()
        self.n_train = len(list(self.file['train'].keys()))
        self.n_test = len(list(self.file['test'].keys()))
        self.mode = 'all'

    def __del__(self):
        self.file.close()

    def __len__(self):
        if self.mode == 'all':
            return self.n_train + self.n_test
        if self.mode == 'train':
            return self.n_train
        if self.mode == 'test':
            return self.n_test

    def __getitem__(self, index):
        result = {}
        if self.mode == 'train':
            sub = self.file[f'train/{index}']
        elif self.mode == 'test':
            sub = self.file[f'test/{index}']
        else:
            if index < self.n_train:
                sub = self.file[f'train/{index}']
            else:
                sub = self.file[f'test/{index - self.n_train}']
        for property in self.property_list:
            result[property] = torch.tensor(np.array(sub[property]))
        return result  

    def dataloader(self, **kwargs):
        return DataLoader(self, batch_size=None, batch_sampler=None, pin_memory=True, **kwargs)


### Old code, delete in the future

class H5PyAnalyzedScanner:
    '''
    Scan the structure and properties in hdf5/h5 file
    Create the file structure in a folder
    Generate the summary data for all the properties
    '''
    def __init__(self, file_list, element_list, summary_properties, atomic_properties, first_level_empty=False):
        self.file_list = file_list.copy()
        self.element_list = element_list.copy()
        self.summary_properties = summary_properties.copy()
        self.atomic_properties = atomic_properties.copy()
        self.first_level_empty = first_level_empty
        self.scanned = False
        self.report = ''

    def iter(self):
        current = {}
        for name in self.file_list:
            file = h5py.File(name)
            if self.first_level_empty:
                main = file[list(file.keys())[0]] # Assume the first layer of h5py is the file_name
            else:
                main = file
            for key in list(main.keys()):
                sub = main[key]
                # Test is it ok data, and the element is correct
                try:
                    atomic_numbers = np.array(sub['atomic_numbers'], dtype=np.int64)
                except:
                    continue
                # Check if the structure is in the element allowed
                for element in atomic_numbers:
                    if element not in self.element_list:
                        break
                else:
                    coordinates = np.array(sub['coordinates'], dtype=np.float32)
                    forces = np.array(sub['forces'], dtype=np.float32)
                    for property in self.atomic_properties:
                        current[property] = np.array(sub[property], dtype=np.float32)
                    for property in self.summary_properties:
                        current[property] = np.array(sub[property], dtype=np.float64)
                    n = coordinates.shape[0]
                    for i in range(n):
                        result = {}
                        result['atomic_numbers'] = atomic_numbers
                        result['coordinates'] = coordinates[i,:,:]
                        result['forces'] = forces[i,:,:]
                        for property in self.atomic_properties:
                            result[property] = current[property][i,:]
                        for property in self.summary_properties:
                            result[property] = current[property][i]
                        yield result
            file.close()

    def scan(self):
        '''
        Scan the structure in file_list
        '''
        # Initialize
        self.n_atoms = 0
        self.n_structures = 0
        self.max_number_atoms = 0
        self.summary_element = {} # Number of elements in each structure, for summary regression
        for element in self.element_list:
            self.summary_element[element] = []
        self.summary_count = {} # Summary properties for each structure
        for property in self.summary_properties:
            self.summary_count[property] = []
        self.atomic_track = {} # Summary of all atomic properties, split by element
        for element in self.element_list:
            self.atomic_track[element] = {}
            for property in self.atomic_properties:
                self.atomic_track[element][property] = []
        # Scan the structure
        for data in self.iter():
            self.n_structures += 1
            n_atoms = len(data['atomic_numbers'])
            self.n_atoms += n_atoms
            if n_atoms > self.max_number_atoms:
                self.max_number_atoms = n_atoms
            for element in self.element_list:
                self.summary_element[element].append((data['atomic_numbers'] == element).sum())
            for property in self.summary_properties:
                self.summary_count[property].append(data[property])
            for property in self.atomic_properties:
                for i in range(n_atoms):
                    self.atomic_track[data['atomic_numbers'][i]][property].append(data[property][i])
        self.scanned = True

    def summary(self):
        assert self.scanned
        report = 'DATASET REPORT\n'
        report += f'Number of structure: {self.n_structures}\n'
        report += f'Number of atoms in structure: {self.n_atoms}\n'
        report += f'Max number of atoms: {self.max_number_atoms}\n'
        report += 'Element list: '
        for element in self.element_list:
            report += f'{element} '
        report += '\nSUMMARY REPORT\n'
        tmp = []
        for element in self.element_list:
            tmp.append(np.array(self.summary_element[element]))
        summary_element = np.stack(tmp, 1)
        for property in self.summary_properties:
            report += f'{property}:\n'
            summary_count = np.array(self.summary_count[property])
            model_mean = LinearRegression(fit_intercept=False)
            model_mean.fit(summary_element, summary_count)
            predict_summary = np.matmul(summary_element, model_mean.coef_.reshape(-1, 1)).reshape(-1)
            var_result = (predict_summary - summary_count) ** 2
            model_var = LinearRegression(fit_intercept=False)
            model_var.fit(summary_element**2, var_result)
            for i, element in enumerate(self.element_list):
                report += f'  Element {element}: {model_mean.coef_[i]} | {model_var.coef_[i]**0.5}\n'
            report += f'  R^2 : {model_mean.score(summary_element, summary_count)} | {model_var.score(summary_element**2, var_result)}\n'
        report += 'ATOMIC REPORT\n'
        for property in self.atomic_properties:
            report += f'{property}:\n'
            for element in self.element_list:
                values = np.array(self.atomic_track[element][property])
                report += f'  Element {element}: {values.mean()} | {values.std()}\n'
        self.report = report
        return report

    def generate(self, n_batch, test_ratio, destination):
        dest = h5py.File(destination, 'w')
        dest.create_group('train')
        dest.create_group('test')
        train_current = 0
        test_current = 0
        train_count = 0
        test_count = 0
        train_max = 0
        test_max = 0
        train_batch = {}
        test_batch = {}
        for property in ['atomic_numbers', 'coordinates', 'forces'] + self.atomic_properties + self.summary_properties:
            train_batch[property] = []
            test_batch[property] = []
        for data in self.iter():
            if random.random() < test_ratio:
                # Add data to test
                for property in ['atomic_numbers', 'coordinates', 'forces'] + self.atomic_properties + self.summary_properties:
                    test_batch[property].append(data[property])
                if len(data['atomic_numbers']) > test_max:
                    test_max = len(data['atomic_numbers'])
                test_count += 1
                if test_count >= n_batch:
                    # Reshape the data
                    for i in range(test_count):
                        n_pad = test_max - len(test_batch['atomic_numbers'][i])
                        test_batch['atomic_numbers'][i] = np.pad(test_batch['atomic_numbers'][i], (0, n_pad), constant_values=-1)
                        test_batch['coordinates'][i] = np.pad(test_batch['coordinates'][i], ((0, n_pad), (0,0)))
                        test_batch['forces'][i] = np.pad(test_batch['forces'][i], ((0, n_pad), (0,0)))
                        for property in self.atomic_properties:
                            test_batch[property][i] = np.pad(test_batch[property][i], (0, n_pad))
                    dest.create_group(f'test/{test_current}')
                    for property in ['atomic_numbers', 'coordinates', 'forces'] + self.atomic_properties + self.summary_properties:
                        dest.create_dataset(f'test/{test_current}/{property}',\
                                            data=np.stack(test_batch[property], axis=0))
                        # Reset
                        test_batch[property] = []
                    test_current += 1
                    test_count = 0
                    test_max = 0
            else:
                # Add data to test
                for property in ['atomic_numbers', 'coordinates', 'forces'] + self.atomic_properties + self.summary_properties:
                    train_batch[property].append(data[property])
                if len(data['atomic_numbers']) > train_max:
                    train_max = len(data['atomic_numbers'])
                train_count += 1
                if train_count >= n_batch:
                    # Reshape the data
                    for i in range(train_count):
                        n_pad = train_max - len(train_batch['atomic_numbers'][i])
                        train_batch['atomic_numbers'][i] = np.pad(train_batch['atomic_numbers'][i], (0, n_pad), constant_values=-1)
                        train_batch['coordinates'][i] = np.pad(train_batch['coordinates'][i], ((0, n_pad), (0,0)))
                        train_batch['forces'][i] = np.pad(train_batch['forces'][i], ((0, n_pad), (0,0)))
                        for property in self.atomic_properties:
                            train_batch[property][i] = np.pad(train_batch[property][i], (0, n_pad))
                    dest.create_group(f'train/{train_current}')
                    for property in ['atomic_numbers', 'coordinates', 'forces'] + self.atomic_properties + self.summary_properties:
                        dest.create_dataset(f'train/{train_current}/{property}',\
                                            data=np.stack(train_batch[property], axis=0))
                        # Reset
                        train_batch[property] = []
                    train_current += 1
                    train_count = 0
                    train_max = 0
        # Flush out the last data
        if test_count > 0:
            for i in range(test_count):
                n_pad = test_max - len(test_batch['atomic_numbers'][i])
                test_batch['atomic_numbers'][i] = np.pad(test_batch['atomic_numbers'][i], (0, n_pad), constant_values=-1)
                test_batch['coordinates'][i] = np.pad(test_batch['coordinates'][i], ((0, n_pad), (0,0)))
                test_batch['forces'][i] = np.pad(test_batch['forces'][i], ((0, n_pad), (0,0)))
                for property in self.atomic_properties:
                    test_batch[property][i] = np.pad(test_batch[property][i], (0, n_pad))
            dest.create_group(f'test/{test_current}')
            for property in ['atomic_numbers', 'coordinates', 'forces'] + self.atomic_properties + self.summary_properties:
                dest.create_dataset(f'test/{test_current}/{property}',\
                                    data=np.stack(test_batch[property], axis=0))
        if train_count > 0:
            for i in range(train_count):
                n_pad = train_max - len(train_batch['atomic_numbers'][i])
                train_batch['atomic_numbers'][i] = np.pad(train_batch['atomic_numbers'][i], (0, n_pad), constant_values=-1)
                train_batch['coordinates'][i] = np.pad(train_batch['coordinates'][i], ((0, n_pad), (0,0)))
                train_batch['forces'][i] = np.pad(train_batch['forces'][i], ((0, n_pad), (0,0)))
                for property in self.atomic_properties:
                    train_batch[property][i] = np.pad(train_batch[property][i], (0, n_pad))
            dest.create_group(f'train/{train_current}')
            for property in ['atomic_numbers', 'coordinates', 'forces'] + self.atomic_properties + self.summary_properties:
                dest.create_dataset(f'train/{train_current}/{property}',\
                                    data=np.stack(train_batch[property], axis=0))
        dest.close()
        