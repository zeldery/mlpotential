'''
Train the xdm components
'''

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mlpotential.dataloader import DataIterator
from mlpotential.combine import DispersionModel
import argparse
import time

def get_argument():
    parser = argparse.ArgumentParser('train_xdm', 
                                     description='Train dispersion model')
    parser.add_argument('-m', '--model', default='model.pt')
    parser.add_argument('-t', '--target')
    parser.add_argument('-d', '--data', default='data.hdf5')
    parser.add_argument('-c', '--checkpoint', default='checkpoint.pt')
    parser.add_argument('-g', '--gpu', default='0')
    parser.add_argument('-e', '--epoch', default='100')
    parser.add_argument('-l', '--learningrate', default='0.001')
    parser.add_argument('-r', '--restart', default='0')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    model = DispersionModel()
    model.read(args.model)
    if args.gpu =='0':
        device = torch.device('cpu')
    elif args.gpu == '1':
        device = torch.device('cuda')
    else:
        raise ValueError(f'Unvalid value for gpu argument of {args.gpu}')

    if args.target == 'M1':
        data = DataIterator(args.data, ['atomic_numbers', 'coordinates', 'M1'])
    elif args.target == 'M2':
        data = DataIterator(args.data, ['atomic_numbers', 'coordinates', 'M2'])
    elif args.target == 'M3':
        data = DataIterator(args.data, ['atomic_numbers', 'coordinates', 'M3'])
    elif args.target == 'V':
        data = DataIterator(args.data, ['atomic_numbers', 'coordinates', 'Veff'])
    else:
        raise ValueError(f'Invalid target {args.target}')

    loader = data.dataloader(shuffle=True)
    criterion = torch.nn.MSELoss()
    criterion_eval = torch.nn.MSELoss(reduction='sum') # To return RMSE, need to sum up all the MSE
    model = model.to(device=device)
    # Record the run
    if args.restart == '0':
        train_time = []
        validation_time = []
        start_iteration = 0
        rmse_train = []
        rmse_test = []
        best_rmse = 100000.0
        best_model = None
        bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
        weight_params = [p for name, p in model.named_parameters() if 'weight' in name]
        optimizer = torch.optim.AdamW([{'params': bias_params, 'weight_decay':0.0},
                                    {'params': weight_params, 'weight_decay':0.0001}
                                        ], lr=float(args.learningrate))
    elif args.restart == '1':
        save_dict = torch.load(args.checkpoint, map_location=device)
        train_time = save_dict['train_time']
        validation_time = save_dict['validation_time']
        start_iteration = save_dict['epoch_finished']
        rmse_train = save_dict['rmse_train']
        rmse_test = save_dict['rmse_test']
        best_rmse = save_dict['best_rmse']
        best_model = save_dict['best_model']
        model.load(save_dict['current_model'])
        model = model.to(device=device)
        bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
        weight_params = [p for name, p in model.named_parameters() if 'weight' in name]
        optimizer = torch.optim.AdamW([{'params': bias_params, 'weight_decay':0.0},
                                    {'params': weight_params, 'weight_decay':0.0001}
                                        ], lr=float(args.learningrate))
        optimizer.load_state_dict(save_dict['optimizer'])
    else:
        raise ValueError(f'Incorrect option for restart {args.restart}')
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=100, threshold=0)

    for epoch in range(start_iteration, int(args.epoch)):
        torch.cuda.synchronize() # For correct time record
        begin_time = time.time()
        data.mode = 'train'
        for batch_data in loader:
            optimizer.zero_grad()
            atomic_numbers = batch_data['atomic_numbers'].to(torch.int64).to(device)
            positions = batch_data['coordinates'].to(torch.float32).to(device)
            if args.target == 'M1':
                ref = batch_data['M1'].to(torch.float32).to(device)
                pred = model.batch_compute_m1(atomic_numbers, positions)
            if args.target == 'M2':
                ref = batch_data['M2'].to(torch.float32).to(device)
                pred = model.batch_compute_m2(atomic_numbers, positions)
            if args.target == 'M3':
                ref = batch_data['M3'].to(torch.float32).to(device)
                pred = model.batch_compute_m3(atomic_numbers, positions)
            if args.target == 'V':
                ref = batch_data['Veff'].to(torch.float32).to(device)
                pred = model.batch_compute_v(atomic_numbers, positions)
            loss = criterion(pred, ref)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        train_time.append(time.time() - begin_time)
        # Evaluation
        begin_time = time.time()
        with torch.no_grad(): # For faster evaluation
            n_atoms = 0
            total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            for batch_data in loader:
                #### Change here for different field name
                atomic_numbers = batch_data['atomic_numbers'].to(torch.int64).to(device)
                positions = batch_data['coordinates'].to(torch.float32).to(device)
                if args.target == 'M1':
                    ref = batch_data['M1'].to(torch.float32).to(device)
                    pred = model.batch_compute_m1(atomic_numbers, positions)
                if args.target == 'M2':
                    ref = batch_data['M2'].to(torch.float32).to(device)
                    pred = model.batch_compute_m2(atomic_numbers, positions)
                if args.target == 'M3':
                    ref = batch_data['M3'].to(torch.float32).to(device)
                    pred = model.batch_compute_m3(atomic_numbers, positions)
                if args.target == 'V':
                    ref = batch_data['Veff'].to(torch.float32).to(device)
                    pred = model.batch_compute_v(atomic_numbers, positions)
                mask = (atomic_numbers.flatten() != -1)
                n_atoms += mask.sum().item()
                loss = criterion_eval(pred.flatten()[mask], ref.flatten()[mask])
                total_loss += loss
            rmse_train.append((total_loss.detach().tolist()/n_atoms)**0.5)
            
            data.mode = 'test'
            n_atoms = 0
            total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            for batch_data in loader:
                #### Change here for different field name
                atomic_numbers = batch_data['atomic_numbers'].to(torch.int64).to(device)
                positions = batch_data['coordinates'].to(torch.float32).to(device)
                if args.target == 'M1':
                    ref = batch_data['M1'].to(torch.float32).to(device)
                    pred = model.batch_compute_m1(atomic_numbers, positions)
                if args.target == 'M2':
                    ref = batch_data['M2'].to(torch.float32).to(device)
                    pred = model.batch_compute_m2(atomic_numbers, positions)
                if args.target == 'M3':
                    ref = batch_data['M3'].to(torch.float32).to(device)
                    pred = model.batch_compute_m3(atomic_numbers, positions)
                if args.target == 'V':
                    ref = batch_data['Veff'].to(torch.float32).to(device)
                    pred = model.batch_compute_v(atomic_numbers, positions)
                mask = (atomic_numbers.flatten() != -1)
                n_atoms += mask.sum().item()
                loss = criterion_eval(pred.flatten()[mask], ref.flatten()[mask])
                total_loss += loss
            scheduler.step(total_loss)
            rmse_test.append((total_loss.detach().tolist()/n_atoms)**0.5)
        torch.cuda.synchronize()
        validation_time.append(time.time() - begin_time)
        # Check best model
        if rmse_test[-1] < best_rmse:
            best_rmse = rmse_test[-1]
            best_model = model.dump()
        # Saving checkpoint
        save_dict = {'optimizer': optimizer.state_dict(), 'epoch_finished': epoch+1, \
                     'rmse_train': rmse_train, 'rmse_test': rmse_test, \
                     'train_time': train_time, 'validation_time': validation_time,\
                     'best_rmse': best_rmse, 'current_model': model.dump(), 'best_model': best_model}
        torch.save(save_dict, args.checkpoint)

if __name__ == '__main__':
    main()

