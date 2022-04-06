import numpy as np
import pandas as pd
import sys, os
from random import shuffle
from tqdm import tqdm
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from lifelines.utils import concordance_index

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data.y = data.y.to(device)
        data.x = data.x.to(device)
        optimizer.zero_grad()
        output = model(data)
        #if batch_idx % LOG_INTERVAL == 0:
         #   print('Output ' + str(output))
          #  print('Truth ' + str(data.y.view(-1, 1).float()))
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds.to(device), output), 0)
            total_labels = torch.cat((total_labels.to(device), data.y.view(-1, 1).to(device)), 0)
    return total_labels.cpu().numpy().flatten(),total_preds.cpu().numpy().flatten()


datasets = [['bdtdc_kd','bdtdc_ki','bdtdc_ic50'][int(sys.argv[1])]] 
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5", "cuda:6","cuda:7"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    if dataset == 'Ki' or dataset == 'Kd' or dataset == 'IC50':
        processed_data_file_train = 'data/bindingdb/processed/bindingDB_' + dataset + '_train.pt'
        processed_data_file_test = 'data/bindingdb/processed/bindingDB_' + dataset + '_test.pt'
    else:
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        if dataset == 'Ki' or dataset == 'Kd' or dataset == 'IC50':
            train_data = TestbedDataset(root='data/bindingdb', dataset='bindingDB_'+dataset+'_train')
            test_data = TestbedDataset(root='data/bindingdb', dataset='bindingDB_'+dataset+'_test')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset+'_train')
            test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        # device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        # model = modeling().to(device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = modeling()
        model= nn.DataParallel(model)
        model.to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            G,P = predicting(model, device, test_loader)
            temp_mse = mse(G,P)
            temp_ci = concordance_index(G,P)
            current_epoch = epoch+1
            # ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
            if temp_mse<best_mse:
                
                torch.save(model.state_dict(), model_file_name)
                if current_epoch == NUM_EPOCHS:
                    G,P = predicting(model, device, test_loader)
                    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),concordance_index(G,P)]
                    with open(result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                best_epoch = current_epoch
                best_mse = temp_mse
                best_ci = temp_ci
                # best_ci = ret[-1]
                print('mse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
            else:
                if current_epoch == NUM_EPOCHS:
                    model.load_state_dict(torch.load(model_file_name))
                    G,P = predicting(model, device, test_loader)
                    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),concordance_index(G,P)]
                    with open(result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', ret[1],ret[-1],model_st,dataset)
                else:
                    print('No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)

