# Resources:

+ README.md: this file.
+ Download data from Google drive (https://drive.google.com/drive/folders/1gMUfB6s8YA_o-dy80eOMnTi4LRYeqk3Q?usp=sharing)

###  Source codes:
+ create_data.py: create data (Kd, Ki, IC50) in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
+ training.py: train a GraphDTA model.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.

# Step-by-step running:

## 0. Install Python libraries needed
+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
+ Install rdkit: conda install -y -c conda-forge rdkit
+ Or run the following commands to install both pytorch_geometric and rdkit:
```
conda create -n geometric python=3
conda activate geometric
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
pip install tqdm
pip install lifelines

```

## 1. Create data in pytorch format
Running
```sh
conda activate geometric
python create_data.py
```
This returns data in pytorch format, stored at data/processed/, consisting of *_test.pt and *_train.pt.

## 2. Train a prediction model
To train a model using training data. The model is chosen if it gains the best MSE for testing data.  
Running 

```sh
conda activate geometric
python training.py 0 0 0
```

where the first argument is for the index of the datasets, 0/1/2 for 'BindingDB_Kd', 'BindingDB_Ki' or 'BindingDB_IC50', respectively;
 the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively;
 and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively. 
 Note that your actual CUDA name may vary from these, so please change the following code accordingly:
```sh
cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5", "cuda:6","cuda:7"][int(sys.argv[3])]
```

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.
For example, it returns two files model_GATNet_davis.model and result_GATNet_davis.csv when running GATNet on Davis data.

###  Pretrained models:

All pretrained models for each dataset can be found in "GraphDTA_Results" folder. Example of inference can be found in model_performance.ipynb. Also, the inference for external data are shown in "predict_generated" folder. 


