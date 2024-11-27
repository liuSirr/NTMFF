# NTMFFDTA
We propose a new network topology and multi-feature fusion-based approach for DTA prediction (NTMFF-DTA), which deeply mines protein multiple types of data and propagates drug information across domains. Data in drug-target interactions are often sparse, and multi-feature fusion can enrich data information by integrating multiple features, thus overcoming the data sparsity problem to some extent. The proposed approach has two main contributions: 1) constructing a relationship-aware GAT network that selectively focuses on the connections between nodes and edges in the molecular graph to capture the more critical roles of nodes and edges in DTA prediction and 2) constructing an information propagation channel between different feature domains of drug proteins to achieve the sharing of the importance weight of drug atoms and edges, and combining with a multi-head self-attention mechanism to capture residue-enhancing features.

## dependencies
python == 3.8.12 <br>
numpy == 1.21.2 <br>
pandas == 1.5.3 <br>
rdkit == 2022.9.5 <be>
scikit-learn == 1.3.2 <br>
Pconsc4 == 0.4 <br>
pytorch == 2.4.1 <br>
PyG (torch-geometric) == 2.6.1 <br>


## data preparation
1. Prepare the data need for train. Run the create_data file to generate the data format needed to train the model. <br>
**python create_data.py** <br><br>

#### Davis and KIBA
https://github.com/thinng/GraphDTA/tree/master/data

## train model
Run the following script to train the model.<br>
**python training2.py** <br>
Then, in the process, the model is simultaneously tested in the validation set, and as the value of mse decreases, the parameter values of the model are updated accordingly, and the parametric model with the best results is finally saved.
## test model
Run the following script to test the model.<br>
**python training_validation.py** <br>
This will return the best MSE model for the test dataset during the validation process.
