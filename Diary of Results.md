**--- 30/04/2025 ---**
*Implemented FixMatch with FedAvg, using a fresh ResNet-18 backbone for both server & clients.*
Parameters: 
- 20% of data per client is labeled
- data is randomly distributed along both clients
- 2 clients
- 2 epochs per client
- 2 communication rounds
- Adam optimizer
- All clients participate
- Unlabeled ratio is 7
- batch-size 32
- dataset: MNIST

Results:
Avg Loss: 0.3492279499769211 Test Accuracy: 0.4179

For a supervised(clients) baseline using same backbone, Test Accuracy: ~0.9888

Global supervised baseline:
Running Fully Supervised model, no Federation
Epoch 2/2, Average Loss: 207.5522
Test Accuracy: 0.9827

`Note: I do not think we actually upload the global weights back to the clients. Nor do we pretrain on only the labeled data for the first X rounds.`


**--- 1/07/2025 ---**
*Implemented FixMatch with FedAvg. Added pre-training rounds and weight averaging is functional now as well. Unsupervised loss is 0 meaning that the model is unable to make accurate predictions which makes sense with the low number of rounds/epochs.*
Parameters:
- All the same, add n_pretrain_rounds = 0.

Results:
- FixMatch+FedAvg: Avg Loss: 2.8579981; Test accuracy: 0.0777
- Every client contains fully labeled data: Avg Loss: 2363.6249460577965; Test Accuracy: 0.1059
- One client/ server with all the data: Average Loss: 202.2567; Test Accuracy: 0.9843