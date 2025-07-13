**--- 30/04/2025 ---**
*Implemented FixMatch with FedAvg, using a fresh ResNet-18 backbone for both server & clients.*

> Note: The average loss is that of the final epoch of the final client at the final communication round.

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


**--- 05/07/2025 ---**
*These results do not have altered parameters, rather they are a test with the new adaptive linear thresholding & a split backbone / head (3layer MLP) on both the FixMatch & Supervised Client.*
> Note: The server model training on its own dataset is not implemented yet.
- **FixMatch+FedAvg**: Avg Loss: *0.5779867470264435*  Final Threshold: *0.6250*   Test Accuracy: *0.1812*
- **Fully Supervised Clients**: Avg Loss: *126.80297243439418* ~~Final Threshold: *0.6250* (not used)~~  Test Accuracy: *0.1105*
- **One client/ server with all the data:** Avg Loss: *214.7905* Test Accuracy: *0.9865*

--- 
`All the previous results use the wrong loss for FixMatch, only the final batch loss was added to the total.
*The next tests should have these parameters*
**10 epochs**
**1 communication**
**2 clients**
==> Look at *OPTUNA* for this

**--- 07/07/2025 ---**
- **2 SSL Clients (FixMatch):** Avg Loss of all clients: 85.79248366951943, Threshold: 0.3000 Test Accuracy: 0.1028
![[lc_07-07-2025.png|500]]
![[cm_07-07-2025.png|500]]
- **2 supervised clients:** Avg Loss of all clients: 83.56035264432448, Threshold: 0.300 Test Accuracy: 0.1010
![[sup+fed_lc_07-07-2025.png|500]]![[sup+fed_cm_07-07-2025.png|500]]

- **Only supervised server**: Epoch 10/10, Loss: 48.6660 Test Accuracy: 0.9911
![[lc_sup_07-07-2025.png|500]]![[sup_cm_07-07-2025.png|500]]

**08-07-2025**
**1 client, same params as before BUT server model == client model** to see how the client performs (threshold maybe highered to 0.6)
![[fix+fed_lc_08-07-2025.png|500]]
![[fix+fed_cm_08-07-2025.png|500]]

**2 client with server only using weights of the first model**
![[fix+fed_lc_2cl_08-07-2025.png|500]]
![[fix+fed_cm_2cl_08-07-2025.png|500]]
