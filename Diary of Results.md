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

**-- 20 07 2025 --**
Trial 18 finished with value: 0.9482949980422862 and parameters: {'lambda_u': 0.10861664382369632, 'threshold_schedule': 'linear', 'lr_backbone': 0.001796602287727979, 'lr_head': 0.0033950709954553163}. Best is trial 18 with value: 0.9482949980422862.

**-- 07-08-2025 --**
*We currently have model collapse again. When we added the averaging of the head weights, we found that our results worsened significantly. But now, we find that at low labeled data % our SSL model performs worse than the supervised. We only use 1 client to identify how this scenario affects our results. This is on MNIST*
```
n_epochs: 2 
n_clients: 1                   
n_rounds: 5                       
labeled_fraction: 0.005           
client_frac: 1.0                  
lambda_u: 1.0                  
u_ratio: 7.0                       
n_pretrain_rounds: 2            
backbone_model: "DEBUG"          
threshold_schedule: "linear"     
initial_threshold: 0.5         
final_threshold: 0.95           
stm: True                        
stm_ordered_dict: False         
stm_ttl: 3                       
stm_max_size: 200                 
stm_keep_prob: 0.01                  
backbone_lr: 0.001                
head_lr: 0.005
```

It was 0.4 but I now fixed the bug (data was being split in an ordered fashion; so 10% of data = only label 0) and will show the results:

``lambda_u = 0``
Test Accuracy: 0.8781
Confusion Matrix:
[[ 912    0   10    0    0    1   10    2    2    6]
 [   0 1063   38    3    0    1    4   10    1    0]
 [   2    0  776   11    9    1    1    9    3   18]
 [   3    5   54  921    0   53    1   10   12   10]
 [   3    0   19    0  896    0    7   13    3   32]
 [  22    0    1   31    3  819   71    4   29   24]
 [  19    3   23    0   16    2  860    0    3    2]
 [   1    0   18    3    0    1    0  800    2    4]
 [  18   64   93   36   23   14    4   60  916   95]
 [   0    0    0    5   35    0    0  120    3  818]]
F1 Macro: 0.8785, F1 Weighted: 0.8791, Recall Macro: 0.8789