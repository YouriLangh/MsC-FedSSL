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

Test Accuracy: 0.1028
Confusion Matrix:
[[   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [ 980 1135 1032 1010  982  892  958 1028  974 1009]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]]
F1 Macro: 0.0186, F1 Weighted: 0.0192, Recall Macro: 0.1000

*lets alter lambda_u to 0.5*
Same ish results.

*Now I alter the initial threshold to 0.8, lets see if we just had noisy labels*
Test Accuracy: 0.4869
Confusion Matrix:
[[ 863    0  277   18   16    6  334    1    9    6]
 [   0 1096  181  106    8   25  113   39  121    9]
 [   0    0  190    2    0    0    0    0    0    0]
 [   0    0    5   59    0    0    0    0    0    1]
 [   0    0    7    0  370   11    8   43    6   43]
 [  70   34  219  548   38  732   72   37  422   15]
 [   0    0   28    2    1    1  233    0    1    0]
 [   0    0   33   68    2    6    0  388   10    1]
 [   0    0    4    0    0    0    0    0    4    0]
 [  47    5   88  207  547  111  198  520  401  934]]
F1 Macro: 0.4214, F1 Weighted: 0.4257, Recall Macro: 0.4837
*Turn off STM now to see if it has negative effect, increase threshold or make it constant turns this all to 0.25*
Test Accuracy: 0.3867
Confusion Matrix:
[[  0   0   0   0   0   0   0   0   0   0]
 [  0  64   0   0   0   0   0   0   0   0]
 [  0   6 560   0  14   2  38   9   1   4]
 [931 713 458 974  44 444 506  63 645  30]
 [  0   0   3   0  93   0   0   0   0   0]
 [ 18   2   0   2   8 398  20   1 118   2]
 [  0   1   0   0  12   0 383   0   0   0]
 [  9 173   3   4   1   0   1 424   3   3]
 [  0  17   0   0   0   0   0   0   1   0]
 [ 22 159   8  30 810  48  10 531 206 970]]
F1 Macro: 0.3423, F1 Weighted: 0.3396, Recall Macro: 0.3879
``lambda_u = 0.1 & initial_threshold = 0.8`` 

Test Accuracy: 0.2486
Confusion Matrix:
[[ 980 1133 1007 1009  181  756  954  626  973  626]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    8    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    2   17    1  788    9    4    1    0  188]
 [   0    0    0    0    0  120    0    0    1    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0  395    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0   13    7    0    6    0  195]]
F1 Macro: 0.2128, F1 Weighted: 0.2103, Recall Macro: 0.2522

*Upon reducing lambda_u to 0.001 we get:*
0.1 collapse


``lambda_u = 0``
Test Accuracy: 0.8969
Confusion Matrix:
[[ 946    0    8    1    1    3   16    0    5   14]
 [   0 1085    1    2   12    0    3   17    0   23]
 [   2   16  964   68   10   10   27   24   31    9]
 [   0    0    6  859    0   44    0    0    5   12]
 [   0    0    7    0  909    3    3    4    3   40]
 [   5    1    0    3    0  678   20    1    2    4]
 [   5    5    2    1   12   12  873    0    7    0]
 [  17    0   29   28    2    4    1  963   13   50]
 [   5   28   15   46   27  136   15    6  907   72]
 [   0    0    0    2    9    2    0   13    1  785]]
F1 Macro: 0.8962, F1 Weighted: 0.8974, Recall Macro: 0.8949

**-- 08-08-2025 --**
*We currently have model collapse (on MNIST). We use the same parameters as yday expect we do:*
``threshold_schedule = "constant" & lambda_u = 1.0``
<span style="color:rgb(255, 0, 0)">Round 1, Avg Loss of all clients: 18.4588<br>
Round 2, Avg Loss of all clients: 6.940<br>
Round 3, Avg Loss of all clients: 72.2808<br>
Round 4, Avg Loss of all clients: 19.178<br>
Round 5, Avg Loss of all clients: 125.5044</span>
Test Accuracy: 0.6001
Confusion Matrix:
[[897   0  12   1   3   1  12   3   3  11]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   9 610   4   0   1   1   5   4   0]
 [  0   2  23 170   0   0   0   2  19   2]
 [  0 235   9   1 785   1   5  11   5  26]
 [ 28  49  99 782   9 870  19  29 754  53]
 [ 52  24 240  14  35  15 920   0  24   2]
 [  0   0  12   9   0   0   0 725   6  11]
 [  2 726  17  13   1   1   0   9 128   8]
 [  1  90  10  16 149   3   1 244  31 896]]
F1 Macro: 0.5673, F1 Weighted: 0.5606, Recall Macro: 0.6134


*We will try to alter the losses by weighing them with their confidence.*
``After adding confidence weighted unsupervised loss``
<span style="color:rgb(255, 0, 0)">Round 1, Avg Loss of all clients: 18.4588 <br>
Round 2, Avg Loss of all clients: 6.940<br>                                        Round 3, Avg Loss of all clients: 60.6782<br>                                      Round 4, Avg Loss of all clients: 16.7235<br>                                      Round 5, Avg Loss of all clients: 95.9838<br>
</span>
Test Accuracy: 0.5694
Confusion Matrix:
[[932   1  36   3   4   3 206   0   1   7]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   1 357   1   0   0   5   1   2   0]
 [  0  12 182 582   0  83   5   2   3   5]
 [  0 161   5   2 816   0 107  10   0  15]
 [ 13   2  11  37   1 427  15   6  43   7]
 [  0   0   0   0   0   0   1   0   0   0]
 [ 15   0  81  84  13  22   3 945  37 153]
 [ 20 858 354 294  94 344 615  18 864  52]
 [  0 100   6   7  54  13   1  46  24 770]]
F1 Macro: 0.5291, F1 Weighted: 0.5238, Recall Macro: 0.5753

*After this, we will try to reduce lambda_u to 0.4 to ensure model doesnt over fit on the pseudolabels as we see that the unsupervised loss reaches 0.0 at some point while the supervised loss skyrockets to 234 or 130 even.*
`lambda_u = 0.4`
<span style="color:rgb(255, 0, 0)">Round 1, Avg Loss of all clients: 18.4588<br>
Round 2, Avg Loss of all clients: 6.940<br>                                            Round 3, Avg Loss of all clients: 43.1158<br>                                          Round 4, Avg Loss of all clients: 64.5994<br>                                         Round 5, Avg Loss of all clients: 46.1041</span>
Test Accuracy: 0.1216
Confusion Matrix:
[[ 234    0    0    5    0    2    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [ 746 1135 1032 1005  982  890  958 1028  974 1009]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]]
F1 Macro: 0.0566, F1 Weighted: 0.0555, Recall Macro: 0.1239

*Removing the STM+ working loader bs to see if supervised loss improves. We collapsed still, but perhaps i had a bug with workingloader = labeled_loader type shit*

**---- 05-09-2025 [Post Pieter Meeting] ----**
To investigate the effectiveness of our architecture, we performed several tests in different scenarios. These scenarios show how the test performance of the global model changes over different lambda_u values (0; 0.5; 1). We compare the performance of these global models for clients with different labeled_fractions. Each graph shows a certain data distribution per client, non-IID to IID ($alpha$ = 0.1, 0.5, IID).
This means that in a specific data distribution scenario, we will visualize the performance of each global model for clients with different fractions of labels for their data across different lambda_u values.
![[Diary of Results - Test1-4.png|500]]
The red line indicates a labeled_fraction of 1, where each client has access to all the labels of its dataset. As the X-axis shows the effect over different lambda_u values, this line should remain stable for all x-axis values. Any case where this is not the case should be seen as random variance and will be investigated further. The colors indicate the labeled_fraction with darker indicating access to more labels. We urge the readers to observe that the axes are NOT the same value across graphs. (A chart with these same axes will be given in the appendix)
![[Diary of Results - Test1-4Shared.png|500]]

*Test 2:* For the "first" test, we investigate the general simplified scenario where our architecture consists of 10 clients over 10 rounds where each client trains for 10 epochs. 
- **IID:** We find that the global model performance in the iid case collapses. We not that this is most likely due to the fact that the individual feature extractors of each client are more distinct and this small might cause collapse? Due to this collapse, we see that the parts where each client has access to less labels has an accuracy of around 0.8. Where the performance increases for labeled_fraction going up.
> Needs more investigation (labeled_fraction = 1 seems off.)
- **Non-IID (0.1):** We note that our model performs worse in non-IID situations and that a mixed trend is observed by increasing the weight of the unsupervised loss. This means either the SSL aspect or FedAvg is not functioning in this scenario. Most likely both have a negative effect.
- **Non-IID (0.5):** Here the trends for increasing the unsupervised loss are still mixed but mostly negative. Most cases where the labeled_fraction is lower than 1 actually outperform the baseline which requires further investigation.
From this we conclude that the performance of the architecture is significantly worse in the non-IID case in lower round scenarios which can be attributed to either FedAvg or SSL.

*Test 4:* Due to these extremely varying results, we investigate that the code actually works by running the same test but with a single client. This eliminates the FedAvg dependency and allows us to focus on correctness of data distribution & SSL.
We find that labeled_fraction = 1 actually now shows as an upper baselines as is expected in all scenarios.
We note that across all IID scenarios, increasing the lambda_u can have a positive effect, though not monotonically. We see signficant improvements in the performance at lambda_u = 0.5. This means that it is a tunable parameter which can differ per scenario. We still see performance degradation in the non-IID case, but the performance is extremely close. We see this similarly to

[[Feedback-Driven Pseudo-Label Reliability Assessment. Redefining Thresholding for Semi-Supervised Semantic Segmentation]] 
> Does mention that the model degradation due to choice of confidence threshold does not follow a consistent pattern.
"As labeled dataset size decreases, the optimal threshold
becomes increasingly unpredictable"

Who found that increasing the threshold has an inconsistent pattern with model performance, we note that this is also the case for FedSSL.

This allows us to see that each individual client actually is able to learn the representation correctly and that SSL has a positive influence. FedAvg on the other hand struggles to create a correct model in higher client numbers and non-IID environments as can be seen in *Test 3*. Significant performance degradation again in non-IID cases but we outperform the baseline in some cases at IID. Though this does not directly refute the statements made in [[(A) Federated Learning Original]] about FedAvg being built for non-IID cases, as they require many rounds. We therefore test our model in higher round scenarios.

Before we can get to that, we investigate whether our custom featuer (STM) has a positive effect. We note that the effects are marginally improving the performance with one notable improvement for lf=0.01 in iid where it goes from 0.2 to 0.6. Other than that the results are mixed.
[[(A) Federated learning with non-iid data]] says: "We further show that this accuracy reduction
can be explained by the weight divergence"

*100 Round test:* We continue the same test (but with STM on) for 2 clients over 100 rounds.
![[Diary of Results Test5-valaccfixed.png]]
In this chart we show the validation accuracy per round of the global model after aggregating the weights of the client models. We see that after pretraining (a certain number of rounds where only labeled data is used to train the client models), a degredation of performance can be found. This degradation is way more pronounced in non-IID cases, showing again that FedAvg struggles in non-IID environments. We see that the model stabilizes and does not improve over the 100 rounds, showing that the performance completely depends on the pretraining rounds. However, for the non-IID case, we see that perhaps this drop can be reduced by adapting a different averaging system other than FedAvg. 

[[(A) Federated learning with non-iid data]] aldo found a decrease in performance for non iid m,ist up to 11%; 51 for cifar

We ran the same test for 3 clients:
![[Diary of Results 100rounds-valacc 3clients.png]]
Here we again observe the degradation in performance once we start using unlabeled samples, but the variance over time is way larger, with a sometimes a positive trend. To further investigate how the architecture works, we run the same test but for 200 rounds, to confirm whether ([[(A) Federated Learning Original]]) FedAvg also works in non-IID environments for low-labeled scenarios.

We ifnd a positive/mixed trend over time for the non-IID situations, in IID  situations we note a negative correlation between the increase in rounds and the accuracy.


This test is only performed for 3 clients:
![[Diary of Results - 200 rounds fixed.png]]
We again ifnd the same variance as we did before with a positive trend over time for the non-IID situations, in IID  situations we note a negative correlation between the increase in rounds and the accuracy.


Finally for this final test, we also look at the losses of each client:
![[Diary of Results - Losses.png]]
This is for the non-iid case with lu=0.5. As the model performance is relatively low, we find that the model is overfitting extremely hard. Perhaps we can decrease the number of epochs at each client though [[(A) Federated Learning Original]] did note:
> "Batch-size = 10, more epochs (20) is best"


> This might need to be somewhere:
> [[Feedback-Driven Pseudo-Label Reliability Assessment. Redefining Thresholding for Semi-Supervised Semantic Segmentation]] Does mention that the model degradation due to choice of confidence threshold does not follow a consistent pattern.
"As labeled dataset size decreases, the optimal threshold
becomes increasingly unpredictable"

> Importantly we note that: we are not using any pretrained weights & none of the hyperparameters have been tuned optimally, just random hyperparams, perhaps we can improve everything. Due to the fact that most of these hyperparameters are extremely situation-dependent & inconsistent in tuning.


