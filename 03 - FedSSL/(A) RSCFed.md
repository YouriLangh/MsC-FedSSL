#FedSSL 

followed by many heterogeneous FL solutions,
which could be categorized into two branches: local training-oriented methods [16] and model aggregation oriented methods.


follow existing methods
[16,26,32] to use a Dirichlet distributionDir($\gamma$)= ($\gamma$= 0.8)
for
(all three benchmark datasets) to generate the non-IID
datapartition in clients. After such a Non-IID data partition
strategy,the number of classes and samples at each client
differ from each other, and thus not all clients contain samples
from all classes.


Here's a breakdown of how RSCFed works in simple terms, addressing your specific points:

1. **Clients do** **not** **perform training and** **then** **get sampled.** Instead, in each synchronization round, the server first **randomly samples a subset of clients** from all available clients. This random sub-sampling operation is performed multiple times (M times) to create several distinct subsets of clients.

    ◦ The current global model is then sent to these _selected_ clients for initialization.

    ◦ Only _after_ being selected and initialised do the clients perform their **local training**.

        ▪ Labeled clients train using a standard cross-entropy loss.

        ▪ Unlabeled clients train using a mean-teacher-based consistency regularization framework.

    ◦ Each client then sends its updated local model back to the server.

2. **How sub-consensus models are formed:** For each of the randomly sampled subsets of clients, the server performs a **Distance-Reweighted Model Aggregation (DMA)** to create a "sub-consensus model".

    ◦ The process starts by calculating a temporary, simple average model (`θavg`) from the local models within that specific subset, typically weighted by the amount of local data each client holds.

    ◦ Then, a crucial step occurs: the DMA module **dynamically adjusts the weights of each client's model within that subset**. It assigns **higher weights to models that are "closer"** to this temporary average model within the subset, and lower weights to those that are further away. The distance is measured using the L2 Norm of the model gradient, scaled by a hyper-parameter `β` and divided by the local data quantity to reduce the impact of local iterations on model drift. This helps to make the sub-consensus model more robust by giving more influence to models that are more in "agreement" with the temporary average of the subset, similar to how RANSAC identifies outliers.

    ◦ This re-weighted aggregation of the sampled client models forms one **sub-consensus model** (`θmsub`).

    ◦ Since the initial random sub-sampling was done `M` times, this results in `M` different sub-consensus models.

3. **Global averaging step:** After all `M` sub-consensus models have been generated, they are aggregated to form the new global model. Critically, these **sub-consensus models are equally weighted averaged** to obtain the final global model for the next round.

They mostly use the labeled clients vs unlabeled clients scenario, though they have like 80% accuracy on SVHN with partially labeled clients:
> i.e., only 10% images are labeled. For this setting,
we adopt same network backbone as in the previous setting.
Since all clients are partially labeled, no weight scaling operation is performed.

![[(A) RSCFed.png]]


![[Liang_RSCFed_Random_Sampling_Consensus_Federated_Semi-Supervised_Learning_CVPR_2022_paper 1.pdf]]