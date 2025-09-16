#FedSSL 
labeled & u,labaled clients


RSCFed [14], the state-of-the-art method significantly
boosts the FSSL performance by first distilling subconsensus
models, and then aggregating the sub-consensus
models to the global model. The sub-consensus models can
handle the Non-IID setting to some extent, but the meanteacher
based consistency regularization framework in unlabeled
clients inevitably causes the accuracy degradation
when the classes are imbalanced distributed. RSCFed

We adopt ResNet18 [6] from PyTorch for all
datasets. For fair comparison, we use the same network
architecture and training protocol, including the optimizer,
data preprocessing, etc. across all FSSL methods. The local
training epoch is set to 11 for labeled clients and 1 for unla
beled
clients. More implementation details are provided in
supplementary material.

NOTE UPPER BUOND ACCURACCY FOR 10 LABELED DIR ALPHA = 0.8 AND 11 LOCAL EPOCHS
![[CBAFed.png]]

we enlarge the weight of labeled client to
about 50% and other nine unlabeled clients share the re
maining
50% weight in each FSSL synchronization round,
when implementing these methods. We also compare our
network against FedAvg [19] trained with all clients as an
upper bound and trained with only one labeled clients as
the lower bound. Besides,




In the context of CBAFed, **fixed pseudo labeling** is a strategy designed to prevent a common issue in Federated Semi-Supervised Learning (FSSL) called **catastrophic forgetting**.

Here's how it works in simple terms:

1. **Initial Snapshot:** At the very beginning of an unlabeled client's local training in each communication round, the model evaluates its entire local unlabeled dataset.

2. **Generate a Fixed Set:** It then generates pseudo-labels (predicted labels) for these unlabeled data points, but only selects those with high confidence (exceeding a certain threshold). These selected data points, along with their pseudo-labels, form a **"fixed training dataset"**.

3. **Train on the Fixed Set:** The unlabeled client then performs all its local training epochs within that round _only_ on this fixed pseudo-labeled dataset. It does _not_ re-evaluate and update these pseudo-labels or the dataset during these local epochs.

This "fixed" approach prevents the model from forgetting knowledge learned from labeled clients. Traditional batch-based pseudo-labeling, which frequently updates pseudo-labels, can lead to model degradation due to the diverse and unevenly distributed (Non-IID) data in FSSL. By using a fixed set, CBAFed ensures more stable training for unlabeled clients, significantly improving performance in Non-IID FSSL

Yes, that is correct.

In CBAFed, the threshold used for the **fixed pseudo labeling strategy** is indeed the **class balanced adaptive threshold** that is calculated and updated every communication round


![[CBAFed.pdf]]