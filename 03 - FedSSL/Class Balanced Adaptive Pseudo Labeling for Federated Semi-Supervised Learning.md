#FedSSL 
Again assumption of some fully labelled & unlabelled clients.

The provided paper addresses the challenge of **federated semi-supervised learning (FSSL)**, where some clients possess labeled data while others have only unlabeled data, often exhibiting **non-independent and identically distributed (Non-IID)** characteristics. To tackle issues like **catastrophic forgetting** and **imbalanced class distributions** that hinder existing FSSL methods relying on standard pseudo-labeling, the authors introduce **CBAFed (Class Balanced Adaptive Pseudo Labeling)**.


We can deal with data heterogeneity either using the aggregation strategy or the local training.

They propose to not use a batch approach to update their model as they think that after X batches u lose degradation, rather they produce all pseudo labels first?

`Note: Why not use model i-1 and i to generate labels n do some hierarchical labelling?
Curriculum pseudo labelling achieves state of the art results using dynamic threshold (basically lower for class c depending on how many pseudolabels there are from class c compared to total )
--> But in FSSL, there exist heterogeneous
data partition problems. Due to the Non-IID partition, the
labeled data are not balanced, so purely using the number
of selected unlabeled data to design threshold is improper. --> use CBAPL

TODO: Check out sources 12,27,28,9,11,12,17,23 (dont want to use attention based)

The "fixed pseudo labeling strategy" in CBAFed is a method designed to address the problem of **catastrophic forgetting** that can occur in federated semi-supervised learning (FSSL) when unlabeled clients train their local models.

In essence, the process is as follows:

1. An unlabeled client receives the global model.
2. Using this model, the client predicts labels for all its unlabeled data points.
3. For each prediction, the model also provides a confidence score (the probability of the predicted class).
4. A pre-defined or adaptive threshold is applied to these confidence scores.
5. **Only the unlabeled data points where the prediction confidence is above the threshold, along with their corresponding predicted labels (pseudo labels), are selected to form the "fixed set."**
6. This "fixed set" is then used as the training data for the unlabeled client during the current communication round. The pseudo labels in this set serve as the target labels for supervised learning on the unlabeled data.

The "fixed" nature of this set is crucial for handling catastrophic forgetting. By determining the pseudo labels and the training set at the beginning of each round and keeping it relatively fixed during local training, the model on the unlabeled client is less likely to drastically forget the knowledge gained from the labeled clients in previous rounds. This is because the supervisory signals (the pseudo labels in the fixed set) provide a consistent learning objective throughout the local training round. This contrasts with batch-based pseudo labeling where pseudo labels might be recalculated and change frequently during local training, potentially leading to instability and forgetting.

![[Li_Class_Balanced_Adaptive_Pseudo_Labeling_for_Federated_Semi-Supervised_Learning_CVPR_2023_paper.pdf]]