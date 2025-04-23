FL more concretely refers to all privacy-preserving decentralized collaborative machine learning techniques.

Our approach would be cross-silo fl (large institutions with their own datasets)
Open problems in FL from [[Challenges, methods, and future directions in FL (2020)]] that we aim to tackle & how: Privacy, benchmarking & supervised learning. We aim to tackle all 3 in thesis, namely using Differential privacy, SSL, and 2-sided benchmarking where we try to adhere to accepted standards

When talking about FL, all the papers can be used in FL, but this one too [[Federated learning review. Fundamentals, enabling technologies, and future applications]]
Don't forget to use the original paper of McMahan & Ramage 2015 (introduced), follow ups: 2016, or 2017

Applications of FL: [[Federated learning review. Fundamentals, enabling technologies, and future applications]]

Apparently in SSL, the amount of incorrect pseudolabels being incorporated into the training is not negligble and impacts performance [[Do not trust what you trust. Miscalibration in Semi-supervised Learning]]

When talking about data amount and data collection (unlabeled data & ethical concerns of labelling) use this paper: [[Applications of Self-Supervised Learning to Biomedical Signals. A Survey (2023)]]
Can use [[FedECG]] paper as well when talking about smart wearables that collect data (they sourced it from Zhang et al, Chap 2.1 in FedECG paper)

When talking about SSL being useful, all SSL papers can be referenced, "Amount of unlabeled data >> labeled data --> ssl is useful"


When talking about SSL & Contrastive learning use this paper [[A Survey on Contrastive Self-Supervised Learning]]:

Use an example for contrastive learning to show that it is a "discriminative" approach [[A Survey on Contrastive Self-Supervised Learning]]

We create augmentations of our samples and these are considered as positive samples, which we strive to minimize the difference in embedding/feature space with, while the rest of the data are considered negative samples, where we want to maximize the distance.

The representations learned in this step can then be used for further downstream tasks.

When talking about preprocessing, the [[Fed‑CL‑ an atrial fibrillation prediction system using ECG signals employing federated learning mechanism]] paper talks about a paper using CWT and CNN with ResNet for training (outperforming rest)

Any mention of SimCLR, BYOL, and MoCo can reference [[Applications of Self-Supervised Learning to Biomedical Signals. A Survey (2023)]]. For SimCLR, do mention the original paper.

"It seems likely that this strategy can be useful when performing DL based biosignal analysis "[[Applications of Self-Supervised Learning to Biomedical Signals. A Survey (2023)]]

For data privacy and security, GDPR of 2016 has imposed heavy regulations [[FedECG]]
Another issue is long-tailed distribtuion & non-iid data (the non-iid is mentioned in ALOT of papers) [[FedCVD]] FedCVD also mentions label incompleteness (weaker hopsitals cant identify/label all data)

We start with semi-supervised, this is from van and Hoos 2020 [[FedECG]]

When talking about the general pipeline of Semi-SL, mention the FixMatch paper, they did an ablation study on the confidence thresholding.

We will perform our experiments similar to how [[FedECG]] doees them as this sets a standard: Compare with diff number of clients, level of non-iid and model aggregation algorithm.

We will use Horizontal FedSSL (Chap 2.3 in FedECG by Yang et al.)

Look at the CinC dataset for ECG data.


In FSSL there is not only vertical FL and such but also 3 lines focusing on [[Class Balanced Adaptive Pseudo Labeling for Federated Semi-Supervised Learning]]:
- Only server has some labelled data
- All clients have a bit of labelled data
- Some clienst are fully labelled others are fully unlabelled
Pseudo-labelling shown to be effective in SSL [[Class Balanced Adaptive Pseudo Labeling for Federated Semi-Supervised Learning]]. Existing
FSSL works only adopt standard pseudo labeling or
consistency regularization on FSSL.

AND for FL theres also local training-oriented methods vs model aggregation oriented methods

If using dynamic threshold, look at [[Class Balanced Adaptive Pseudo Labeling for Federated Semi-Supervised Learning]] again

Do we do labels-at-client??