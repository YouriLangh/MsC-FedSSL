#FL 

Here we focus on horizontal federated learning, where
the parties share the same feature space but different sample
space.

This paper says that the amount of non-iid cases tested in papers is not sufficient and offers a better benchmark AND their results!!!

IMPORTANT to verify results of others with our own.

FedProx is SOTA at the time of this paper and in general?


> Finding (2):
> No algorithm consistently outperforms the other
> algorithms in all settings. The state-of-the-art algorithms
> significantly outperform FedAvg only in several cases.

This paper has alot of findings that are just very valuable for insight.

They mention that just averaging the weights or something might not be good enough for non-iid?
> "Aggregation of Heterogeneous Batch Normalization:
> From
> our Finding (7), simple averaging is not a good choice for
> batch normalization. Since the batch normalization in each
> party records the statistics of local data distribution, there is
> also heterogeneity among the batch normalization layers of 
> different parties. The averaged batch normalization layer may
> not catch the local distribution after sending back to the parties.
> A possible solution is to only average the learned parameters
> but leave the statistics (i.e., mean and variance) alone [4]. More
> specialized designs for particular layers in deep learning need
> to be investigated."

![[Federated Learning on Non-IID Data Silos.pdf]]