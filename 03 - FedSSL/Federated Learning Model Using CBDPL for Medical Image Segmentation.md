#FedSSL

> Note: Uses confidence-based dynamic pseudo-labeling, which wouldbe our approach (named CBDPL). Closest to our research.


They don't only use pseudo-labeling, but they only use a subset of generated pseudo-labels at each step (to avoid flooding). And they keep the pseudolabels in the dataset. Does this lead to overconfident results?

This paper also makes sure that classes with less labels are weighted heavier?
`But why? Is the labeling for the odds of being included or just??`

![[Federated_Learning_model_using_CBDPL_for_Medical_I.pdf]]