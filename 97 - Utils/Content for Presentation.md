1. Overview of what we will see

2. Situate my work with others (CBAFed, DDRFed, CBDPL which keeps subset labels for a little & takes a bigger and bigger subset) + add 2-3 generic ones (which combine fedSSL & perhps their focus & dataset & summary) & Talk about how FixMatch + FedAvg is widely used (though the combination is also critiqued and it has been mentioned that the two can not just simply be combined)

3. Recap on federated learning
4. Why do we choose semi-supervised & what is it?
	- Faster than self-supervised, more realistic to our scenario

5. What is the focus of the thesis?
	- Adaptive pseudolabel thresholds (similar to learning schedules, first you learn perhaps useless data but this gets filtered out later on) (e.g. santa claus n such)
	- Retaining labels for X rounds (Short term memory inspiration)
	- *NOT self-supervised server* (too much combination of different methods, can alter results making it harder), we used backbone & head approach with different LR but opted out of it
	
	**I think we want to focus on threshold altering per epoch on the model**
	
	**Though both have been performed in research; the combination of both has not. And might prove for an interesting study.**

6. Other choices:
	- Labels at clients (semi-labeled)
	- We use FixMatch: It is the simplest model out there and obtains close-to state-of-the-art results
	- We want to focus on performance across datasets & an ablation study to prove the generalization power of our approach (and some focus on non-iid as that is prominent in the field)
	 - Anything else we prefer not to address as we either find it is addressed softly (privacy) or not within our scope.
	- we currently use a Resnet18 as it is a reasonably simple model structure, but we might go bac kto ResNet9 to again have simpler results and allow for simpler deduction
	- We use FedAvg to aggregate weights, though this is not the best approach and perhaps some type of weighted or even gradient approach with weights might be better,; it is simple and allows us to deduce our results.
	- We used to use u_ratio to ensure the samples in a batch are X times more unlabeled than labeled, this caused issues as we didn't use all available labeled batches.
7. Current progress
	- All models are made
	- we performed evaluation to see what was going wrong in our implementation and have gained deepr understanding
	- lots of metrics collected & scripts made to perform these visualizations & tests

8. Next: 
	- Storing the pseudolabels at the client for X rounds
	- Hyperparam tuning & different datasets
	- Non-iid
	- writing.

9. Future work:
- Addition of a self-supervised server which and spread back up into backbone & head, might be interesting
- More focus on non-iid in different cases
- Different schemes for the change in the adaptive pseudolabel threshold
- Threshold changning over rounds rather than epochs?
