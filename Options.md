
### **Options Thesis Revised**

1. **Federated Learning Types:**
    
    - **Horizontal FL:** Clients share the same feature space but have different data samples.
        
    - **Vertical FL:** Clients share the same data samples but have different feature spaces.
        
    - **Federated Transfer Learning:** Used when clients have neither many overlapping samples nor features but can benefit from each other's knowledge.
        
2. **Ownership of Data of FL Clients:**
    - Only the server has some labeled data _(FedMatch does this, not FedCon, which focuses on contrastive learning for representation learning)._
        
    - Some clients are fully labeled, and others are fully unlabeled _(common in federated semi-supervised learning, e.g., FedMatch, FedSEAL)._
        
    - Clients can have varying ratios of labeled and unlabeled data _(e.g., FixMatch-based federated learning approaches)._
		
    - New Approach: Everyone has access to a public dataset, and the predictions of a client model or server model on this dataset can be transmitted. (FedDS only additional data on server FedDF is similar, FedAD, FedMD, Cronus)
        
3. **Benchmarking / Evaluation:**
    
    - **Ablation study** _(which components improve performance the most, AUC)._
        
    - **Comparisons with state-of-the-art models** _(supervised, semi-supervised, or fully federated approaches)._
        
    - **Non-IID levels** _(use Dirichlet distribution with various alphas)._
        
    - **Impact of different FL setups** _(e.g., client participation, number of communication rounds, amount of labelled data, data per client)._
        
4. **Domain of Application:**
    
    - ECG: PTB-XL, MIT-BIH _(multi-class classification, rhythm detection)._
        
    - Medical Imaging _(CheXpert, HAM10000)._
	
    - Many more
        
5. **Focus:**
    
    - **Aggregation Methods:**
        
        - Model distillation approaches: RSCFed _(relies on consensus-driven distillation)._
            
        - Sending model predictions instead of weights _(improves privacy and tackles non-IID, knowledge distillation)._

        - Sending weights: FedAvg
            
    - **Local Training-Oriented Methods:**
        
        - SSL or Self-SL: Contrastive learning-based methods _(e.g., FedCL, FedU, FedMatch)._
            
6. **Model Layout:**
    
    - CNNs, ResNet, Vision Transformers, _(depending on application domain)._
        
7. **Privacy & Security:**
    
    - Homomorphic encryption _(adds security but increases computation)._
        
    - Differential privacy _(noise injection for privacy but may reduce accuracy)._
        
8. **Heterogeneity & Non-IID Data:**
    
    - **Heterogeneity** _(broader term that includes non-IIDness and system-level issues):_
        
        - Client heterogeneity _(specialized hospitals, different training distributions)._
            
        - Device heterogeneity _(varying compute power and memory)._
            
        - Server-client model variations _(e.g., split learning, hierarchical FL)._
            
    - **Non-IID Data:** _(a type of heterogeneity referring to data distribution differences between clients.)_
        
        - Dirichlet distribution _(α controls non-IID level)._
		
        - Compile many datasets
            
        - FedCVD _(specific framework for handling non-IID vision tasks)._
            
        - Feature-based non-IID _(different clients collect different types of features)._
            
        - Label distribution skew _(some clients only have certain labels)._
            
9. **Handling Unlabeled Data:**
    
    - **Semi-supervised Learning:**
        
        - Pseudo-labeling _(assigning labels to unlabeled data and retraining)._
            
        - Confidence thresholding _(only keeping high-confidence pseudo-labels)._
		        - Dynamic thresholds _(adjusting threshold based on client performance)._
            
    - **Self-supervised Learning:**
        
        - Contrastive learning _(instance-based, e.g., FedU)._
            
        - Student-teacher models _(FedMatch, FedSEAL)._ --> WRONG
            
        - Context-based approaches _(leveraging dependencies in sequential data)._
            
        - Temporal-based learning _(applicable in video, time-series, e.g., FedST)._
            
    - **Transfer Learning:** _(pretrained models adapted to federated settings)._
        
    - **Unsupervised Learning:**
        
        - FedCA _(Federated clustering for anomaly detection)._
            
        - FedX _(uses knowledge distillation but in a decentralized way)._
	
    - **Supervised Learning:**


For classification tasks, CIFAR-10/100 dataset [55]
(MIT license), ImageNet dataset [56], and TinyImageNet
dataset [57] are used.

---

Regarding data heterogeneity, for instance, larger institutions have more detailed patient data than small institutions due to diverse patient populations and
disease manifestation (i.e., statistical heterogeneity); some institutions
collect and annotate interest organs due to the various interests in
clinical studies (i.e., label heterogeneity). Each
LLM, foundational model that we can update using local models

Approach:
Server has publically available and labelled data of the use-case (future proof)
And clienst have their own data

Pretrain global model on public dataset multiple times
Then distribute (reduces # comm rounds)
Then the local models have to help with generalization/personalization

Do we say clients do SSL and server supervised? or both SSL? seems weak though. Most approaches assume 1 Supervised atleast.


### **Difference Between Data Heterogeneity & Non-IID Data**

- **Data Heterogeneity** is a broader term that includes different aspects:
    
    - **Statistical heterogeneity:** Non-IID data (imbalanced labels, feature shifts).
        
    - **System heterogeneity:** Differences in compute power, memory, or network conditions.
        
    - **Model heterogeneity:** Different architectures per client (e.g., edge devices vs. cloud).
        
- **Non-IID Data** is a subset of heterogeneity:
    
    - **Label distribution skew:** Clients have different label proportions.
        
    - **Feature distribution shift:** Clients collect data from different sensors or conditions.
        
    - **Concept shift:** Different regions or environments affect data generation.
        

