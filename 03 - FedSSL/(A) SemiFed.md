
#FedSSL 

In this work, we focus
on a new scenario for cross-silo federated learning, where
data samples of each client are partially labeled.

We propose a new framework dubbed SemiFed that unies
two dominant approaches for semi-supervised learning: con-
sistency regularization and pseudo-labeling.
SemiFed first applies advanced data augmentation techniques to enforce
consistency regularization and then generates pseudo-labels using the model's predictions during training. SemiFed takes advantage of the federation so that for a given image, **the pseudo-label holds only if multiple models from different clients produce a high-condence prediction and agree on the same label**.


They only perturb once for consistency regularization. and try to minimize distance using KL distance.
& only 1 perturbation for pseudolabeling

They dont perform pseudolabeling every step and rather do it every X steps, and this allows them to add unlabeled samples to the labeled dataset.
Their main innovation is the agreement of K models to allow it to be added to labeled dataset.

**Loss on Pseudo-Labels:** Once an unlabeled data sample meets the stringent criteria for pseudo-label assignment (high confidence from multiple models and multi-model agreement), it is assigned a pseudo-label and is then **transferred from the client's unlabeled dataset (Dk u) to its labeled dataset (Dk l)**. From that point onwards, the local model is trained employing a **standard cross-entropy loss** on these newly generated pseudo-labels, treating them equally with the original true labels. This means that the `lks` (supervised loss), which is a standard cross-entropy loss, is applied to these pseudo-labeled samples.



SemiFed uses **two complementary mechanisms** to leverage unlabeled data, applied at **different times and for different purposes**:

---

### 1. **Consistency Regularization (Continuous during local training)**

- **Goal:** Make model predictions stable under small input perturbations, leading to smoother decision boundaries.
    
- **How:**
    
    - In every local epoch, each unlabeled sample `x` is augmented into `x̃` (e.g. with RandAugment).
        
    - The model’s predictions on `x` and `x̃` are compared using KL divergence to compute an **unsupervised consistency loss (`lku`)**.
        
- **Integration:** This loss is **continuously combined with the supervised loss** from labeled data during local training.
    

---

### 2. **Pseudo-Labeling (Occasional at specific communication rounds)**

- **Goal:** Convert reliable unlabeled samples into labeled ones to expand the training set.
    
- **How:**
    
    - At specific rounds (`Tp`), each client collects predictions on its unlabeled data from **its own model, the global model, and all other clients’ models**.
        
    - A sample gets a pseudo-label only if it meets **two strict conditions**:
        
        1. **High confidence:** All models’ top predicted probability ≥ threshold `γt`.
            
        2. **Model agreement:** At least `u` of the `K+1` models agree on the same class.
            
    - If both conditions are met, the sample is **moved from unlabeled to labeled data** and trained on as if it had a true label.
        

---

🔑 **Key idea:**

- **Consistency regularization** improves general robustness from all unlabeled data.
    
- **Pseudo-labeling** selectively converts **high-confidence, cross-model-agreed** samples into new labeled data.
    
- Together, they let SemiFed **gradually expand its labeled set while keeping training stable and reliable**.
    



![[SemiFed.pdf]]