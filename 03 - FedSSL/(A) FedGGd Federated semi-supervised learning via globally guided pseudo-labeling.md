#FedSSL #Code

Can be quoted for the applications.
This paper not only gives a good background for the other work in SSL/FSSL but just has a nice structure.

**This paper focuses on labeled on partial client: "Majority is unlabeled, some are fully labeled"**

https://github.com/whatement/FedGGp/blob/main/main.py

alpha = 0.3, which is apparently standard.
"Following (Li et al., 2023), we configur
10
clients, all of which par
ticipate
in each communication round"

500 round limit!!
the **labeling rate for partially labeled clients is set to 0.05**

This paper has FedAvg+Fixmatch!! Look at their impl.


| Class group        | Threshold | Weight     | Contribution strength |
| ------------------ | --------- | ---------- | --------------------- |
| **A (head)**       | High      | ≈ 1 (hard) | Strong, selective     |
| **D (tail)**       | Low       | ≈ 1 (hard) | Strong, frequent      |
| **Am (ambiguous)** | Medium    | < 1 (soft) | Weak, stabilizing     |
## 3. How are probabilities aggregated on the client?

Each client builds a **class-wise accumulation vector** Qk∈RCQ_k \in \mathbb{R}^CQk​∈RC.

For every unlabeled sample xxx that passes the threshold:

#### $Q_k[c] += w_x * p_c(x)$ 

So:

- Head/tail-class samples (hard weight ≈ 1) contribute almost fully
    
- Ambiguous-class samples (soft weight < 1) contribute fractionally
    
- Rejected samples contribute nothing
    

This accumulation is done **over all local unlabeled samples** (or over mini-batches, equivalently).

FedGGp (Federated Globally Guided pseudo-labeling) is a federated semi-supervised learning method designed for label-scarce and non-IID settings. Its main features are:

- **Global prediction assessment:** It builds a global prediction vector (Φ) on the server to measure class-wise prediction bias across all clients, instead of relying on each client's biased local predictions.
    
- **Adaptive thresholds:** Based on this global bias assessment, classes are split into _advantageous, disadvantageous,_ and _ambiguous_ groups, and each group gets its own adaptive confidence threshold for selecting pseudo-labels.
    
- **Soft consistency regularization:** For ambiguous (underrepresented) classes, it relaxes the usual strict consistency check and allows partially consistent predictions (weighted continuously), so that more samples from rare classes can be used.
    
- **Bias correction:** Uses Bayesian-inspired rebalancing factors to correct class imbalance when calculating global prediction probabilities.
    

Together, these steps create more **balanced and reliable pseudo-labels** across classes and clients, improving FSSL training despite label scarcity and heterogeneity



NotebookLM:
FedGGp is a method used in Federated Semi-Supervised Learning (FSSL) that allows models to be trained collaboratively using both labeled and unlabeled datasets, even when labels are scarce and data is unevenly distributed among clients.

The pipeline generally follows the conventional Federated Averaging (FedAvg) workflow and operates in rounds, involving a central server and multiple participating clients:

1. **Server Broadcasts and Tracks Bias:**
    
    - The central server starts by initialising model parameters and a "global prediction evaluation vector" (Φ), which is initially uniform.
    - In each round, the server broadcasts the current global model parameters and the Φ vector to all active clients. The Φ vector helps estimate the overall prediction bias across different classes.
2. **Clients Perform Local Training with Guided Pseudo-labeling:**
    
    - Each client receives the global model and the Φ vector.
    - Clients then classify their classes into three groups based on the Φ vector: **advantageous**, **disadvantageous**, and **ambiguous** classes, according to their prediction probabilities.
    - For **labeled data**, clients use standard supervised learning.
    - For **unlabeled data**, clients use a specially designed **globally guided pseudo-labeling strategy** to generate predicted labels:
        - They apply **adaptive thresholds**, which vary for each class category, to generate high-quality pseudo-labels. Stricter filtering is used for advantageous classes, while more pseudo-labels are sought for discriminated classes.
        - For ambiguous (underrepresented) classes, a **soft consistency regularization mechanism** is employed. This helps recover low-confidence pseudo-labels near decision boundaries, ensuring these classes contribute to balanced model training.
    - After local training, each client constructs a pseudo-label set and collects prediction probabilities for these samples, accumulating them into a vector (Q).
    - Clients then send their updated local model parameters and their accumulated prediction vector (Q) back to the server.
3. **Server Aggregates and Updates Bias:**
    
    - The server aggregates the updated model parameters received from all clients to form a new global model.
    - Crucially, the server updates its global prediction evaluation vector (Φ) using the accumulated prediction vectors (Q) from the clients. This update process involves a rebalancing factor (alpha) to calibrate class-wise confidence and mitigate prior bias.

This process iteratively helps to generate balanced pseudo-labels, addressing prediction bias across classes even in scenarios with diverse data distributions and limited labels.


FedGGp's approach to training with unlabeled data is a sophisticated **globally guided pseudo-labeling strategy** designed to overcome challenges like **prediction bias** and **data heterogeneity** prevalent in Federated Semi-Supervised Learning (FSSL).

Here's a detailed breakdown of the unlabeled data training pipeline:

1. **Initial Pseudo-label Generation**:
    
    - Clients first receive the global model parameters (𝜃) and the global prediction evaluation vector (Φ) from the server.
    - For their local unlabeled data (`D_u`), clients apply common **data augmentation strategies** to create augmented samples (`x_aug`).
    - The model then predicts labels for these augmented samples, with the **pseudo-label (`ŷ`) being the class with the maximum predicted probability** (`argmax p_𝜃(y|x_aug)`). The corresponding confidence is `p̂ = max p_𝜃(y|x_aug)`.
2. **Global Prediction Assessment for Bias Classification**:
    
    - To counter local prediction biases, FedGGp incorporates a **global perspective** using the `Φ` vector. This `Φ` vector, maintained and updated by the server, estimates class prediction bias.
    - On the client side, the received `Φ` vector is used to dynamically **partition the classes (`C`) into three distinct groups** based on their prediction probabilities relative to a uniform baseline and an "Imbalance Factor" (IF) boundary:
        - **Advantageous classes (A)**: These classes have prediction probabilities exceeding the uniform prior baseline and typically receive the majority of pseudo-labels due to high-confidence outputs.
        - **Disadvantageous classes (D)**: These classes have biased prediction probabilities within the IF imbalance boundary, allowing for partial sample inclusion in pseudo-labeling.
        - **Ambiguous classes (Am)**: These are classes with severely suppressed prediction probabilities below the IF imbalance boundary, leading to most samples being initially excluded from pseudo-labeling due to classification confusion.
3. **Hierarchical Pseudo-labeling Strategy**:
    
    - FedGGp then applies a **hierarchical pseudo-labeling strategy** that combines globally guided adaptive thresholds with a soft consistency regularization mechanism, tailored to each class category.
    - **Globally Guided Adaptive Thresholds**: Instead of a fixed threshold, FedGGp employs **adaptive thresholds (`τ`)** that are specific to each class category to derive high-confidence pseudo-labels:
        - For **advantageous classes (A)**, a **strict filtering** approach is used with an adaptive threshold `τ_A = (Φ(c)/Φ_max) * E_B(p̂)`, where `E_B(p̂)` is the expected maximum prediction confidence in the current batch.
        - For **disadvantageous classes (D)**, the adaptive threshold is reduced to `τ_D = (Φ(c)/Φ_max) * (E_B(p̂) - σ_B(p̂))` to capture more pseudo-label predictions, where `σ_B(p̂)` is the standard deviation of confidence scores.
        - For **ambiguous classes (Am)**, the **lowest classification threshold** is set at `τ_Am = 1/|C|`. This aims to broaden the boundary of pseudo-labels for these underrepresented classes.
    - **Soft Consistency Regularization**:
        - Traditional consistency regularization (C.R.) typically uses a one-hot encoded vector (`𝟙(y1 = y2)`) to enforce prediction consistency across different perturbations of the same input, filtering out samples with unstable predictions. However, this often excludes most ambiguous samples, exacerbating unfairness for underrepresented classes.
        - FedGGp proposes **soft consistency regularization (`R_s`)** to **recover samples from underrepresented classes (Am)** that would normally be discarded. This method extends the one-hot encoded vector to a **continuous form**, assigning weights based on classification consistency confidence, formulated as `R_s := (p_2c←y1 * p_2c←y2)`.
        - To control the quality of recovered samples, a **Top-k mechanism** is introduced as a secondary constraint, filtering predictions with lower confidence.
        - For advantageous (A) and disadvantageous (D) classes, traditional consistency regularization (`R_h(y1) := 𝟙(y1 = y2)`) is retained.
4. **Pseudo-label Weighting Function (`Wggp`)**:
    
    - These components are combined into a pseudo-label weighting function, `Wggp(x,Φ)`, which determines whether a sample's prediction should be utilized as a pseudo-label and with what weight:
        - For `y1 ∈ A`: `Wggp(x,Φ) = 𝟙(p̂1 > τ_A) ⋅ R_h(y1)`
        - For `y1 ∈ D`: `Wggp(x,Φ) = 𝟙(p̂1 > τ_D) ⋅ R_h(y1)`
        - For `y1 ∈ Am`: `Wggp(x,Φ) = 𝟙(p̂1 > τ_Am) ⋅ R_s(y1)`
5. **Local Optimization and Pseudo-label Set Construction**:
    
    - During local training, for each mini-batch of unlabeled data, clients predict pseudo-labels using the `Wggp` function.
    - The final training objective for client `k` then minimizes a function that includes a term for unlabeled data, where `L_u` (typically cross-entropy loss) is weighted by `Wggp(x,Φ)`. This allows the model to learn from these pseudo-labeled samples.
    - After local epochs, each client constructs a pseudo-label set (`D_p`) based on the `Wggp` function and collects the prediction probabilities (`p1`) for samples within this set, stacking them into a vector `Q_k`. This `Q_k` vector is then sent back to the server, representing the accumulated prediction probability for each class, specifically designed to minimize communication costs.

This comprehensive strategy ensures that pseudo-labels are generated in a balanced manner across classes, mitigating prediction bias and allowing effective learning even in label-scarce and heterogeneous environments.


![[Federated semi-supervised learning via globally guided pseudo-labeling.pdf]]