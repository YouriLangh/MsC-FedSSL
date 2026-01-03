# DDRFed (2025)

DDRFed improves federated semi-supervised learning by first reducing data skew with synthetic global data, then using adaptive class-aware thresholds, and finally extracting useful information even from low-confidence unlabeled samples.
They study **Federated Semi-Supervised Learning (FSSL)**, where:

- Data is split across many clients (phones, hospitals, companies, etc.)
    
- **Most data is unlabeled**
    
- Data across clients is **non-IID** (each client sees different classes or proportions)
    

This combination causes two big problems:

1. **Too little labeled data** → pseudo-labels are unreliable
    
2. **Non-IID data** → clients learn very different models that don’t aggregate well
They assume:

- A **small number of fully labeled clients**
    
- A **large number of fully unlabeled clients**

### 1. Fix non-IID data using a diffusion model (before training starts)

**Idea:**  
If clients all see very different data, first give them something _globally representative_.

**What they do:**

- Each client trains a **diffusion model** on its local data
    
- The server aggregates these diffusion models (using FedAvg)
    
- The server then **generates synthetic images** that reflect the **global data distribution**
    
- These synthetic samples are **evenly distributed to all clients**
    

**Why this helps:**

- Clients now share a small, privacy-safe dataset with similar class coverage
    
- This reduces extreme data skew without sharing real data
    

Think of it as:

> “Everyone gets a small, shared reference dataset so they’re no longer completely isolated.”

1-s2.0-S0167739X25001153-main

---

### 2. Use **dynamic, class-balanced confidence thresholds**

Most SSL methods say:

> “Only trust predictions above confidence 0.95.”

This breaks badly in FSSL because:

- Some classes are rare
    
- Early predictions are weak
    
- Minority classes never get pseudo-labels
    

**What they do instead:**

- The server tracks the **global class distribution**
    
- Each class gets its **own threshold**
    
- Thresholds:
    
    - Start **low** early in training
        
    - Increase as the model becomes more confident
        
    - Are **lower for rare / unstable classes**
        
    - Are **higher for common / stable classes**
        

**Result:**

- More samples are used early
    
- Minority classes are not ignored
    
- Pseudo-labels stay balanced
    

1-s2.0-S0167739X25001153-main

---

### 3. Learn even from _low-confidence_ samples (key novelty)

Normally:

- Low-confidence samples are discarded
    

Their idea:

> Even if we don’t know what class a sample _is_, we may know what classes it is _not_.

**How it works:**

- For low-confidence samples:
    
    - Apply two different augmentations
        
    - Check which classes are **consistently unlikely**
        
- They then assign **negative labels**:
    
    - “This sample is _not_ class A, B, C…”
        
- This suppresses wrong classes and sharpens decision boundaries
    

This is called **Residual Class Negative Learning (RNL)**.

**Why it matters:**

- All unlabeled data contributes to learning
    
- The model does not overfit to only “easy” samples
    
- Helps late-stage training and generalization

Their training happens in **phases**:

1. **Diffusion warm-up phase**
    
    - 40 federated rounds
        
    - Only training the diffusion model
        
2. **Classifier training**
    
    - 200 rounds using only labeled clients (stabilization)
        
    - 500 rounds of full federated semi-supervised training
        

Datasets:

- **CIFAR-10**
    
- **Fashion-MNIST**
    

Non-IID splits using Dirichlet distributions (γ = 0.1 and 0.8)


## What results they get (and how good they are)

### Performance gains (accuracy)

Compared to prior FSSL methods, DDRFed achieves:

- **+6.93% accuracy on CIFAR-10**
    
- **+2.17% accuracy on Fashion-MNIST**
    

This is under **severe non-IID** and **few labeled clients**.

It consistently outperforms:

- Fed-Consist
    
- RSCFed
    
- CBAFed
    
- PDCFed
    

And it closes much of the gap toward fully supervised upper bounds.


# FedCR (2025)
Federated semi-supervised learning with contrastive representations against
noisy labels

Two realistic problems are combined:

1. **Labels are only available on the server**, not on clients
    
2. **Those server labels are noisy** (incorrect annotations)
    

This is a very difficult setting, because:

- Clients cannot correct mistakes (they have no labels)
    
- Standard losses (cross-entropy, pseudo-labeling) tend to **memorize label noise**
    
- Errors introduced at the server propagate to all client

> **If labels are noisy, do not rely on them directly. Instead, learn representations that are robust to noise, and only weakly guide them with labels.**

They therefore **avoid pseudo-labeling entirely** and instead use **contrastive representation learning** as the backbone.

### 1. Learn noise-robust representations using contrastive learning

Instead of training directly on labels, they:

- Learn **feature representations** using **contrastive learning**
    
- The idea:  
    _different augmented views of the same image should be close in feature space_
    

They use a **BYOL-style Siamese architecture**:

- **Online network** (shared + updated globally)
    
- **Target network** (EMA-updated, never shared)
    

This avoids:

- Negative pairs
    
- Representation collapse
    
- Overfitting to wrong labels
    

Because contrastive learning does not require labels, it naturally resists label noise

1-s2.0-S156849462500732X-main

.

---

### 2. Server-side training with a noise-robust loss

On the **server**, labeled data _is_ used — but cautiously.

They combine:

- **Contrastive regularization**
    
- **Generalized Cross-Entropy (GCE)** instead of standard cross-entropy
    

Why GCE?

- Cross-entropy heavily penalizes wrong predictions → memorizes noisy labels
    
- GCE interpolates between:
    
    - Cross-entropy (fast convergence)
        
    - Mean Absolute Error (noise-robust)
        

This makes the server model:

- Less sensitive to mislabeled samples
    
- More stable early in training
    
    1-s2.0-S156849462500732X-main
    
    .
    

---

### 3. Clients only do unsupervised contrastive learning

On each federated round:

1. Server sends **online network parameters**
    
2. Clients:
    
    - Train **only contrastive loss**
        
    - Use _no labels_
        
3. Clients upload **only the online network**
    
4. Server aggregates (FedAvg)
    
5. Server **re-anchors the model using noisy labeled data**

Their training proceeds as follows:

1. **Server pre-training**
    
    - Hundreds of rounds (≈700)
        
    - Uses noisy labels + contrastive regularization
        
2. **Federated training**
    
    - 200 global rounds
        
    - Each round:
        
        - 5 clients selected
            
        - Clients train contrastive loss
            
        - Server aggregates and retrains
            

This staged approach ensures:

- A strong initial representation
    
- Stable convergence under noise

100 clients, 5 active


# FedMatch (2021)

This paper introduces **Federated Semi-Supervised Learning (FSSL)** as a _practical_ federated learning setting where **most client data is unlabeled**.

The authors identify two realistic issues with existing FL:

1. Clients often **cannot or do not label their data**
    
2. Naively combining FL with SSL methods (e.g. FixMatch + FedAvg) **does not work well**, especially under non-IID data and disjoint supervision
    

They argue that **label scarcity is the norm, not the exception**, in on-device and privacy-sensitive applications

2006.12097v3

.

---

## What situation they assume (very important)

Unlike later FSSL papers, **FedMatch explicitly studies two scenarios**:

### 1. Labels-at-Client

- Each client has:
    
    - A **small fraction of labeled data** (e.g. 5%)
        
    - A large amount of unlabeled data
        
- Server only aggregates models
    

### 2. Labels-at-Server (harder case)

- **All labeled data is only available at the server**
    
- Clients have **only unlabeled data**
    
- Clients never compute supervised losses
    

This second scenario is a **key contribution** of the paper and is illustrated in Figure 1

Their key insight is:

> **Supervised and unsupervised learning interfere with each other in federated settings, especially when labels are disjoint.**

FedMatch solves this by:

1. **Aligning clients through inter-client consistency**
    
2. **Separating supervised and unsupervised learning parameters**


### 1. Inter-client consistency loss (model-to-model agreement)

Instead of only enforcing consistency:

- between _augmentations of the same image_ (as in FixMatch)
    

They also enforce consistency:

- **between predictions of different clients**
    

How:

- Server selects **H “helper” models** that are most similar to a client
    
- Similarity is measured using model outputs on a shared dummy input
    
- The client is trained to:
    
    - Agree with its own prediction
        
    - Agree with predictions of helper clients
        

Only predictions where **multiple models agree** are turned into pseudo-labels

This reduces:

- Noisy pseudo-labels
    
- Client drift
    
- Overfitting to local bias
    

(See Figure 2 and Section 3.1)

2006.12097v3

#### Server selects helper models

For each client:

- Server finds the **H nearest models** in embedding space
    
- These are the **helper models**
    
- Helper models are:
    
    - Frozen
        
    - Not updated during the client’s training
        
    - Sent as reference models only
        

Think of helpers as:

> “Other clients whose models currently think similarly to you.”

Inter-client consistency loss = “only trust an unlabeled sample if _several different clients’ models agree on it_, and then train the local model to match that agreed prediction.”


---

### 2. Parameter decomposition for disjoint learning

This is the **most important technical idea**.

Instead of one parameter set, the model is split into:

- **σ (sigma)** → supervised learning
    
- **ψ (psi)** → unsupervised learning
    

So:

θ=σ+ψ\theta = \sigma + \psiθ=σ+ψ

Training:

- Labeled data updates **σ only**
    
- Unlabeled data updates **ψ only**
    
- Regularization keeps ψ sparse and close to σ
    

Why this matters:

- Prevents **catastrophic forgetting**
    
- Makes **labels-at-server possible**
    
- Reduces communication (ψ is sparse)
    

This separation is critical in the labels-at-server scenario, where clients never see labels (Section 3.2)

2006.12097v3

.

## How training proceeds (when things happen)

Each communication round:

### Labels-at-Client

1. Clients train:
    
    - σ on labeled data
        
    - ψ on unlabeled data
        
2. Server aggregates both
    
3. Server selects helper models
    
4. Repeat
    

### Labels-at-Server

1. Server trains σ on labeled data
    
2. Clients train ψ on unlabeled data only
    
3. Server aggregates ψ
    
4. Server sends updated σ and helper ψ models
    

Algorithms 1 & 2 describe this precisely

2006.12097v3

.
100 clients (batch),

### Main findings

1. **Naive FL + FixMatch / UDA fails badly**
    
    - Especially in labels-at-server
        
    - Performance collapses over rounds
        
2. **FedMatch is consistently better**
    
    - +5–15 pp over naive FSSL
        
    - Stable convergence
        
    - Robust under non-IID and streaming data
Resnet-9...

#### Labels at client
### How training works (per round)

1. **Server broadcasts**
    
    - Global σ (supervised parameters)
        
    - Global ψ (unsupervised parameters)
        
    - Helper models (ψ from similar clients)
        
2. **Each client trains locally**
    
    - On labeled data → update **σ**
        
    - On unlabeled data → update **ψ**
        
    - Inter-client consistency applies on unlabeled data
        
3. **Client sends back**
    
    - Updated σ and ψ (as sparse deltas)
        
4. **Server aggregates**
    
    - σ and ψ independently
        
    - Updates model embeddings
        
    - Selects new helpers


#### Labels at server
### How training works (per round)

1. **Server trains first**
    
    - Uses labeled data
        
    - Updates **σ only**
        
    - ψ is frozen
        
2. **Server broadcasts**
    
    - Updated σ
        
    - Current ψ
        
    - Helper models (ψ from similar clients)
        
3. **Each client trains**
    
    - Only on unlabeled data
        
    - Updates **ψ only**
        
    - Uses inter-client consistency
        
4. **Clients send back**
    
    - Only ψ updates
        
    - No σ updates (clients don’t touch σ)
        
5. **Server aggregates ψ**
    
    - Combines unsupervised knowledge
        
    - Keeps σ under server control

# Improving Semi-supervised Federated Learning by Reducing the Gradient Diversity of Models (2021)

## 1. What problem does this paper study?

The paper studies **semi-supervised federated learning (SSFL)** in a **realistic but difficult setting**:

- **Server**: has a _small labeled dataset_
    
- **Clients**: have _only unlabeled data_
    
- **Data is non-IID across clients**
    
- **No data is shared**
    

This is harder than standard FL and harder than centralized SSL.

The authors identify **one main bottleneck** that explains why many SSFL methods fail:

> **Large gradient diversity between clients makes FedAvg unreliable.**

The paper shows that **high gradient diversity directly correlates with poor accuracy**.


## 3. What method do they propose?

They **do not invent a new SSL loss**.

Instead, they improve SSFL by fixing **three practical design choices**:

### (1) Training objective: Consistency Regularization (CRL)

Clients use **FixMatch-style consistency regularization**:

- Weak augmentation → pseudo-label
    
- Strong augmentation → supervised by pseudo-label
    
- Only high-confidence predictions are used
    

This already works _reasonably well_ but still produces **high gradient diversity**.

### (2) Normalization: GroupNorm instead of BatchNorm

Key insight:

- **BatchNorm depends on batch statistics**
    
- Different clients → different statistics → unstable gradients
    

They replace:

- ❌ BatchNorm
    
- ✅ **GroupNorm (GN)**
    

Result:

- Lower gradient diversity
    
- More stable aggregation
    
- Higher accuracy
    

This alone gives a **large performance boost**.

---

### (3) Aggregation: Grouping-based model averaging (instead of FedAvg)

This is their **main contribution**.

Instead of:

- Averaging all client models at once (FedAvg)
    

They:

1. Randomly split participating clients into **groups**
    
2. Average models **within each group**
    
3. Average the group averages
    

This reduces the effect of **extreme or conflicting updates**.

Think of it as:

> “Smooth the updates _before_ globally averaging them.”

This significantly reduces gradient diversity when:

- Many clients participate
    
- Data is highly non-IID
    

## 4. Do clients agree on predictions or share models?

**No inter-client agreement loss** like FedMatch.

Important distinction:

- Each client trains **independently**
    
- There is **no prediction agreement**
    
- No helper models
    
- No ensemble voting
    

This paper is **purely optimization-focused**, not consensus-focused.


Odd hyperparams, they do cover alot of fields though, iid and not.
R= alpha kinda
K = clients
C= communicatng clients
N_s = labeled data
T= # comm rounds

# FedIRM (2021)

This paper studies **federated semi-supervised learning (FSSL)** in a **medical imaging** context, where:

- **Only a few hospitals have labeled data**
    
- **Most hospitals have only unlabeled data**
    
- **Data cannot be shared** due to privacy regulations
    
- Models trained only with consistency regularization at unlabeled clients tend to **forget the task**
    

The key observation is:

> **Unlabeled clients trained with consistency regularization alone gradually drift away from the actual diagnostic task.**

This is a _task-knowledge loss_ problem, not just an optimization issue.

They assume a **mixed-client FSSL setting**:

- **Labeled clients**: a small subset of hospitals with annotations
    
- **Unlabeled clients**: the majority, with no labels
    
- All clients participate in FedAvg

## Core idea (intuition)

Their key idea is:

> **Even if hospitals do not share data or labels, they can share _class relationships_ learned from labeled data.**

Instead of transferring:

- Labels
    
- Predictions
    
- Pseudo-labels
    

They transfer **disease relationship structure**.


### 2. Extract disease relations at labeled clients

At each labeled client:

- The model extracts **pre-softmax feature vectors**
    
- For each disease class:
    
    - Compute a **mean feature vector**
        
- Convert these into **soft class distributions**
    
- Stack them into a **disease relation matrix**
    

This matrix captures:

- Which diseases are commonly confused
    
- Structural diagnostic knowledge
    

This information is **client-independent**.

---

### 3. Estimate disease relations at unlabeled clients (carefully)

Unlabeled clients:

- Generate pseudo-labels
    
- Use **uncertainty filtering** (via Monte Carlo dropout)
    
- Only confident predictions are used
    
- Compute a corresponding relation matrix
    

This avoids reinforcing incorrect relations.

---

### 4. Inter-client Relation Matching (IRM)

The server:

- Aggregates relation matrices from labeled clients
    
- Broadcasts the **average relation matrix**
    

Each unlabeled client:

- Aligns its own relation matrix to the labeled one
    
- Minimizes **KL divergence** between the two
    

This regularization:

- Preserves task structure
    
- Prevents semantic drift
    
- Guides unlabeled learning without labels
    

---

## How training proceeds

Each federated round:

1. Server sends global model
    
2. Labeled clients train with supervision
    
3. Unlabeled clients train with:
    
    - Consistency loss
        
    - **IRM loss**
        
4. Server aggregates parameters (FedAvg)
    
5. Server updates and re-sends disease relations
    

No data, features, or samples are ever shared.

### Experimental setting

- 10 total clients
    
- 2 labeled + 8 unlabeled
    
- 20% labeled data overall


# SemiFed (2021)
This paper studies **semi-supervised federated learning (SSFL)** in a **cross-silo** setting where:

- **Each client has a small amount of labeled data**
    
- **Most local data is unlabeled**
    
- **Data can be non-IID across clients**
    
- The server **has no data**
    

The authors focus on a practical issue:

> Naively applying centralized SSL methods in FL can degrade performance when labeled and unlabeled data come from mismatched class distributions.


Non-IID is constructed using **Dirichlet splits (α = 0.5)** across 10 clients, producing moderate label skew (Figure 2).

## Core idea (high-level intuition)

Their key idea is:

> **Use the federation itself to make pseudo-labels safer.**

Instead of trusting:

- A single client’s prediction (standard pseudo-labeling),
    

they require:

- **Multiple models (local + global) to agree** before assigning a pseudo-label.
    

This is meant to reduce:

- Confirmation bias
    
- Error propagation from incorrect pseudo-labels

### 1. Consistency regularization (baseline)

Each client trains using:

- Supervised cross-entropy on labeled data
    
- Consistency loss on unlabeled data:
    
    - Original image vs. strongly augmented version
        
    - KL divergence between predictions
        

They use **RandAugment** for strong perturbations.

This part is essentially **UDA/FixMatch-style consistency training**, applied locally.

### 2. Federated pseudo-labeling via model agreement

After a warm-up period:

1. The server sends:
    
    - The global model
        
    - All participating clients’ models
        
2. For each unlabeled sample:
    
    - It is evaluated by **K + 1 models**
        
    - A pseudo-label is assigned only if:
        
        - Prediction confidence ≥ γ
            
        - At least **u models agree** on the same class
            
3. Accepted pseudo-labels are:
    
    - Added permanently to the labeled set
        
    - Trained using standard cross-entropy
        

This agreement-based filtering is the **only federated-specific innovation**.

## Does SemiFed address non-IID data?

### **Yes — but only weakly and indirectly.**

What it _does_ address:

- Reduces **noisy pseudo-labels** under mild label skew
    
- Prevents some catastrophic self-training errors
    

What it _does not_ address:

- Client drift due to heterogeneous gradients
    
- Missing-class clients
    
- Severe Dirichlet skew (α ≪ 0.5)
    
- Representation misalignment across clients
    

FedAvg is used unchanged.

---

## What results they report

### Datasets

- CIFAR-10
    
- SVHN
    

### Key findings

- SemiFed consistently outperforms:
    
    - Supervised-only FL
        
    - VAT
        
    - UDA-consistency
        
- Gains are:
    
    - ~0.7–4.4% on CIFAR-10
        
    - Smaller on SVHN (easier dataset)
        

Under non-IID:

- Improvements remain
    
- But performance gap to fully supervised remains large
    

(Table 1, Table 2)

> Lin et al. propose SemiFed, a semi-supervised federated learning framework that combines consistency regularization with agreement-based pseudo-labeling. By requiring multiple client models and the global model to agree before assigning pseudo-labels, the method reduces confirmation bias when exploiting unlabeled data. While SemiFed improves over consistency-only baselines under both IID and moderately non-IID data partitions, it relies on standard FedAvg and does not explicitly address severe statistical heterogeneity or client drift.

# Huang
More precisely:

- **No semi-supervised learning**
    
    - The method **does not mix labeled and unlabeled samples within the same objective** (no pseudo-labels, no FixMatch-style thresholds).
        
    - Unlabeled data is **never converted into labels**.
        
- **Yes, self-supervised (representation-level)**
    
    - Unlabeled _public_ data is used to **align representations across heterogeneous clients** via a self-supervised objective (cross-correlation on logits).
        
    - This aligns closely with **Barlow Twins / information-bottleneck–style self-supervision**, but applied _across clients_ rather than views of the same image.
        
- **Supervised locally**
    
    - Each client still performs **standard supervised training** on its own private labeled dataset.
        

So the clean categorization is:

> **A federated learning method that uses self-supervised objectives for inter-client communication and supervised learning for local optimization.**

This is exactly how the authors motivate it in the abstract and Section 3

Huang_Learn_From_Others_and_Be_…

.

---

## What setting does the paper assume?

### 1. Data setting

- **Fully labeled private client data**
    
    - Every client has labels for its own dataset.
        
- **Unlabeled public data**
    
    - Shared across clients
        
    - Used _only_ for communication / representation alignment
        
    - No labels required, no annotation assumptions
        

There is **no label scarcity assumption**, unlike SemiFed, FixMatch-FL, or FSSL papers.

# CBAFED (2023)

CBAFed addresses **federated semi-supervised learning (FSSL)** under a **highly asymmetric client setup**:

- **Few labeled clients** (often only one)
    
- **Many fully unlabeled clients**
    
- **Non-IID data distributions** across clients
    
- Severe **class imbalance** between labeled and unlabeled data
    

The authors argue that existing FSSL methods fail mainly because **standard pseudo-labeling is ill-suited to non-IID federated settings**, leading to:

1. Catastrophic forgetting at unlabeled clients
    
2. Over-dominance of head classes
    
3. Noisy pseudo-labels under skewed distributions
    

CBAFed re-designs pseudo-labeling specifically for FSSL rather than borrowing it directly from centralized SSL


CBAFed consists of **four tightly coupled components**:

1. **Warm-up on labeled clients**
    
2. **Fixed pseudo-labeling on unlabeled clients**
    
3. **Class-balanced adaptive thresholds**
    
4. **Residual weight connections during training and aggregation**


## 1. Warm-up stage (labeled clients only)

Training begins with a **purely supervised warm-up**:

- Only labeled clients participate
    
- Standard cross-entropy loss is used
    
- No unlabeled data is involved yet
    
- Typically lasts only a few communication rounds (often one)
    

This produces a **reasonably stable initial global model**, which is crucial because:

- All pseudo-labels later depend on this model
    
- Poor warm-up leads to failure downstream
    

During this stage, **residual weight connections** are already applied to improve stability (explained later).

---

## 2. Fixed pseudo-labeling on unlabeled clients

This is the **core conceptual change**.

### Key difference from standard pseudo-labeling

In most SSL methods (e.g. FixMatch, FlexMatch):

- Pseudo-labels are updated **every batch**
    
- Model predictions evolve continuously
    

In CBAFed:

- Pseudo-labels are generated **once per communication round**
    
- The selected pseudo-labeled set is **fixed during local training**
    

### How it works

At the start of round ttt, for each unlabeled client:

1. Initialize the local model with the global model
    
2. Predict labels for all local unlabeled samples
    
3. Select samples whose confidence exceeds a threshold
    
4. **Freeze this selected set** and train only on it
    

This prevents:

- Rapid feedback loops
    
- Overwriting previously learned labeled knowledge
    
- Catastrophic forgetting caused by non-IID updates
    

---

## 3. Class-balanced adaptive pseudo-label thresholds

Instead of a **single global confidence threshold**, CBAFed computes **class-specific thresholds**.

### Why this matters

Under non-IID data:

- Head classes dominate pseudo-label selection
    
- Tail classes receive few or no pseudo-labels
    
- Fixed thresholds either:
    
    - Admit noisy labels (too low), or
        
    - Exclude tail classes entirely (too high)
        

### How thresholds are computed

At each round, the server:

1. Collects class counts from:
    
    - True labels (labeled clients)
        
    - Pseudo-labels (unlabeled clients)
        
2. Estimates the **empirical class distribution**
    
3. Computes thresholds that:
    
    - Increase for over-represented classes
        
    - Remain sufficiently high for rare classes (lower-bounded)
        

This differs from FlexMatch-style curriculum learning:

- Thresholds depend on **global class distribution**, not local confidence ratios
    
- Prevents near-zero thresholds for rare classes
    

---

## 4. Tail-class data discovery

CBAFed additionally exploits **“not-confident” unlabeled samples**.

For samples whose:

- Maximum confidence is below the threshold
    
- Second-highest prediction corresponds to a rare class
    

The method:

- Reassigns the label to this second-most likely class
    
- Includes the sample in training
    

This explicitly combats **head-class bias** and encourages discovery of tail classes.

---

## 5. Residual weight connections (training & aggregation)

To further stabilize learning, CBAFed introduces **residual weight connections**:

- During local training:
    
    - Current weights are blended with weights from previous epochs
        
- During global aggregation:
    
    - Current global model is blended with previous global models
        

This acts like **temporal weight averaging**, improving robustness when:

- Training data is scarce
    
- Class distributions are imbalanced
    
- Local updates are noisy
    

Unlike Mean Teacher:

- No EMA teacher is used
    
- No consistency loss is involved
    
- The mechanism operates purely at the **parameter level**


## End-to-end training loop

Each communication round proceeds as:

1. Server sends:
    
    - Global model
        
    - Adaptive class thresholds
        
    - Class distribution statistics
        
2. Labeled clients:
    
    - Train on labeled data with residual weight connection
        
3. Unlabeled clients:
    
    - Generate fixed pseudo-labels using adaptive thresholds
        
    - Train on the fixed pseudo-labeled set
        
4. Clients return:
    
    - Updated models
        
    - Class counts
        
5. Server:
    
    - Aggregates models with residual weight connection
        
    - Updates thresholds for the next round

FSSL Setting Simulation Following [14], training datasets
contain 10 clients: one labeled and nine unlabeled. We
use a Dirichlet distribution Dir(γ), where γ = 0.8 for five
datasets [14] to generate Non-IID data partition in clients.
Implementation Details We utilize the SGD optimizer
with a momentum of 0.9, and implement our method with

![[FSSL SOTA.png]]



# RSCFed (2022)
RSCFed tackles **federated semi-supervised learning under non-IID data** when **client reliability is uneven**:

- Some clients are **fully labeled** → accurate but locally biased
    
- Some clients are **unlabeled** → less accurate, noisier updates

In non-IID FSSL, some client models are outliers and should not contribute equally to the global update.

## High-level idea

Instead of averaging _all_ client models at once, RSCFed:

1. **Randomly samples subsets of clients**
    
2. **Builds several “sub-consensus” models**
    
3. **Aggregates those sub-consensus models** into the global model
    

This is inspired by **RANSAC**: outliers are unlikely to dominate many random subsets.


### Labeled clients

- Train normally with **cross-entropy**
    
- Return updated models
    

### Unlabeled clients

- Use a **Mean-Teacher consistency framework**
    
- Student model learns to match a sharpened teacher prediction
    
- Only consistency loss is applied (no pseudo-labels)
    

This part is **standard FSSL** and not the main novelty.


## What happens on the server (the key part)

### Step 1: Random sub-sampling

At each round:

- The server randomly selects **K clients**
    
- This is repeated **M times**
    
- Each selection forms a **client subset**
    

---

### Step 2: Sub-consensus model construction

For each subset:

- Local models are aggregated **within the subset**
    
- Instead of simple averaging, RSCFed applies **distance-reweighted aggregation**
    

#### Distance-reweighted aggregation (intuition)

- Compute the average model of the subset
    
- Measure how far each client model is from that average
    
- Models that deviate strongly are **down-weighted**
    
- Models close to the consensus get **higher weight**
    

This suppresses:

- Noisy unlabeled clients
    
- Highly biased labeled clients
    

---

### Step 3: Global aggregation

- All **sub-consensus models** are finally averaged
    
- The result becomes the new global model

## What setting RSCFed assumes

- **Federated semi-supervised learning**
    
- Mixed client types:
    
    - Fully labeled + fully unlabeled clients
        
    - Or partially labeled clients
        
- **Explicitly non-IID data** (Dirichlet partitions)
    
- Server has **no data**

## Why this does NOT cancel out useful information

Two reasons:

### 1. Weighting is **not symmetric**

Weights are:

- Exponential in distance
    
- Normalized per subset
    

So:

- Being close in one subset gives **much more influence**
    
- Than being far in several subsets removes
    

This is **not linear averaging**.

## Mean Teacher — explained simply

Mean Teacher uses **two versions of the same model** on an unlabeled client:

1. **Student model**
    
    - Trained normally
        
    - Receives gradients
        
2. **Teacher model**
    
    - **Not trained directly**
        
    - Its weights are an **exponential moving average (EMA)** of the student’s weights
        

So:

- Teacher = smoothed version of the student over time
    
- Student = current model being optimized


### What happens on unlabeled data

For an unlabeled sample xxx:

1. Apply **weak augmentation** → teacher predicts a soft target
    
2. Apply **strong augmentation** → student predicts output
    
3. Train the student to **match the teacher’s prediction**
    

This enforces:

> “Your predictions should be stable over time and perturbations.”

No labels, no pseudo-labels, no thresholds.


They also perform partially labeled test, with 10% of data being labeled.

Cant access their data splitting!!!! verify this shit

# FedGGp (2024)
It targets the **“labeled on partial clients”** setting, but in a **much harsher form**:

- Many clients are **fully unlabeled**
    
- Labeled data is **scarce and class-imbalanced**
    
- Data is **strongly non-IID**
    
- There may be **no fully labeled client at all**
    

This is more realistic than the classic “one fully labeled client” assumption

ssrn-5201047

.

---

## Core problem the paper addresses (in simple terms)

Most FSSL methods fail because:

- Local models become **biased toward head classes**
    
- Pseudo-labels reinforce this bias over rounds
    
- Tail classes gradually **disappear from training**
    
- Consistency alone cannot fix this when labels are scarce
    

The authors argue that the _main failure mode_ is **biased pseudo-label generation**, not aggregation.

---

## High-level idea of FedGGp

> **Use the global model to detect which classes are over- or under-predicted, then adapt pseudo-labeling rules accordingly.**

Instead of trusting each client’s local confidence, the server maintains a **global prediction bias estimate**.

## How training works (step by step)

### 1. Global prediction evaluation (server-side)

- Clients do **not** send data
    
- Each client sends only a **class-wise sum of prediction probabilities** on unlabeled data
    
- The server aggregates this into a vector **Φ**, estimating:
    
    - Which classes are **over-represented (head)**
        
    - Which are **under-represented (tail)**
        
    - Which are **ambiguous**
        

This estimate is **smoothed over rounds** to stay stable


# CBDPL (2025)
## 3. Model architecture and training pipeline

### Backbone

- **DenseNet-121** encoder
    
- Segmentation head
    
- Loss = **Cross-Entropy + Dice loss** (to handle class imbalance)
    

### Training is split into two phases:

### Phase 1 — Supervised FL on labeled data

Each client:

1. Trains locally on its labeled subset
    
2. Uses CE + Dice loss
    
3. Sends weights to the server
    
4. Server aggregates using **FedAvg**
    

This produces a global model MLM_LML​ trained only on labeled data

Federated_Learning_model_using_…

.

---

### Phase 2 — Semi-supervised learning with CBDPL

Each client:

1. Uses the global model to predict labels for **unlabeled images**
    
2. Computes **softmax confidence scores**
    
3. Assigns pseudo-labels **only if confidence > τ (initially 0.8)**
    

Formally:

y^k=arg⁡max⁡cfθ(uk)if max⁡fθ(uk)>τ\hat{y}_k = \arg\max_c f_\theta(u_k) \quad \text{if } \max f_\theta(u_k) > \tauy^​k​=argcmax​fθ​(uk​)if maxfθ​(uk​)>τ

Key CBDPL components:

- **Confidence thresholding**
    
- **Gradual pseudo-label inclusion** (threshold can increase over rounds)
    
- **Class-balanced weighting** to avoid dominant-class bias
- Weight pseudo-labels heavier if they come from less represented classes?
    

Pseudo-labeled samples are then **merged with labeled data locally**, and clients retrain before another FedAvg round

## 4. What CBDPL actually is (conceptually)

CBDPL is **not a novel SSL paradigm**, but a **filtered pseudo-labeling strategy** with:

- Hard pseudo-labels
    
- Confidence-based acceptance
    
- Simple thresholding (no entropy minimization, no consistency regularization)
    

So compared to FixMatch:

- ❌ No weak/strong augmentation consistency
    
- ❌ No explicit confirmation-bias correction
    
- ✅ Explicit confidence filtering
    
- ✅ Class-balancing heuristic
    

This makes CBDPL **closer to classic self-training**, adapted to FL.


## 5. Results and what they show

### Performance

- Final **global accuracy ≈ 97.55%**
    
- Comparable to centralized training
    
- Pseudo-labeled training reaches:
    
    - Accuracy ≈ **92.7%**
        
    - Lower loss than supervised-only phase
        

They also report:

- Stable training across clients
    
- Low variance in 5-fold cross-validation
    
- Visual segmentation examples that align reasonably with ground truth
### Clients and data split

- The dataset is the **Multi-Cancer Dataset** (Kaggle), covering **8 cancer types / 26 subclasses**
    
- Data is split across **4 federated clients**
    
- **Each client has both labeled and unlabeled data**
    
    - **30% labeled**
        
    - **70% unlabeled**
        
- This is a **horizontal FL setup** (same task, different samples)

# Federated Semi-Supervised Learning for Medical Image Segmentation via Pseudo-Label Denoising (2023)

### Phase 1 — **Federated Supervised Learning (FSL)**

✔ **Only labeled clients participate**

- Runs for **~100 federated rounds**
    
- Purpose: learn a **strong, generalizable initialization**
    
- Uses standard FedAvg on labeled clients only
    
- Output: a converged **FSL model**
    

👉 **Unlabeled clients do nothing yet**

---

### Phase 2 — **Federated Semi-Supervised Learning (FSSL)**

✔ **Labeled _and_ unlabeled clients participate together**

This is the long main phase.

#### What happens here:

- **Labeled clients**
    
    - Continue training **every round**
        
    - Still use ground-truth labels
        
- **Unlabeled clients**
    
    - Receive pseudo-labels generated from the current global model
        
    - Train **every round** using those pseudo-labels
        
    - Pseudo-labels are **regenerated periodically** (every 50 rounds)
        
- **Server**
    
    - Aggregates **both labeled and unlabeled updates**
        
    - Uses weighted aggregation:
        
        θ←μ ξ+(1−μ) ψ\theta \leftarrow \mu\,\xi + (1-\mu)\,\psiθ←μξ+(1−μ)ψ
        
        (labeled clients dominate)
        

👉 **At no point are labeled clients removed**