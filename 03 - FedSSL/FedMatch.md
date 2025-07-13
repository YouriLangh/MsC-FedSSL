#FedSSL 

Code available
They test their code with labels-at-client & labels-@-server & non-iid case.
I think they just create extra models for the same unlabeled instance and then enforce consistency with the predictions of all those models

The two core components of the **FedMatch** paper—**local training with parameter decomposition** and **inter-client consistency**—in a way that’s intuitive.

---

## 🧠 **1. Local Training with Parameter Decomposition (Disjoint Learning)**

### 🧩 The Problem:

In normal semi-supervised learning, we train the same model on both labeled and unlabeled data. But that doesn’t work well in federated settings where:

- Some clients may **only have unlabeled data** (e.g., workout apps where users can’t label data themselves).
    
- Mixing labeled and unlabeled data training can cause **catastrophic forgetting**—the model “forgets” the labeled data knowledge as it trains more on unlabeled data.
    

### 🔧 The Solution: Parameter Decomposition

They **split the model parameters** into two parts:

- `σ` (sigma): trained only on **labeled data**
    
- `ψ` (psi): trained only on **unlabeled data**
    

They do:

- Supervised loss on `σ` (keeping `ψ` frozen)
    
- Unsupervised consistency loss on `ψ` (keeping `σ` frozen)
    

This:

- Prevents interference between supervised and unsupervised tasks
    
- Makes it possible to do **supervised learning at the server** (which has the labels) and **unsupervised learning at the clients** (which only have unlabeled data)
    

They also apply:

- **L2 and L1 regularization** on `ψ` to keep it sparse and close to `σ`, minimizing drift.
    

---

## 🔁 **2. Inter-Client Consistency Loss (ICCL)**

### 🤝 The Problem:

In federated settings, each client trains on different data (non-IID), so their models might make very different predictions—leading to inconsistencies across clients.

### 💡 The Solution: Enforce Consistency Across Clients

- Each client receives a few “**helper models**” from other clients (selected by the server based on model similarity).
    
- On the same unlabeled input, a client is encouraged to **match its predictions** with the helper models.
    

Formally, they add a **KL divergence** loss between the client’s prediction and those of the helper models.

### 🏷️ They also propose:

- An “**agreement-based pseudo-labeling**” scheme: Only assign a pseudo-label if multiple helpers **agree** on the class with high confidence.
    

This helps:

- Regularize the model to not drift too far from the global consensus
    
- Make better use of unlabeled data, especially when labeled data is scarce or unavailable
    

---

## 🧪 Summary of Benefits

- **Parameter decomposition** allows supervised learning to be centralized (at the server) and unsupervised learning decentralized (at clients).
    
- **Inter-client consistency** boosts generalization by aligning clients’ predictions, reducing noise from non-IID data distributions.
    

---

## 🧠 Setting:

Imagine client **C1** receives:

- An **unlabeled image** `u` (say, a picture of a cat)
    
- Its own model (with parameters `θ = σ + ψ`)
    
- A few **helper models** (say `H = 2` helpers: `C2`, `C3`) from other clients, which were selected based on model similarity
    

Each model will predict a **softmax probability distribution** over possible labels, e.g., {cat, dog, airplane, ...}

---

## 🔍 Step-by-step Example:

### 🔹 Step 1: Get predictions from frozen helper models

Let’s say the output of the helpers on `u` is:

```plaintext
C2: p₂ = [0.70 cat, 0.20 dog, 0.10 airplane]
C3: p₃ = [0.65 cat, 0.25 dog, 0.10 airplane]
```

These are softmax probabilities over 3 classes.

### 🔹 Step 2: Client C1 gets its own prediction

```plaintext
C1: p₁ = [0.40 cat, 0.50 dog, 0.10 airplane]
```

### 🔹 Step 3: Compute the Inter-Client Consistency Loss

The loss encourages C1 to **align its predictions with those of the helpers**.

The paper uses the **average of the helpers’ predictions** as the “target” distribution (which we freeze):

```plaintext
p* = average(p₂, p₃) = [(0.70+0.65)/2, (0.20+0.25)/2, (0.10+0.10)/2]
   = [0.675, 0.225, 0.10]
```

Then we compute the **KL divergence** from `p*` to `p₁`:

KL(p* || p₁) = ∑ p*_i * log(p*_i / p₁_i) = 0.675 * log(0.675 / 0.40) + 0.225 * log(0.225 / 0.50) + 0.10 * log(0.10 / 0.10)

The exact value is less important here. This loss is then backpropagated to update **ψ**, the unsupervised part of the model.

---

## 🧠 Bonus: Pseudo-labeling + Cross-Entropy

If the helper models strongly agree (say they both predicted "cat" with high confidence), then the client might also generate a **pseudo-label** like:

```plaintext
ŷ = one-hot("cat") = [1, 0, 0]
```

And use a standard **cross-entropy loss** to train on that:

CrossEntropy(y^,p1)=−log(0.40)←encouragesthemodeltoboostitscatscoreCrossEntropy(ŷ, p₁) = -log(0.40) ← encourages the model to boost its cat score

So the total loss on `u` might be:

Loss=CrossEntropy(y^,p1)+λ∗KL(p∗∣∣p1)Loss = CrossEntropy(ŷ, p₁) + λ * KL(p* || p₁)

---

## ✅ Summary:

On an **unlabeled data point**, a client:

1. Predicts `p₁` using its model.
    
2. Receives `H` helper model predictions (`p₂`, `p₃`).
    
3. Computes average `p*`.
    
4. Applies **KL loss** to push `p₁` closer to `p*`.
    
5. Optionally uses a **pseudo-label** if helpers agree, with cross-entropy.
    
Code example:
#### Inputs:
#### u        - unlabeled example
#### model    - local model with parameters θ = σ + ψ
#### helpers  - list of H frozen helper models received from server
#### τ        - confidence threshold for pseudo-labeling
#### π(u)     - strong augmented version of u

### Step 1: Get predictions
p_self = model.predict(u)            # Softmax output from local model
p_aug  = model.predict(π(u))         # Softmax output from local model on augmented input
p_helpers = [h.predict(u) for h in helpers]  # Softmax outputs from helper models

### Step 2: Agreement-based pseudo-label
### One-hot predictions from client and helpers
votes = one_hot(p_self)
for ph in p_helpers:
    votes += one_hot(ph)

# Choose class with most votes
y_hat = argmax(votes)   # pseudo-label (one-hot)

### Step 3: Check confidence (optional)
avg_conf = mean([ph.max() for ph in p_helpers])
if avg_conf < τ:
    skip this example  # discard low-confidence pseudo-label

### Step 4: Compute loss terms

### (a) Cross-entropy with pseudo-label on augmented image (FixMatch-style)
loss_ce = CrossEntropy(y_hat, p_aug)

### (b) Inter-client consistency (FedMatch): KL from helpers to client
loss_kl = 0
for ph in p_helpers:
    loss_kl += KL(ph, p_self)
loss_kl /= len(p_helpers)  # average KL

### Step 5: Combine and backprop
total_loss = loss_ce + λ_icc * loss_kl
total_loss.backward()
optimizer.step()

![[FedMatch.pdf]]