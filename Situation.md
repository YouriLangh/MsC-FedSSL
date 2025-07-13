
- **FedMatch (2022)**: combines FixMatch with FedAvg
    
- **FedProto (2021)**: uses prototype-based classification and clustering
    
- **Fed-ET (2021)**: ensembles local models for teacher and distills global student
    
- **FedSTAR (2023)**: does semi-supervised training with server-side unlabeled data augmentation

## Strategy 2: _FedSSL with Teacher-Student and Role Reversal_

1. **Clients train** local **teacher** models with their few labeled samples + pseudo-labeled unlabeled data (e.g., FixMatch).
    
2. **Send teacher weights** to the server.
    
3. **Server aggregates** these into a global teacher model.
    
4. **Server uses weak data** to train a global **student** model (i.e., mimic the global teacher with public data).
    
5. **Send student back** to clients as initialization or regularizer (via distillation loss).
    
6. Repeat.
    

This alternates **knowledge distillation** between strong but sparse (client) and weak but dense (server) supervision.


Yes, this **teacher-student strategy with asymmetric roles in federated semi-supervised learning** **has been explored**, but **not exactly** in the specific form you're proposing — where:

- **Clients are teachers (trained on semi-supervised labeled + unlabeled data)**
    
- **Server is the student (trained on noisy or weak public data using knowledge distillation)**
    

This **asymmetric knowledge distillation with role reversal** is a **novel twist** compared to most existing works, which typically:

1. Have the **server as the teacher** (pretrained on public data) and **clients as students** (as in [FedDF, 2022]).
    
2. Use **peer-to-peer distillation** (e.g., FedMD, 2019), where models distill from each other using a public dataset.
    
3. Rely on **central server models** aggregating labeled gradients (e.g., standard FedAvg + FixMatch), but not using a _student model_ trained on pseudo-labels from the clients.
    

Your version—**clients training better-quality teachers on trusted data, and the server learning from that to deal with noisy public data**—is **less explored and promising**.

---

### ✅ Relevant but Different Works

|Paper|What it does|Difference from your idea|
|---|---|---|
|**FedMD (2019)**|Clients with different models distill knowledge into a shared student via logits on public data|Peer-to-peer distillation, not asymmetric roles|
|**FedDF (2022)**|Server maintains a global model and performs distillation from client models using logits|Server is teacher, not student|
|**FedGen (2021)**|Server generates synthetic features to help clients learn|Doesn’t do distillation|
|**FedGKT (2020)**|Split neural network and distillation for edge devices|Server still has partial raw data access|
|**FedMatch (2022)**|Semi-supervised learning on clients (FixMatch), FedAvg used to aggregate|No teacher-student distillation|

So: **parts of your idea exist**, but **your formulation (clients distill into a global server student model via public data) is novel in its asymmetry and application of weak data on the server side.**

---

### ✅ How to Know if the Student Mimics the Teachers

If you're training the **server-side student model** on public unlabeled data by using **teacher-predicted logits** as targets (soft labels), here’s how you measure it:

#### 1. **Prediction Alignment (KL Divergence or MSE)**

For each public input xx:

- Let T(x)T(x) be the softmax/logits from the **aggregated teacher**
    
- Let S(x)S(x) be the prediction from the **student**
    

Then use:

- **KL Divergence**:
    
    KL(T(x)∥S(x))=∑iTi(x)⋅log⁡(Ti(x)Si(x))\text{KL}(T(x) \parallel S(x)) = \sum_{i} T_i(x) \cdot \log\left(\frac{T_i(x)}{S_i(x)}\right)
- **MSE Loss** between logits or probabilities:
    
    MSE(T(x),S(x))=∥T(x)−S(x)∥2\text{MSE}(T(x), S(x)) = \| T(x) - S(x) \|^2

You can compute the average over the public dataset to quantify _how close_ the student mimics the teacher.

---

#### 2. **Agreement Rate (Hard Labels)**

- Convert predictions to **argmax** (class labels)
    
- Measure **percentage agreement** between teacher and student:
    
    Accuracyteacher-student=1N∑i=1N1[arg⁡max⁡T(xi)=arg⁡max⁡S(xi)]\text{Accuracy}_{\text{teacher-student}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\arg\max T(x_i) = \arg\max S(x_i)]

This gives a more interpretable measure like "student agrees with teacher on 85% of public data."

---

#### 3. **Downstream Evaluation**

You can also evaluate:

- How well the student performs on _client tasks_ (test on a few client test sets)
    
- Whether using the student to initialize clients improves convergence or accuracy in the next round
    

---

Let me know if you want a short paper-style method section based on this — it would make for a strong thesis or publication basis.

