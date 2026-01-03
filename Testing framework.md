Lets do 5 clients for 100 rounds F=1.0 (rounds as ablation) 
(7h for 50 rounds 5 clients ish for fixmatch on 1 gpu, and for supervised half a minute, so 25m for 50 rounds.) Prolly do 100 rounds. 100 rounds on multi gpu is like 3h

Give background on network architectures.
Before supervised and unsupervised 
can go in depth about dirichlet distribution
Go into different classifications tasks and 

In methodology try to attack which extensions adapt each assumption

Ensure all the code works as intended.
verify 10x
Fix the FL section perhaps w/o sota.

fix optuna tuning code n run a sample test (to ensure i can tune my extensions, not main framework)
Fix any text afterwards.
look for any issues in the code.


Metrics i would like to report for flexmatch, simply the threshold per round per class per client.

For STM: The number of samples accepted into the stm vs the number of samples that were confident but not accepted. and the size& distribution of samples within the stm each round




$\mu$ = 5
lambda_u = 1.0
n_rounds = 100
n_clients = 5

**Label budgets:** 1000, 2000, 4000
    
- **Heterogeneity:** IID, Dirichlet α = 0.1, 0.5, iid

(revise supervised client)



















### **1. Baseline Characterization (Primary Contribution)**

Evaluate naive FixMatch + FedAvg under systematically varied realistic FL conditions:

- **Datasets:** CIFAR-10
    
- **Metrics:** AUROC, accuracy, F1-macro, recall-macro
    
- **Clients:** K = 5
    
- **Participation:** F = 1.0
    
- **Communication rounds:** 50 or 100
    
- **Label budgets:** 1000, 2000, 4000
    
- **Heterogeneity:** IID, Dirichlet α = 0.1, 0.5, iid
    
- **Comparisons:**
    
    - centralized supervised upper bound
        
    - federated supervised lower bound
        
    - naive FixMatch + FedAvg
        

---

### **2. Extension Evaluation (FixFed)**

Select _one representative scenario_ (or one per α if feasible).

- **Tuning:**
    
    - Tune **AT** and **STM** separately using Optuna **only for the selected scenario**
        
    - Then tune the combination (FixFed)
        
- **Experiments:**
    
    - FixMatch + FedAvg + STM → analyze retention quality
        
    - FixMatch + FedAvg + AT
        - Have to ablate over the different stm modes
    - FixFed (STM + AT) → compare against baseline in the same scenario
        


---

### **3. Generalization Tests**

Using the same selected scenario:

- Evaluate baseline and FixFed on:
    
    - CIFAR-100
        
    - SVHN
        


---

### **4. Ablation Studies**


- **Client participation:** reduce participation (e.g., 5–10%)
    
- **Training budget:** vary number of rounds (100 vs 200)
    
- **Backbones:** Vary the backbone (SimpleCNN vs ResNet-18)

Ablate for each extension their parts (AT: initial threshold??) (STM: The size of allowed samples rate)













## Supported by literature 

From **CBAFed (Class Balanced Adaptive Pseudo-Labeling)**:

> “Batch-wise pseudo labeling produces volatile supervision and leads to catastrophic forgetting on unlabeled clients.”

From **CBDPL (Confidence-Based Dynamic Pseudo-labeling)**:

> “…the incremental update approach prevents noise propagation and stabilizes pseudo-label influence.”

These papers explicitly show that **FixMatch’s dynamic pseudo-labeling is _harmful_ in Federated Learning**.


In Federated Semi-Supervised Learning (FSSL), catastrophic forgetting is triggered **during the local training phase** on non-IID clients. When pseudo-labels are regenerated at every batch or every epoch (classic FixMatch behavior), the local model rapidly overwrites previously correct knowledge with newly generated—often noisy—pseudo-labels. CBAFed (Class-Balanced Adaptive Pseudo-Labeling) explicitly identifies this as the core cause of error amplification, stating that _batch-based pseudo-labeling leads to volatile supervisory signals and catastrophic forgetting during local optimization on unlabeled clients_ (CBAFed, 2023). To prevent this, CBAFed introduces a **fixed pseudo-label set**, constructed once per communication round and held constant across all local epochs, which “stabilizes the supervision signal” and prevents the model from “forgetting former learned knowledge.”

CBDPL (Confidence-Based Dynamic Pseudo-Labeling) reaches the same conclusion from a different direction: label quality must be **filtered before integration**, not constantly re-evaluated during training. CBDPL shows that pseudo-label volatility increases noise propagation and that stability can be achieved by **retaining high-confidence pseudo-labels** rather than re-computing them continually (CBDPL, 2022).

Based on these findings, we adopt **within-round pseudo-label retention**: pseudo-labels are generated once at the start of the round, filtered by confidence (FixMatch / FlexMatch thresholding), optionally per class, and then **frozen across epochs**. This preserves FixMatch’s training dynamics while preventing the local catastrophic forgetting observed in non-IID settings. Unlike cross-round memory methods, this keeps pseudo-labels consistent with the current global model while addressing the volatility and forgetting that occur _within_ a federated round, where the literature shows the problem originates.


"Instead of dynamically updating pseudo-labels within each iteration (as in FixMatch) or freezing them across all rounds (as in CBAFed), we apply a hybrid strategy.  
We retain pseudo-labels only **within the local round**, using the model state received from the server to generate a stable buffer.  
Admission into the buffer is gated using confidence filtering (CBDPL-style), but we do not perform partial/incremental integration across rounds.  
This avoids catastrophic forgetting inside a round, while keeping training aligned with FixMatch's per-epoch consistency loss."


max_new_per_round = int(buffer_capacity * integration_rate)
max_new_per_class = max_new_per_round // num_classes   # <= CRUCIAL


Add two simple SSL improvements that are used in literature:

- ✅ Adaptive thresholding (FlexMatch / CBAFed)
    
- ✅ Pseudo-label retention (CBAFed / CBDPL / FedGGp)

TODO:
Have to go over thesis and ensure our methodology follows new plan & ensure that our topic/title and such make sense and have practical applications, rather than just being some random ass results that are not applicable.

Update and extend experiments with new approaches/tests?
Write down that we use epoch based as that is common in fssl (methodology and fssl part)
Update methodology
Describe state of the art better. Investigate this again.

_"client-side memory bank of pseudo-labels"_  
or  
_"label caching across rounds"_

AUROC or 1vs all AUC
In discussion add why X or Y perhaps, like why we chose those 2 extensions.

Add deep learning or machine learning into supervised into unsupervised into SSL? and then this allows to branch out into self training.

The combination of both has been used in SSL!

And figure out another metric to discuss
Mention tht the extensions we consider are commonly used extensionsi n SSL and have each been applied in various frameworks as components. Motivating that they mighteb useful. Did these people ablate over them, creating a fixmatch+Fedavg baseline ish thing?
Update self-training with:
[[SelfTraining]]
Additionally, perhaps give some background on training and how labels are used (weak vs strong labels)
✅ So: **adaptive thresholding = common FixMatch extension**,  
✅ **pseudo-label retention = mostly used in Federated SSL / rarely in vanilla SSL.**
ADD: 
[[GPT]]

|                                       |                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------- |
| **CReST / CReST+ (Wei et al., 2021)** | Expand labeled set gradually in curriculum, retain confidently labeled samples permanently. |

|                                   |                                                               |
| --------------------------------- | ------------------------------------------------------------- |
| **FreeMatch (Wang et al., 2023)** | Maintains per-class confidence memories to adjust thresholds. |

|                                 |                                                         |
| ------------------------------- | ------------------------------------------------------- |
| **Entropy-Based Self-Training** | Retains labels with lowest entropy / highest certainty. |


Investigate the multi gpu result of increeasing rounds and see if it improves performance (even if shaky but increasing)

Optuna tuning stm AND at tgt is also required as AT alters stm, so have to redo the steps of stm


TODAY: Run a sample test with stm on on both alpha 0.1 and iid to see how results are received and such.
Then fix the tests for this but do not run yet
And then run a sample of optuna on multigpu/single gpu (we need to alter multip gpu) and see how that works out.



