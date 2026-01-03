
https://caliweb.vub.be/?page=course-offer&id=004892&anchor=1&target=pr&year=2324&language=en&output=html

### Thesis rebranding
The thesis before was talking about how FixMatch+FedAvg is underexplored, mostly focusing on the fact that people have unfaithfully reimplemented this naive combination and overlooked it.

After looking through the fixmatch paper again, they do note the use of unlabeled epochs over Imagenet in their appendix. Though this is not their core approach, this better fits the FL framework. Therefore, we can not assume anymore that people have unfaithfully reimplemented FixMatch+FedAvg. Moreover, this means that in different FL scenarios, this baseline is explored (still not too thoroughly, and not in our specific domain).

Therefore we alter the topic of the thesis to "Creating a unifying and reproducible baseline for FixMatch+FedAvg and investigating the 2 most common extensions". This does not shift the thesis at all but better aligns the core message.

To do this, I will also adopt this unlabeled epoch terminology.


Lets do 5 clients for 50 or 100 rounds F=1.0 (prolly 50 rounds and then use rounds as ablation) (7h for 50 rounds 5 clients ish for fixmatch on 1 gpu, and for supervised half a minute, so 25m for 50 rounds.) Prolly do 100 rounds.

The tests i will run are as follows (using multi-metrics: accuracy, f1 macro, recall macro and auroc):

To evaluate these questions, I will setup the following testing framework:
I will use a multi-metric approach (AUROC, accuracy, F1 macro and recall macro)
I will use the following hyperparameters for the initial Extended CIFAR-10 test:
Number of clients (K) = 5, Number of rounds = 50 or 100 (not decided yet)
Participation ratio (F) = 1.0, unlabeled batch size ratio ($\mu$) = 5.0
labeled_budgets = 1000, 2000, 4000. Data distribution settings = IID, alpha = 0.5, alpha = 0.1 .

I will run the naive Fixmatch+Fedavg in all these settings and evaluate the trends and compare it to a centralized supervised upper bound and a federated supervised lower bound.

After this, due to computational complexity, i will select 1 scenario (either per alpha or a singular alpha) and perform extension tests.

These extension tests will comprise of first tuning each extension separately (using optuna) and then also tune the combination of both of them together.
I will then run:
FixMatch+FedAvg+STM:
- Discuss the amount of labels accepted into short term memory and their correctness
FixMatch+FedAvg+AT:
FixFed (STM+AT): (and compare to naive results of that scenario)

Next will be the generalization tests;
Again, 1 specific situation and i will evaluate both the baseline and fixfed over 2 more datasets.

Finally the ablations, where i will compare fixfed ot the baseline:
- I will ablate over the different types of backbones 
- Ablate over the client participation (i will use 100 clients here and only select 5 or 10%) for both fixfed and fixmatch+fedavg
- Ablate over the number of rounds for both to see the effect of 100 or 200 rounds.


First thoroughly investigate Naive Fixmatch+FedAvg on Cifar-10.
- Use different label budgets and different data distributions to investigate naive combination and compare to upper/lower bound. Most likely we will see the same trend in more IID scenarios where performance suffers. If this is the case, we investigate what the cause is by decreasing the participation fraction.
- Charts: Lineplot: Accuracy/Round, Lineplot: accuracy per label fraction, Radarplot: all metrics for 1 label budget for each alpha (3 radar plots basically). And a table with all results.


After this, I will tune the STM using optuna, and same for adaptive thresholding and the combination of both.
Notably, only 1 scenario will be selected from above, we focus on a specific label budget and alpha.
This will be my extensions tests:
FixMatch+FedAvg+STM:
- Discuss the amount of labels accepted and their correctness
FixMatch+FedAvg+AT:
FixFed (STM+AT): (and compare to naive results of that scenario)
Just a table perhps with multi-metric and perhaps a lineplot accuracy/round
Finally
Compare FixFed and naive on 2 datasets: SVHN + CIFAR-10.
a table.

Ablation:
Alter the backbone model and see effects, mtoivate by need of small models on heterogeneous devices.
Multi metric table / the radar plot
Alter client participation, we could run 100 clients and only select 1 (1gpu) as their datasets would be minute. # Active clients vs accuracy (Could also use 10 clients here and reduce participation) Do this for fixfed and F+F
Increase nr of rounds for FixFed to see performance, motivated by the curve in first big test? 

### **1️⃣ Core Results (Naïve FixMatch + FedAvg)**

- Purpose: Establish a baseline for your experiments.
    
- Run the **naïve combination** across all relevant conditions:
    
    - Number of clients (1, 2, 5)
        
    - Label amounts (1000, 2000, 4000, 10000)
        
    - Data heterogeneity (α = 0.1, 0.5, IID)
        
    - Possibly a few rounds (standard and low, e.g., ≤5)
        
- Metrics: Accuracy, F1_macro, Recall_macro, AUROC, error rate.
    
`Just remove 1 of the seeds as we've already ran these`

> This shows clearly where the naïve approach struggles and gives a direct comparison to centralized / supervised federated baselines.


Also show the increase over time of val accuracy which motivates increasing nr of rounds even more 
`Just remove 1 of the seeds as we've already ran these`

Also investigate what is the cause of this poor performance in an ablation. How? 
	Ok; to investigate what causes poor performance of non-iid (alpha = 0.1), we already ahve the fact that supervised performance is extremely low.
	So we just need to run a test with 1 and 2 clients (fraction reduced) where we see how well it performs (removes FL component). The 1 client will indicate the SSL part and sees how that is under non-iid. and 2 clients reduces the effect of FedAvg and lets us investigate if it is negatively impacting or boosting performance. 

> 6 tests

Also discuss the trends that we have observed. How lambda effect is more flat indicating that the SSL part is also obviously affected.
`Just remove 1 of the seeds as we've already ran these`



### **Extension Results (FixFed with Enhancements)**

- Purpose: Isolate the effect of your proposed improvements.
    
- For **each extension**:
    
    - Adaptive threshold only
        
    - Short-term memory only
        
    - Both combined
        
- **Run these tests on the same conditions as your naïve model** (each α, mid-range label amount (4K), 5 clients).
    
- Metrics: Same as above.

> 3 extension settings, 3 seeds, 3 alphas: 27 tests
> BUTT: have to first run some optuna, But alpha also affects how many labels are kept, so prolly have to lower threshold to keep more labels. Reduce to 2 clients and 2 rounds and None iterations which loops over labeled dataset 1x.

CBDPL is the only of retention that refines the labels over rounds, while the rest resets it each round.



### **3️⃣ Optional Sensitivity / Robustness**

- Once you’ve shown the extensions improve performance in the “core” setting, you can test the full extended model across other conditions: FOR 1 alpha
    
    - Vary clients, datasets, backbones, rounds. 
>  (5 clients vs 1 client vs 10: 2 x 1 x 3) + (just run svhn, 1 x 1 x 3) + (2 x 1 x 3) + (2 x 1 x 3) = 21 tests

- 
        
    - Focus on the combined enhancements only (no need to re-test each individual extension again for all combinations).
For the STM, as we will be increasing size, ensure we store the idxs and their pseudo labels per epoch (this gives us size and nr of correct added samples) 
Dont test different ttl values, just size, keep prob and ordered dict mode


For ablation, we can investigate the effect of different lambdas and other hyperparameters if we have time.
> Discuss how errorbounds flare out at higher lambdas especially for lower labeled amounts.
	

Next test to run:
> For all these tests, choose a specific labeled amount, 2K maybe? Note that lambda 1 sees degradation over lambda 0.5 so we will use 0.5 for further tests
> 
> First run a sample test with 5 clients and 1 epoch/round for fraction to ensure code works
- Reduce the fraction of clients and fix the clients chosen (eg 1 and 2) and see effect on preformancei n alpha = 0.1 ==> ~~ADD F TO FILE ID.~~ (50% frac and 20%).
- > For this test, we forcefully select the same X clients each round to reduce the effect of randomness (We also use lambda 0.5 due to the random effect of lambda on this condition)

> Understand stm keeping behavior at alpha = 0.1 and iid for fixed threshold
- Run a small test with 2 clients and 2 rounds and iters = None and STM on in alpha = 0.1 and iid and see how many labels are kept and how many are rejected and distribution of those labels. ==> ADD # REJECTED TO TRAINING LOG.
- Run a simple optuna sample to see how it reacts on hpc (do 2 clients 2 rounds but its a fraction of 0.5, this will show us if the stm is even used or not)
> Most likely have to alter code for multi gpu sadly.

- Optuna tune the STM and the AT in separate multi gpu tests. Make it  2 clients and 2 rounds and None iterations which loops over labeled dataset 1x, and do like 30 trials. Use the results from the previous test to understand what values to pass. Only tune the size, and the keep probability. Mention that ttl is 3 and the goal would be to do more epochs and tune this as well. ==> REMEMBER THAT I DO NOT SEND THEIR TRAINING LOGS ON MULTI GPU AND ALSO THAT THIS PROLLY DIFFERS FROM ALPHA IID AND 0.1
- After these results are in; create best possible setting for both (again, this can depend on the alpha) and the combination and use this to run the same test as before (all alphas) BUT a specific labeled amount. Perhaps here run not only 5 clients but also 10 to compare to semifed
- Then all that remains to be done is the ablations.
- And if theres time, in extensions or ablations mimic semifed and comapre frameworks.



Investigate : 

# GPT THOUGHTS

## A sharpened version (recommended)

### Polished abstract (drop-in)

Federated Learning (FL) enables multiple clients to train a shared model without sharing raw data. In practice, labeled data is scarce due to cost and expertise constraints, yielding the Federated Semi-Supervised Learning (FSSL) setting where most client data is unlabeled. Prior work often proposes specialized optimization or aggregation schemes, while the behavior of naïvely combining standard semi-supervised learning (SSL) with FedAvg is underexplored.

We study a simple FixMatch-plus-FedAvg baseline under label scarcity and client heterogeneity, and evaluate two lightweight extensions: (i) **self-training via pseudo-label retention** across rounds, and (ii) **adaptive confidence thresholds** for unlabeled loss. We perform controlled experiments varying the number of clients, communication rounds, participation rate, and data heterogeneity (IID → highly non-IID Dirichlet).

Across datasets and backbones, we analyze accuracy, label-efficiency, calibration, and communication cost. We find that [insert key result], that performance under high heterogeneity degrades primarily due to [FedAvg/SSL component], and that the proposed extensions [improve/maintain] robustness under label scarcity with modest overhead. Code: \url{[https://github.com/YouriLangh/MsC_Thesis}](https://github.com/YouriLangh/MsC_Thesis%7D).

### Tightened research questions

- **RQ (main).** How do **adaptive thresholding** and **pseudo-label retention** change performance and robustness of FixMatch+FedAvg under label scarcity and non-IID data?
    
- **RQ1 (scaling).** What is the effect of **rounds** and **#clients/participation rate** on convergence, accuracy, and communication cost?
    
- **RQ2 (heterogeneity).** How does **IID→non-IID (Dirichlet α)** affect naïve vs. extended methods?
    
- **RQ3 (generalization).** Do results hold across **datasets (CIFAR-10/100, SVHN)** and **backbones (shallow vs. ResNet-18)** compared to **centralized supervised** and **federated-supervised** baselines?
    

---

## Experiment design: what’s great + what to tweak

### Baselines (keep)

- **Upper bound:** centralized supervised (all labels only).
    
- **Lower bound:** federated supervised (labels only at clients).
    
- **Naïve:** FixMatch + FedAvg.
    

**Add one lightweight baseline:** **FedAvg + FixMatch with FedProx or SCAFFOLD** _once_ at α=0.1 (non-IID worst case). This isolates whether divergence is FedAvg’s fault. You don’t need to sweep—just a single control point strengthens the causal story.

### Core grid (good)

- **Distributions:** IID, mid non-IID (α≈1), heavy non-IID (α≈0.1).
    
- **Label budgets:** 5 amounts (e.g., {500, 1k, 2k, 4k, 8k total}).
    
- **λ_u:** 2 values (state which; e.g., {1, 5}).
    
- **Rounds:** 5, 10, 20 (ablation).
    
- **Clients:** 2, 5, 10; **participation**: {100%, 50%, 20%}.
    

**Tweak:** keep the giant grid _only_ for the naïve baseline + bounds. For extensions, **freeze**: one label budget (2k), one λ_u, and sweep α ∈ {IID, 1, 0.1}. This limits cost while answering RQ2/RQ3.

### Extensions & tuning (careful!)

- **Optuna** for self-training and adaptive thresholding: great—but avoid overfitting.
    
    - **Tune on a _single_ setting** (e.g., CIFAR-10, α=1, 2k labels, 5 rounds, 5 clients).
        
    - **Lock the best hyperparams** and **transfer** to other α / datasets / backbones.
        
    - Report the **transfer gap**—that’s your robustness claim.
        

**Guardrails**

- Use **client-side validation splits** aggregated centrally for model selection. No test leakage.
    
- **3 seeds** per setting; report mean ± 95% CI (bootstrap or t-interval).
    
- Fix **unlabeled pool size** across methods; vary only what you claim to vary.
    
- Log **communication cost** (bytes/round × rounds) and **compute** (FLOPs or time proxy).
    
- Track **calibration** (ECE) and **client fairness** (mean±std client accuracies).
    

### The “is SSL or FedAvg to blame?” study (nice!)

- Your plan to drop participation (100%→50%→20% and 5→2→1 clients) is good.
    
- Add **one run with SCAFFOLD or FedProx** at α=0.1 to triangulate the culprit.
    

### Ablations (make them truly single-variable)

- **Backbone:** shallow vs. ResNet-18 (hold α, labels, rounds, clients fixed).
    
- **Rounds:** 5/10/20 (hold others fixed).
    
- **Clients:** 2/5/10 (hold others fixed).  
    Report **learning curves** (val accuracy vs. rounds) to support your “does val improve each round?” motivation.
    

---

## Measurements & reporting

- **Primary:** top-1 accuracy on a **global test set** (same for all methods).
    
- **Secondary:** ECE, **client-level variance**, communication (MB), wall-clock/round.
    
- **Label-efficiency curve:** accuracy vs. #labels (for naïve vs. extended).
    
- **Sensitivity plots:** (i) τ histogram/trajectory for adaptive thresholding; (ii) retained pseudo-labels over rounds with precision of retained set.
    
- **Stat tests:** paired t-test or bootstrap between naïve and extended at α=0.1, 2k labels. Report **effect size** (Cohen’s d).
    

---

## Threats to validity (call these out)

- **Hyperparameter transfer bias:** mitigated by single-setting tuning + frozen transfer.
    
- **Seed sensitivity:** mitigated by 3 seeds + CIs.
    
- **Client sampling randomness:** report with/without fixed sampling.
    
- **Compute budget:** clearly state GPU hours; justify why you restricted sweeps for extensions.
    

---

## Minor wording/structure fixes

- “**This creates setting**” → “This creates **the** setting”.
    
- “**Our results shows**” → “**Our results show**”.
    
- Rename “research questions:” to “**Research questions.**” and format as a list, not inline.
    
- Use consistent notation: Dirichlet **α**, λ_u, participation **p**.
    

---

## What to cut if time is tight

1. Run the **full grid** only for naïve + bounds.
    
2. For extensions, **tune once**, then **transfer** (α sweep + 1 extra dataset + 1 backbone).
    
3. Replace FedProx/SCAFFOLD sweeps with a **single control point** at α=0.1.
    

---

## Verdict

Your framework is well-designed and clearly aligned with the thesis. With the small additions (one control non-iid optimizer, careful tuning/transfer protocol, and stricter single-variable ablations) you’ll have a rigorous, defensible story about **when** FixMatch+FedAvg fails under heterogeneity and **how** lightweight extensions recover performance—without exploding the compute budget.




# What to change (quick)

- **Rounds (T):** use **25–50** for baseline; ablate **{10, 25, 50, 100}** on one setting.
    
- **Local compute:** keep **E=1** local epoch, and **fix local iterations per round** so FixMatch’s step-based budget is comparable.  
    
- **Participation (q):** main = **100%**; heterogeneity stress test: **50%** and **20%**.
    
- **Clients (K):** main = **10**; ablate **{2, 5, 10}** on one condition.
    
- **Non-IID (Dirichlet α):** **{∞ (IID), 1.0, 0.1}**.
    

# Minimal, defensible experiment plan (MVP)

## A) Baseline grid (CIFAR-10)

- Label budgets: **{0.5k, 1k, 2k, 4k}** total
    
- λᵤ (unlabeled weight): **{1, 5}**
    
- Heterogeneity: **α ∈ {∞, 1.0, 0.1}**
    
- FL setup: **K=10, q=1.0, T=50, E=1, L=1024**
    
- Seeds: **3**
    

This gives you the learning curves you need; it will also show 0.1 struggling.

## B) “Who is to blame?” sanity

- Fix **2k labels, α=0.1**, vary **q ∈ {1.0, 0.5, 0.2}** and **K ∈ {10,5,2}** (keep **T=50, E=1, L=1024**).
    
- (Optional single-point control) swap FedAvg→**FedProx (μ=0.01)** at **α=0.1** to triangulate FedAvg vs SSL without a sweep.
    

## C) Extensions (FixFed) — tuned once, then transferred

- **Optuna tuning (cheap proxy):** CIFAR-10, **α=1.0**, **2k labels**, **K=5**, **q=0.5**, **T=10**, **L=256**, **μ (ratio) = 3**.  
    Pruner = **ASHA**, 30 trials → keep top **3–5**.
    
- **Lock best hparams**; **final runs** on CIFAR-10 at **{α=∞, 1.0, 0.1}**, **K=10**, **q=1.0**, **T=50**, **L=1024**, **μ=7**.
    
- **Generalization**: CIFAR-100 and SVHN at **α=1.0**, **2k labels**, **T=50**, **K=10**, **q=1.0** (no ablations; just naive vs FixFed).
    

# Concrete hyperparameters (final runs)

- **Backbones:** Simple CNN and **ResNet-18** (Image size 32). Use **EMA=0.999**.
    
- **Optimizer:** SGD **lr=0.03**, momentum **0.9**, weight decay **5e-4**; **cosine** decay over total local steps.
    
- **Batch sizes:** **64 labeled**, **64×μ unlabeled** per iter (μ below).
    
- **Unlabeled ratio (μ):** **7** (final), **3–5** (tuning).
    
- **FixMatch baseline:** weak=flip+random crop (padding 4), strong=**RandAugment(N=2, M=10)**, **τ=0.95**, **λᵤ ∈ {1,5}**.
    
- **Aug pipeline:** prefer GPU aug (torchvision v2/Kornia). `cudnn.benchmark=True`, mixed precision.
    
- **FedAvg:** weighted by client dataset size; **client sampling is fixed-seed** per round for reproducibility.
    

# Your two extensions (simple, explicit)

1. **Adaptive threshold τ(t)** (no per-class complexity):  
    ( \tau(t) = \tau_{\min} + (\tau_{\max}-\tau_{\min}) \cdot (t/T_{\text{steps}})^\gamma )  
    Use **τ_min=0.80**, **τ_max=0.95**, **γ=1.5**. (t = global step index across rounds; monotone ↑)
    
2. **Pseudo-label retention (short-term memory):**  
    Per client, keep a ring buffer of **up to 5k** unlabeled samples with **EMA-smoothed confidence ≥ 0.98**; age-decay **0.9/round**; replay a **mini-batch fraction 0.25** each local round mixed with fresh unlabeled. Remove entries if their confidence drops below 0.9.
    

> Both are **orthogonal to FedAvg**; no server changes.

# Reporting (kept minimal)

- **Accuracy** (mean ± 95% CI, 3 seeds), **ECE**, **bytes communicated** (MB), **rounds-to-X%** convergence.
    
- Show **val/test vs rounds** curves for: baseline (C10; α∈{∞,1,0.1}), then overlay FixFed.
    
- One table for **generalization** (CIFAR-100, SVHN): naive vs FixFed at α=1.0.
    

# Time savers you can safely use

- For Optuna: **μ=3**, **L=256**, **T=10**, **K=5**, **q=0.5**, pruner=ASHA, report metric every **200 iters**.
    
- For ablations: fix **labels=2k**, **α=1.0**, **backbone=ResNet-18** unless the ablation target is backbone.
    
- Cache datasets; precompute decode/resize; use **AMP**.
    

# Why this fixes your comparability concern

- You **fix the unit of budget** to **iterations/round** (L) and **rounds (T)** and report both.
    
- You keep **E=1** (simpler, less drift) and scale via **L** to match FixMatch’s step semantics.
    
- You provide a **single, transparent protocol** others can copy.
    

---

If you want, paste your current config (GPU, batch throughput), and I’ll compute an estimated wall-clock for the MVP grid so you can plan runs realistically.



we'll do 3 epochs (usually 1-10)
34 local iters (34 x 3 epochs = 102 which is 1/10th of 1024)
50 rounds
5-10 clients (5 for tests)
Rounds Test yourself, try a 10 rounder, a 20 and a 50 rounder.
(usually its 100+)
batch-size(10-64)
uratio = 7

Client fraction is expensive, we can test it though.

Do criticize the "epoch" notation used, as its not common to Fixmatch
==> They do mention it in FixMatch; but nevertheless, its more faithful to use local iterations.


remove local iters arg.

set local epochs to 5 and just do 1/5th round test max of ggp

Use 5 & 10 % labeled data not more
5 epochs n see how many rounds
add an alpha

Ablation add tests: 
vary fraction?

lr = 0.03


### ✅ Experiments Required by the Abstract

#### **1. Unified FixMatch + FedAvg Baseline (CIFAR-10)**

Evaluate baseline robustness under realistic FSSL constraints:

- **Vary label scarcity**
    
    - total labeled samples = {500, 1 000, 2 000, 4 000}
        
- **Vary client heterogeneity (Dirichlet α)**
    
    - α ∈ {∞ (IID), 1.0, 0.1 (highly non-IID)}
        
- **Vary partial participation**
    
    - participation rate q ∈ {100%, 50%, 20%}
        
    - with fixed number of clients (e.g., K = 10)
        

Metrics to report:

- accuracy, macro-F1, calibration (ECE), pseudo-label acceptance rate
    

---

#### **2. Failure Mode Analysis (Baseline FixMatch + FedAvg)**

Goal: identify when the naïve baseline breaks.

On one fixed scenario (e.g., labels = 2k, α = 0.1):


---

#### **3. FixFed Extensions (same constraints)**

On the same core experimental setting (labels = 2k, α ∈ {∞, 1.0, 0.1}):

- **FixFed-A:** adaptive confidence thresholding only
    
- **FixFed-B:** pseudo-label retention only
    - Verification of nr of pseudolabel accepted/rejected and if they are correct to true label
- **FixFed-C:** both combined (full FixFed)
    

Compare each against the naïve baseline.

Metrics to report:

- accuracy, macro-F1, pseudo-label precision / confidence statistics
    

---

#### **4. Generalization Across Datasets**

Use **one configuration** (e.g., labels = 2k, α = 1.0, q = 100%) and run:

- CIFAR-100 (harder, more classes)
    
- SVHN (different visual modality)
    

Compare:

- Naïve FixMatch+FedAvg vs. FixFed
    

---

### 🔁 Optional (only if time allows)

- Backbone ablation (Simple CNN vs ResNet-18)
    

Metrics i use: f1 macro, accuracy, auroc, recall macro.


- **Accuracy & F1 vs rounds** (CIFAR-10): panels for α ∈ {∞, 1.0, 0.1}; overlay FixFed variants. vs baselines
    
- **Label-efficiency curves**: accuracy (and F1) vs total labels {0.5k,1k,2k,4k}.
    
- **Participation sensitivity**: bars for q = 100/50/20% at α=0.1.
    
- **Client fairness**: boxplot of per-client accuracy at α=0.1 (baseline vs FixFed).
    
- **Pseudo-label dynamics**: line of accepted pseudo-labels/round and a confidence histogram (baseline vs FixFed).
    
- **Generalization bars**: CIFAR-100 & SVHN, baseline vs FixFed (accuracy + F1).
    
- _(Optional)_ **Calibration**: reliability diagram + ECE table for α=0.1.


FINAL:
# What to run (final, minimal, publishable)

## 0) Global settings (fixed unless ablated)

- **Rounds T:** 50
    
- **Local training:** epochs over **unlabeled** data; labeled batches sampled **with replacement**
    
- **Batch sizes:** 64 (labeled), μ = 5 (unlabeled); τ = 0.95
    
- **Optimizer:** SGD lr 0.03, mom 0.9, wd 5e-4; cosine by total steps
	- What if i dont use momentum? just simple learning rate.
    
- **Backbone (default):** ResNet-18 (use “simple CNN” only in the backbone ablation)
    
- **Seeds:** 3 (report mean ± 95% CI)
    

---

## 1) Baseline grid (CIFAR-10) — **must have**

**Purpose:** characterize the naïve baseline and its failure modes.

- **Heterogeneity:** α ∈ {∞ (IID), 1.0, 0.1}
    
- **Label budgets (total):** { 1k, 2k, 4k, } ← four is enough
    
- **λᵤ:** {1}
    
- **Clients K / participation q:** K = 5, q = 100%
    



**Outputs:** accuracy, macro-F1, curves of val/test vs rounds.  
**Interpret:** note where α=0.1 struggles; check whether curves improve each round.
- Depending on the curves improving over time consistently, we can make a test testing this from fixfed

Make the table as we had, then the accuracy per round but also accuracy per label budget



## 2) “Who’s to blame?” sanity (one slice) — **must have**

**Fix:** labels=2k, α=0.1 (hard case), λᵤ=5, backbone=ResNet-18.

- **Vary participation:** q ∈ {100%, 50%, 20%} (K=5)
    
    

(Optionally one point with **FedProx μ=0.01** at α=0.1 to triangulate FedAvg vs SSL—single run, no sweep.)

---

## 3) Extensions (FixFed) — **tune once, transfer** — **must have**

**Tuning (cheap proxy):** α=1.0, labels=2k, K=5, q=50%, **T=10**, μ=3, **ASHA** pruning, 30 trials.

- Tune **adaptive τ schedule** (τ_min, τ_max, γ)
    
- Tune **pseudo-label retention** (buffer size, min conf, replay fraction)
    

**Final FixFed runs (compare to naïve):** CIFAR-10, **α ∈ {∞, 1.0, 0.1}**, K=10, q=100%, **T=50**, μ=5.  
(Use the tuned hyperparams **frozen**; do not re-tune per α.)

---

## 4) Generalization — **nice to have** (keep tiny)

**One setting only:** labels=2k, α=1.0, K=10, q=100%, T=50, μ=5

- Datasets: **CIFAR-100, SVHN**
    
- Compare **naïve vs FixFed** (no ablations here)
    

---

## 5) Ablations — **do only 1 per category**

- **Backbone:** simple CNN vs ResNet-18 at **labels=2k, α=1.0**
    
- **Rounds:** T ∈ {10, 25, 50} at **labels=2k, α=1.0**
    
- **Clients:** K ∈ {2, 5, 10} at **labels=2k, α=1.0**
    

> One ablation = one small 3-point sweep. Don’t combine factors.


# Table/figure blueprint (so you don’t overproduce)

- **Tbl.1** Baseline CIFAR-10 grid (α × labels; λᵤ grouped)
    
- **Fig.1** Curves vs rounds (α panels)
    
- **Tbl.2** “Who’s to blame?” (α=0.1: vary q and K)
    
- **Tbl.3** FixFed vs naïve (CIFAR-10 across α)
    
- **Tbl.4** Generalization (CIFAR-100, SVHN one row each)
    

## Why this aligns with your thesis

- Establishes a **unified baseline** (clear protocol; same defaults everywhere).
    
- **Diagnoses** failure modes under heterogeneity/participation.
    
- Shows **FixFed (simple extensions)** improves robustness **without FedAvg changes**.
    
- Fits a **tight time budget** by limiting sweeps and re-use of tuned params.
    

If you want, I can turn this into a one-page “Experiment Plan” you can paste into Chapter 4, plus a ready-to-fill results table template.



**The motivation of this thesis is to establish a _unified, reproducible baseline_ for Federated Semi-Supervised Learning (FSSL) using FixMatch + FedAvg and to analyze when and why this naïve combination fails.**

Although many FSSL papers _cite_ FixMatch + FedAvg as a baseline, their training protocols vary (different definitions of epochs vs. iterations, different client participation, different data distributions), preventing fair comparison and making it unclear whether improvements reported in literature come from algorithmic innovation or simply from different experimental setups.

This thesis systematically evaluates a faithful implementation of FixMatch + FedAvg under comprehensive and realistic federated constraints—label scarcity at every client, non-IID data distributions, partial participation, and limited communication rounds. After exposing the baseline’s failure modes, we introduce minimal SSL extensions (adaptive thresholding and pseudo-label retention, called **FixFed**) to test whether simple, non-algorithmic refinements improve robustness without modifying the federated optimizer.


