1. **Abstract**  
2. **Introduction**  
  -Motivation  
  -Contributions  
  -Research Questions  
3. **Background & Related Work**    (LINK ALL THE PAPERS)
  - Semi-Supervised Learning (SSL or Semi-SL) & more in depth fixmatch / Shortly discuss Self-Supervised Learning (SSL) as a separate branch in research
  - Federated Learning (FL)  
    -Aggregation Methods  
    -Core Challenges (Non-IID Data, Communication, etc.)  
    -Security & Privacy (Briefly, since it's not the focus)  
4. **Methodology**  
  -Problem Definition  
  -Proposed Framework  (and pseudocode algorithm)
  -Experimental Setup  
    - Datasets (MNIST/CIFAR/ Probably ImageNet in the future) 
    - How we ran the tests & benchmarks
    - Implementation Details  
    - Baselines (Federated Supervised / Federated SSL / Centralized)
5. **Results & Discussion** 
  -Performance Comparison (top-1 accuracy %% (which is just accuracy of highest confidence guess i think) %% comparison)  to other models
  -Ablation study
  -Non-iid setting tests
  -Further finetuned tests with more clients or lambda_u higher or pretraining 
  -Impact of SSL/Semi-SL on FL  
  -Challenges & Limitations  
6. **Discussion**
	-Overhead of STM in terms of training or such.
7. **Conclusion & Future Work**  
  -Summary of Findings 
  -Future Directions 


Show graph of data distibution before

### Then, in the methodology section, be explicit:

> “We include multiple datasets (e.g., ECG and image classification) and compare against centralized supervised, federated supervised, and naïve FixMatch + FedAvg baselines to ensure fair and comprehensive evaluation.”

*Main RQ*  
How does extending a naive FixMatch + FedAvg combination with adaptive thresholding and pseudo-label retention affect model performance and robustness under varying degrees of data heterogeneity and label scarcity?

*Sub RQs*  
- **RQ1.1:** How does the number of participating clients affect the convergence speed and final performance of the extended FixMatch + FedAvg model?
  
- **RQ1.2:** How does the model perform in low-round communication settings (e.g., ≤5 rounds), and what causes poor performance in such scenarios?

- **RQ2.1:** How does the data distribution (IID vs. non-IID) impact the performance of the federated semi-supervised model across different datasets (image classification)?
 
- **RQ2.2:** How robust is the model to variations in label availability across clients (label scarcity)?
 
- **RQ3.1:** How does the extended model generalize across different datasets and architectures compared to centralized and supervised federated baselines?


#### Different tests:
Just enable or disable non-iid and make 2 linecurves for both situations containning all models.

And a table with diff alpha values and i report mean & variances of the different runs (where each run has different alpha)



If using PTBXL dataset, then mention in thesis that we have to be careful of using validation set
- Add in discussion in thesis: In the STM: when the same instance gets multiple pseudolabels in the STM i replace it if the pseudolabels confidence is higher, & when in non-ordered mode i simple add latest one? Because with explo/explor its possible to get same instance if u hit the odds. And argue about the different optiosn & why we chose this way. And mention that other one is also justified.
- Also mention the adapative threshold choice
- Argue for normalization perhaps?
- Argue for our data splitting approach and that rather than ensuring X labels per class this is more realistic
- Explain that we possibly are going to weight ouir losses by their confidences.




### Tests Again

|Model|Label %|Lambda_u|Dirichlet α|Clients|Acc|F1|
|---|---|---|---|---|---|---|
|Centralized|0.05|N/A|N/A|N/A|||
|Fed Supervised|0.05|0.0|1.0|10|||
|Fed FixMatch|0.05|0.7|1.0|10|||
|Extended FixMatch|0.05|0.7|1.0|10|||

Run similar rows for 0.01 label fraction, α=0.05, clients=20, n_rounds, etc.

---

| Research Question                | Tests Needed                                                     |
| -------------------------------- | ---------------------------------------------------------------- |
| **RQ1.1: Clients & Convergence** | Vary client count; track convergence curves (acc/loss per round) |
| **RQ1.2: Low Comm. Settings**    | Vary communication rounds (5, 10, 20); track early accuracy      |
| **RQ2.1: IID vs Non-IID**        | Vary Dirichlet α for all models & test without non-iid           |
| **RQ2.2: Label Scarcity**        | Vary label fraction (0.01 to 0.2)                                |
| **RQ3.1: Generalization**        | Test across datasets (MNIST/CIFAR) and architectures             |
| **Ablation**                     | Turn off STM or adaptive threshold individually                  |
| **Baseline Performance**         | Compare centralized vs fed supervised vs your method             |


labeled_frac 0.01 -- 0.2 supervised(lambda_u = 0 vs optimal) for diff datasets
non-iid-alpha: 0.1 to 1 and then also iid all moedls AND (lambda_u = 0 vs optimal lambda) for diff datasets

nr of clients for all models loss curve for diff datasets (maybe a table)
nr of rounds for all models loss curve for diff datasets (maybe a table)

ablation: metrics/accuracy for all diff settings for diff datasets


lambda_u should be tuned by optuna

Standard values:
nr clients = 10 - 100
> Note: If testing the clients then it should be from 5-50

Nr_rounds usually 100
Pretraining maybe 1/5 of rounds? and 1 for 5?
If testing then go from 1-5-100

Local epochs: 5 is sstandard


| Factor                     | Values to Try                   | Notes                                         |
| -------------------------- | ------------------------------- | --------------------------------------------- |
| **Label Fraction**         | `0.01, 0.05, 0.1, 0.2`          | Test λ_u = 0 (supervised) vs optimal FixMatch |
| **Dirichlet α**            | `0.05, 0.1, 0.5, 1.0, ∞ (IID)`  | Cover full non-IID spectrum                   |
| **# Clients**              | `5, 10, 20, 50`                 | For convergence & RQ1.1                       |
| **# Communication Rounds** | `5, 10, 20, 50, 100`            | For RQ1.2 (early comm. vs late)               |
| **Pretraining Rounds**     | `1, 2, 4, 10, 20`               | *1/5 of rounds?*                              |
| **Lambda_u**               | `0 (supervised), 0.1, 0.7, 1.0` | Tune via Optuna if feasible                   |
| **Local Epochs**           | `5` (fixed)                     | As baseline                                   |
