The lack of standardization in Non-IID measurement across different studies makes direct, cross-benchmark comparison challenging. However, based on the effectiveness reported within their specific FSSL (Federated Semi-Supervised Learning) and optimization benchmarks, a leaderboard prioritizing recent methods designed for handling severe heterogeneity can be constructed.

The most successful algorithms fall into two main categories: **Federated Optimization Methods** (which modify how global averages are calculated) and **Federated Semi-Supervised Learning (FSSL) Methods** (which address heterogeneity alongside label scarcity).

---

## Leaderboard of Top-Performing Methods in Non-IID Settings

The following leaderboard ranks methods based on their reported performance superiority in non-IID/heterogeneous environments, accounting for their publication timing.

| Rank  | Method                     | Publication Year | Key Non-IID/Heterogeneity Achievement                                                                                                                                                                                                  |
| :---- | :------------------------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **FedGGp**                 | 2025 (Projected) | **Superior performance in complex FSSL tasks (CIFAR10/100)** and scenarios with **high data heterogeneity** (Dir(0.1)), outperforming Twin-Sight and CBAFed.                                                                           |
| **2** | **FedCR**                  | 2025 (Projected) | **Outperforms all competitors** in terms of measurement performance across all tested IID and **non-IID FSSL cases** (CIFAR-10, CIFAR-100, SVHN), showing robustness against noise and heterogeneity.                                  |
| **3** | **Proposal** (Cha & Chang) | 2025 (Projected) | A dynamic optimization method achieving **significant accuracy improvements** and **faster convergence** than FedAvg, FedProx, and SCAFFOLD in both **mild and severe Non-IID scenarios** on MNIST and CIFAR-10.                       |
| **4** | **CBAFed**                 | 2023             | Achieves **superior performance** over all state-of-the-art FSSL competitors (RSCFed, Fed-Consist, FedIRM) on five benchmark datasets under heterogeneous data partition (Dirichlet $\gamma=0.8$).                                     |
| **5** | **RSCFed**                 | 2022             | **Significantly outperforms** older FSSL methods like FedIRM and Fed-Consist in the joint training of labeled and unlabeled clients under Non-IID data.                                                                                |
| **6** | **SCAFFOLD**               | 2020             | **Consistently the fastest** and most efficient convergence method among baseline optimization algorithms (SGD, FedAvg, FedProx) in severe non-convex and heterogeneous settings (0% similarity/sorted data).                          |
| **7** | **FedProx**                | 2020             | The **most commonly employed solution** for Non-IID data (used in $\sim 25%$ of surveyed papers). It achieves the best accuracy among baselines in label distribution skew and quantity skew cases in the NIID-Bench systematic study. |




Discussion of Non-IID Testing Methods

The core challenge in benchmarking these methods is the heterogeneity itself, as it is generally difficult to benchmark the Non-IID performance of different algorithms. However, the sources confirm that all of the above algorithms were tested rigorously in non-IID settings:

### 1. Federated Optimization Benchmarks (FedProx, SCAFFOLD, Proposal)

These methods were tested extensively on heterogeneous data distributions designed to simulate real-world non-IID conditions:

- **SCAFFOLD (2020) & FedProx (2020):** These algorithms were extensively evaluated alongside FedAvg and FedNova in a comprehensive benchmark study (NIID-Bench). They used **six different partitioning strategies** to cover various cases, including:
    
    - **Label Distribution Skew:** The most challenging scenario, where parties only have samples of a single class ($#C=1$) or the class distribution is skewed (e.g., $p_k \sim \text{Dir}(0.5)$).
    - **Feature Distribution Skew:** Where local data features differ across clients (e.g., noise-based or synthetic methods).
    - **Quantity Skew:** Where clients have different local data sizes ($q \sim \text{Dir}(0.5)$).
    - In these specific comparative tests, **SCAFFOLD** usually achieved the best accuracy in the **feature distribution skew** case, while **FedProx** usually achieved the best accuracy in **label distribution skew and quantity skew** cases.
- **Proposal (Cha & Chang) (2025):** This method explicitly addresses Non-IID data with **quantity skew**. It was tested using the **Dirichlet distribution** with concentration parameters $\beta = 0.5$ (mild Non-IID) and $\beta = 0.01$ (severe Non-IID) on MNIST and CIFAR-10.
    

### 2. Federated Semi-Supervised Learning (FSSL) Benchmarks (FedCR, FedGGp, CBAFed, RSCFed, FedMatch)

FSSL methods are always tested in non-IID environments, often focusing on **label skew** and the scarcity of labels:

- **FedMatch (2021):** Was validated on **Batch-NonIID** tasks, where the distribution of the number of instances per class was arbitrarily controlled to simulate class-imbalanced environments. This testing was performed under both the **Labels-at-Client** and the more challenging **Labels-at-Server** scenarios.
- **RSCFed (2022) & CBAFed (2023):** These methods focus on the FSSL scenario where **few clients are fully labeled while others are fully unlabeled**. They generate Non-IID data partitions in clients using a **Dirichlet distribution, $\text{Dir}(\gamma)$, where $\gamma=0.8$** is a common setting. The comparative results demonstrated RSCFed and particularly CBAFed's ability to handle high heterogeneity in this setting.
- **FedGGp (2025):** Was tested on varying degrees of data heterogeneity on CIFAR-10 by adjusting the Dirichlet factor $\delta \in {0.1, 0.3, 0.8}$. The **Dir($0.1$) setting represents the highest data heterogeneity** (most severe non-IID) where FedGGp demonstrated a distinct advantage.
- **FedCR (2025):** Demonstrated its effectiveness on **non-IID data partitions** for CIFAR-10, CIFAR-100, and SVHN datasets. FedCR's design is specifically tailored to address Non-IID data distributions by stabilizing client updates and incorporating noise-robust loss functions.