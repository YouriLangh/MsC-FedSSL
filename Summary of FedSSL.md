**DDRFed & CBAFed** use dynamic thresholding + SSL clients
In our approach we will also keep these labels for X rounds.

Federated Semi-Supervised Learning (FSSL) is a machine learning paradigm that addresses the practical challenge of **training models collaboratively while protecting local data privacy, by leveraging both a small amount of labeled data and a large amount of unlabeled data**. This contrasts with traditional federated learning, which often makes the unrealistic assumption that all clients possess fully labeled data.

Key challenges that FSSL aims to solve include:

- **Label scarcity**: The high cost and expertise required for manual data labeling make it impractical for all clients to have fully labeled datasets.
- **Non-Independent and Identically Distributed (Non-IID) data**: Data distributions can vary significantly across different clients, which can severely degrade model accuracy and is a persistent challenge in federated learning.
- **Communication constraints**: Reducing the amount of data transmitted between clients and the server is crucial for efficiency.
- **Privacy concerns**: Centralizing data for machine learning can compromise personal privacy, which FL aims to avoid.
- **Miscalibration**: Semi-supervised learning (SSL) methods relying on pseudo-labels can often be miscalibrated, leading to overconfident but incorrect predictions.
- **Noisy labels**: Real-world labels can contain inaccuracies that, if unaddressed, can significantly degrade model performance.
- **Catastrophic forgetting**: Local models may forget previously learned knowledge when trained on non-IID data partitions, especially with batch-based pseudo-labeling.

FSSL scenarios are primarily categorized based on the location of labeled data:

- **Labels-at-Client**: Clients have a mix of both labeled and unlabeled data, or some clients are fully labeled while others are fully unlabeled.
- **Labels-at-Server**: Labeled data is exclusively available at the central server, while clients possess only unlabeled data.

Here is a list summarizing various Federated Semi-Supervised Learning approaches and what they aim to solve:

- **SemiFed**
    
    - **Core Mechanisms**: Unifies **consistency regularization** (using advanced data augmentation to enforce consistent output distributions under perturbations) and **pseudo-labeling** (generating artificial labels when multiple models from different clients produce high-confidence, agreeing predictions).
    - **Problems Solved**: Addresses the scenario where data samples of each client are **partially labeled**. Aims to improve model accuracy despite limited labeled examples, and is effective under both homogeneous and **heterogeneous (non-IID) data distribution** settings. It ensures robustness under non-IID data distributions. (It alters the confidence threshold per round apparently)
- **Benchmarking Semi-supervised Federated Learning (Zhang et al.)**
    
    - **Core Mechanisms**: Empirically evaluates various factors in FSSL. Finds that **Group Normalization (GN) combined with Consistency Regularization Loss (CRL)** yields better generalization. Proposes a **novel grouping-based model averaging method** to improve convergence efficiency and reduce communication when the number of users is large. Introduces a metric `R` (total variation distance) to measure **non-IIDness** in class distributions.
    - **Problems Solved**: Addresses the realistic scenario where **users have only unlabeled data and the server has a limited amount of labeled data**. Provides a thorough study on factors affecting test accuracy, including algorithm design, non-IIDness, communication period, number of users, and amount of labeled data in the server. Aims to establish a valid baseline for this more realistic SSFL setting.
- **FedECG**
    
    - **Core Mechanisms**: A framework for **ECG abnormality prediction** in a privacy-preserving manner. Involves signal pre-processing (denoising, segmentation, transformation), a modified **ResNet-9 model**, and incorporates **pseudo-labeling and data augmentation**. Uses a **novel model weights clustering algorithm** (based on K-means) on the server to improve model convergence and address **non-IID data**.
    - **Problems Solved**: Tackles three main issues in ECG monitoring: **non-IID data** (due to personal characteristic variations from diverse devices), **label scarcity** (ECG data is often unlabeled and requires professional annotation), and **personal privacy** (centralizing data compromises privacy).
- **FedMatch**
    
    - **Core Mechanisms**: Proposes an **inter-client consistency loss** (enforcing consistency between predictions across multiple models). Uses **parameter decomposition for disjoint learning** (separate parameters for supervised and unsupervised tasks) to preserve knowledge from labeled data and prevent forgetting. Reduces communication costs by sending only differences of parameters as sparse matrices.
    - **Problems Solved**: Addresses **deficiency of labels** in federated learning under two scenarios: **Labels-at-Client** (clients have partly labeled data) and **Labels-at-Server** (labeled data only at server, clients have unlabeled data). Mitigates inter-task interference between labeled and unlabeled data learning.
- **DDRFed (Dynamic Class-Balanced Threshold Federated Semi-Supervised Learning)**
    
    - **Core Mechanisms**: Integrates a **diffusion model** for globally distributed data generation (GDGM) to create a dataset conforming to global distribution, alleviating **client-side non-IID issues**. Employs **dynamic class-balanced thresholds (DCBT)** that adaptively adjust for each class based on model learning status to distinguish high and low-confidence samples, ensuring sufficient labeled data during training. Utilizes **Residual Class Negative Learning (RNL)** to leverage knowledge from low-confidence samples for model optimization.
    - **Problems Solved**: Effectively utilizes **all available data** (including low-confidence samples) and mitigates **training bias**. Addresses the major challenges of FSSL: **limited labeled data** and **non-IID data distribution**.
- **CBAFed (Class Balanced Adaptive Pseudo Labeling)**
    
    - **Core Mechanisms**: Uses a **fixed pseudo-labeling strategy** to overcome the **catastrophic forgetting problem** (preserving learned knowledge from previous rounds). Designs **class-balanced adaptive thresholds** based on the empirical distribution of all local training data to encourage a balanced training process for **imbalanced classes** under **non-IID settings**. Proposes a **residual weight connection** for local supervised training and global model aggregation to help models reach a better optimum, especially with imbalanced data and small training amounts.
    - **Problems Solved**: Addresses the scenario where few clients have fully labeled data, and others are fully unlabeled. Aims to fundamentally solve problems in FSSL influenced by imbalanced class distribution and non-IID settings.
- **RSCFed (Random Sampling Consensus Federated learning)**
    
    - **Core Mechanisms**: Instead of direct aggregation, it **distills multiple sub-consensus models** by **random sub-sampling over clients**. Introduces a **distance-reweighted model aggregation (DMA)** method to dynamically adjust weights for models closer to the sub-consensus model, enhancing robustness.
    - **Problems Solved**: Addresses **uneven reliability among models** from fully-labeled, fully-unlabeled, or partially labeled clients under the **Non-IID setting**. Aims to generalize FSSL to practical non-IID settings and avoids information leakage between clients.
- **FedCR (Federated Semi-supervised learning framework with Contrastive Representations)**
    
    - **Core Mechanisms**: Focuses on the **labels-at-server scenario**. Employs **instance-based contrastive representation learning** with a siamese-like network (online and target networks) to learn general features and prevent overfitting to noise. Uses a **privacy-preserving update strategy** (only online network parameters exchanged). Designs a **hybrid loss function** combining a new **contrastive regularization** (to filter information associated with wrong labels) and an **improved cross-entropy loss** (e.g., integrating mean absolute error) to enhance robustness against noisy labels and promote rapid convergence.
    - **Problems Solved**: Addresses the issue of **inaccuracy of real-world labels (noisy labels)** on the server. Mitigates the negative impact of mislabeled data during supervised learning. Improves performance under varying levels of label noise and in **non-IID environments**. Addresses limitations of conventional SSL where labeled data cannot promptly provide constraints.
- **SelfFed (Self-Supervised Federated Learning for Medical Images)**
    
    - **Core Mechanisms**: A two-phase framework: **pre-training** and **fine-tuning**. Pre-training performs **augmentative modeling** (e.g., using **Swin Transformer-based encoder with MAE**) to learn robust representations in a decentralized manner. Fine-tuning introduces a novel **contrastive network** and a **novel aggregation strategy**.
    - **Problems Solved**: Primarily addresses **data heterogeneity** (non-IID data distributions, feature distribution skew) and **label scarcity** in **medical imaging**. Aims to learn universal representations robust to uncurated data.
- **FedDS (Federated learning with entropy-weighted ensemble Distillation and Self-supervised learning)**
    
    - **Core Mechanisms**: Exploits **unlabeled data available on the server**. Uses **entropy-weighted ensemble distillation (EED)** to aggregate multiple client models into a server model, assigning higher weights to more accurate (lower entropy) predictions. Jointly leverages a separate **self-supervised learning (SSL) loss** (e.g., image rotation prediction) for improving generalization and stability of the server model.
    - **Problems Solved**: Effectively handles situations with **limited data per client and few clients (client data deficiency)**. Aims to suppress the contribution of unreliable clients and is robust to challenging scenarios with **noisy labels or Byzantine clients**. Mitigates accuracy degradation stemming from unreliable clients.