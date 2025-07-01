#FedSSL 

This paper offers a good summary of the state of the art and what approaches fail at what/should be revised.

*Summary of training after pdf.*

![[Federated semi-supervised learning with contrastive representations against.pdf]]

The Federated Semi-supervised learning framework with Contrastive Representations (FedCR) employs an iterative process involving both server-side and client-side training to train a centralized model. This approach is designed to leverage both labeled data (at the server) and unlabeled data (at clients) while addressing challenges like label noise and data privacy.

### Server-Side Training

1. **Pre-training Phase:**
    
    - The training process in FedCR initially emphasizes **server-side operations**. In the first round, a global model, comprising a classification module ($\sigma$) and a contrastive representation module (with an online network $\theta$ and a target network $\xi$), is **pre-trained on the server using available noisy labels**.
    - This pre-training is beneficial for faster convergence and improved classification accuracy. Image augmentation techniques (like random flipping and cropping) are used to enhance the dataset for supervised learning on the server.
2. **Subsequent Server Training Rounds:**
    
    - In subsequent rounds of federated semi-supervised learning, after clients have performed their local updates and uploaded parameters, the server performs its own training step.
    - The server updates the global model (${\theta, \xi, \sigma}$) utilizing the noisy labeled data it possesses.
    - **Loss Functions:**
        - When labels are **clean**, the standard **cross-entropy loss ($\mathcal{L}_{ce}$)** is used.
        - However, to address **label noise**, FedCR employs an **improved generalized cross-entropy loss ($\mathcal{L}_{gce}$)**. This function combines the benefits of mean absolute error loss (robust to noise) and cross-entropy loss, ensuring impartial treatment during training and bolstering robustness against noise.
        - A **contrastive regularization function ($\mathcal{L}_{cr}$)** is also applied to preserve information related to true labels while filtering out information associated with wrong labels. This function helps to minimize the distance between images of the same class while pushing images from different classes apart, even with noisy labels.
    - This collaborative effect of self-supervised learning and supervised learning helps prevent the model from memorizing erroneous labels.

### Client-Side Training (Local Model Update)

1. **Broadcasting Global Model:**
    
    - For each communication round, the server **broadcasts the global online network's parameters ($\theta$)** to a randomly selected subset of participating clients (denoted as $K$ clients).
    - Each selected client initializes its local contrastive networks using these received parameters.
2. **Unsupervised Local Training:**
    
    - Clients primarily engage in **unsupervised learning** using their entirely unlabeled local data.
    - The **contrastive representation module** (consisting of an online network $\theta$ and a target network $\xi$) is central to this. Given an input image, two distinct augmented views are generated. The online network processes one view, and the target network processes the other.
    - The core objective here is to **reduce the distance between representations of different augmented views of positive samples**. The target network's parameters ($\xi$) are updated locally via an **Exponential Moving Average (EMA)** of the online network's parameters ($\theta$), which helps stabilize training and prevent representation collapse. A **stop-gradient strategy** is applied to the target network to ensure only the online network is directly optimized, stabilizing the learning process.
    - The clients calculate a local loss, which is specifically the **contrastive regularization function ($\mathcal{L}_{cr}$)**, to learn general features and avoid memorizing noise.
3. **Uploading Local Parameters:**
    
    - After completing their local unsupervised training, clients **upload only their updated local online network parameters ($\theta_k$) back to the server**.
    - The local target network parameters are discarded. This update strategy is crucial for **ensuring privacy and security** by preventing potential leakage of client information. It also helps conserve computing resources and mitigates the challenge of non-Independently Identically Distributed (non-IID) client data.

### Global Model Aggregation

1. **Receiving Local Updates:**
    
    - The server receives the online network parameters from the selected $K$ clients.
2. **Weighted Averaging:**
    
    - The server then performs **global aggregation** using a **weighted averaging approach** to refresh the global online network ($\bar{\theta}$). The weights are based on the number of samples each client possesses.
    - After aggregating the online network parameters, the server integrates these with other relevant global model parameters (including $\xi$ and $\sigma$) residing on the server side.

These steps—broadcasting, local client updates, global aggregation, and server training—are **repeated for multiple communication rounds**, allowing the model to collaboratively learn from distributed datasets while preserving client data privacy and improving robustness against noisy labels.