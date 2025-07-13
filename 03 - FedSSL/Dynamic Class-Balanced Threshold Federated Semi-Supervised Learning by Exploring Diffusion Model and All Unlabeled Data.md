#FedSSL

Methods use random sampling & other techniques to address challenges of FSSL

What is this negative learning approach?

FSSL faces two main challenges:
- Limited Labeled Data: There's not much labeled data available on each device, making it hard to train accurate models.
- Non-Independent and Identically Distributed (non-IID) Data: The data on different devices can be very different (e.g., one hospital might have more scans of one type of disease than another). This "data heterogeneity" makes it difficult for a single global model to perform well on all devices

This paper introduces a new method called **DDRFed (Dynamic Class-Balanced Threshold Federated Semi-Supervised Learning)** to improve how machine learning models are trained across multiple devices (like phones or hospitals) when only some of the data on these devices is labeled. This type of learning is called **Federated Semi-Supervised Learning (FSSL)**.

**The Problem DDRFed Tries to Solve:**

FSSL faces two main challenges:

- **Limited Labeled Data:** There's not much labeled data available on each device, making it hard to train accurate models.
- **Non-Independent and Identically Distributed (non-IID) Data:** The data on different devices can be very different (e.g., one hospital might have more scans of one type of disease than another). This "data heterogeneity" makes it difficult for a single global model to perform well on all devices.

**The DDRFed Solution (Architecture):**

DDRFed tackles these challenges using three main components:

1. **Global Distribution Data Generation Module (GDGM) using a Diffusion Model:**
    
    - In a preliminary step, each device trains its own part of a **diffusion model** using its local data. Diffusion models are a type of generative model that can create new data similar to the data they've been trained on.
    - These locally trained diffusion models are sent to a central server, which combines them into a **global diffusion model**.
    - The **global diffusion model** then generates a new, synthetic dataset that reflects the overall data distribution across all devices. This generated dataset is then distributed back to all the devices.
    - **This helps with the non-IID problem** by giving each device access to data that is more representative of the global data distribution, without sharing the original private data.
2. **Dynamic Class-Balanced Threshold (DCBT):**
    
    - Each device uses this method to decide which of its unlabeled data samples it can confidently assign a "pseudo-label" to (acting as if it were a real label).
    - Instead of using a fixed confidence threshold for all classes, DCBT **dynamically adjusts the threshold for each class** based on the server's information about the global class distribution and how stable the model's predictions are for each class.
    - In the early stages of training, the thresholds are generally lower to utilize more unlabeled data. As the model learns, the thresholds for each class adapt to maintain a balance and prioritize high-confidence samples.
    - **This addresses the limited labeled data issue** by making better use of the available unlabeled data throughout the training process.
3. **Residual Class Negative Learning (RNL):**
    
    - This component focuses on utilizing the **low-confidence unlabeled samples** that were not assigned pseudo-labels by DCBT.
    - For these low-confidence samples, the method looks at how consistent the model's predictions are when the same sample is augmented in two different ways (one weak, one strong).
    - If the model's top prediction is consistent across augmentations, the method then encourages the model to **have low confidence in the other classes** (assigning "negative pseudo-labels" to these residual classes).
    - **This further enhances the model's learning** by teaching it what a sample is _not_, especially when real labeled data is scarce.

**How DDRFed Works (Simplified Training Process):**

1. **Pre-training (GDGM):** The diffusion model is trained in a federated manner, and a global synthetic dataset is generated and distributed.
2. **Federated Learning:** This process repeats for several rounds:
    - The server sends the current global model to each device.
    - Each device:
        - Combines its local data (both labeled and the generated synthetic data) with its unlabeled data.
        - Uses DCBT to select high-confidence unlabeled samples and assign pseudo-labels.
        - Uses RNL to learn from the low-confidence unlabeled samples.
        - Trains its local model using the labeled data, the pseudo-labeled data, and the negative learning from RNL.
        - Sends the updated local model back to the server.
    - The server aggregates the models received from the devices to create a new, improved global model. It also calculates updated global class distribution information for the next round.

**Key Results:**

The paper shows through experiments on two public image datasets (CIFAR-10 and Fashion-MNIST) that **DDRFed outperforms several state-of-the-art FSSL methods** under various challenging conditions, including different levels of data heterogeneity and different ratios of labeled to unlabeled clients.

- **Significant Accuracy Improvements:** DDRFed achieved higher accuracy and AUC (Area Under the ROC Curve, a measure of classification performance) compared to other methods. For example, it showed a **6.93% accuracy improvement on CIFAR-10** when the data distribution was relatively balanced.
- **Handles Non-IID Data Well:** The use of the diffusion model to generate globally representative data effectively mitigates the negative impact of non-IID data on model performance.
- **Efficient Use of Unlabeled Data:** DCBT and RNL work together to utilize both high-confidence and low-confidence unlabeled samples, leading to better model learning, especially when labeled data is scarce.
- **Ablation Studies:** The authors conducted ablation studies (systematically removing parts of DDRFed) which confirmed that **all three components (GDGM, DCBT, and RNL) contribute positively** to the overall performance.

In simpler terms, DDRFed is a smart way to train AI models across different devices with limited labeled data. It uses a data generation technique to make the data on each device more representative of the overall data, a dynamic threshold to confidently label some unlabeled data, and a negative learning approach to learn from the less confident data. The experimental results show that this combined approach leads to significant improvements in model accuracy compared to existing methods.



In the DDRFed framework, the training process at a specific client differs depending on whether the client possesses labeled data or is fully unlabeled. Both types of clients, however, benefit from the globally distributed synthetic dataset generated by the diffusion model (GDGM).

Here's a breakdown of how training works at a specific client:

**For a Labeled Client:**

1. **Data Preparation:** A labeled client possesses a small amount of labeled data and receives a portion of the globally generated dataset. In each communication round, the labeled client merges its local labeled data with the portion of the generated dataset.
2. **Thresholding (DCBT):** The client utilizes the Dynamic Class-Balanced Threshold (DCBT) method, which relies on the global class distribution and its standard deviation provided by the server. This dynamically adjusts the confidence threshold for each class. DCBT is used to distinguish between high-confidence and low-confidence samples within the client's **unlabeled data** (which in this case would be considered the generated data, as the local data is labeled).
3. **Training with High-Confidence Samples:** High-confidence samples (from the generated data, if treated as unlabeled) are used for training, and the local class distribution is recorded. The training for labeled clients incorporates the supervised loss ($L_s$) calculated on its labeled data, as well as unsupervised loss ($L_{u1}$) potentially from pseudo-labeled generated data, and the residual class negative learning loss ($L_{rnl}$). The supervised loss uses cross-entropy between the model's prediction on strongly augmented labeled data and the true label.
4. **Residual Class Negative Learning (RNL):** For low-confidence data (again, likely from the generated set if treated as unlabeled), the client applies Residual Class Negative Learning (RNL). This involves consistency optimization between different augmented versions of the data to estimate the top-$n$ accuracy and applying negative pseudo-label constraints to residual classes (classes outside the top-$n$). This helps the model learn from the low-confidence samples by pushing down the probabilities of incorrect classes.
5. **Local Model Update:** The client updates its local model based on the combined loss from the labeled data and the processed unlabeled (generated) data.
6. **Model and Statistics Upload:** After training, the labeled client uploads its updated model and its local class distribution information to the server.

**For a Fully Unlabeled Client:**

1. **Data Preparation:** A fully unlabeled client possesses only unlabeled local data and receives a portion of the globally generated dataset. In each round, it merges its local unlabeled data with the received generated data.
2. **Thresholding (DCBT):** Similar to the labeled client, the unlabeled client uses DCBT to determine high-confidence and low-confidence samples from its combined unlabeled local data and the generated data.
3. **Training with High-Confidence Samples:** High-confidence unlabeled samples are assigned pseudo-labels based on the model's predictions. The unsupervised loss ($L_{u1}$) is calculated using cross-entropy between the model's prediction on weakly augmented high-confidence samples and their assigned pseudo-labels. The local class distribution is also recorded based on these pseudo-labels.
4. **Residual Class Negative Learning (RNL):** The unlabeled client also applies RNL to the low-confidence samples from its local unlabeled data and the generated data. It computes the consistency between differently augmented versions of these samples and applies negative pseudo-labels to the residual classes based on the estimated top-$n$ accuracy. The training objective for fully unlabeled clients is solely the combination of the unsupervised loss ($L_{u1}$) and the residual class negative learning loss ($L_{rnl}$).
5. **Local Model Update:** The unlabeled client updates its local model based on the loss calculated from the pseudo-labeled high-confidence samples and the RNL applied to the low-confidence samples.
6. **Model and Statistics Upload:** The unlabeled client uploads its updated model and its local class distribution information (derived from the pseudo-labels) to the server.

In essence, both labeled and fully unlabeled clients in DDRFed follow a similar training procedure involving dynamic thresholding and residual negative learning on their available (possibly synthetic) unlabeled data. The key difference lies in the labeled clients also incorporating a supervised loss term based on their real labeled data during local model updates. The generated data from the global diffusion model helps to bridge the gap caused by non-IID distributions for both types of clients.


In the DDRFed framework, **strongly augmented samples are used in Residual Class Negative Learning (RNL) to evaluate the reliability of the weakly augmented sample's predicted class**. Here's why:

- **Assessing Model Confidence:** Strong augmentations introduce significant variations to the input data, making it more challenging for the model to make consistent predictions. If the model's prediction for a strongly augmented version of a low-confidence sample still places high probability on the class predicted by the weakly augmented version (even if it's not the absolute top-1 with high confidence), it suggests that there might be some underlying evidence for that class.
- **Estimating Top-$n$ Accuracy:** RNL aims to identify a set of the most likely classes (top-$n$) for a low-confidence sample by checking the consistency of predictions between weakly and strongly augmented versions. The strongly augmented version helps in determining which classes consistently receive relatively high probabilities, even under significant data perturbation.
- **Assigning Negative Pseudo-Labels:** Once the top-$n$ likely classes are estimated (informed by the strongly augmented predictions), RNL assigns **negative pseudo-labels** to the remaining classes (residual classes) for the weakly augmented version. This tells the model that the low-confidence sample _does not_ belong to these residual classes.

In simple terms, **Residual Class Negative Learning (RNL) is a way for the model to learn from its uncertain predictions on unlabeled data**. Instead of just ignoring the low-confidence samples, RNL tries to understand what the sample is _not_. Here's a simplified breakdown:

- **Look at an uncertain image in two ways:** Apply a little distortion (weak augmentation) and a lot of distortion (strong augmentation).
- **See what the model thinks in both cases:** If the model's top few guesses are somewhat consistent between the two distorted versions, those are probably the more likely classes.
- **Tell the model what it's _not_:** For the classes that the model doesn't consistently think are likely (the "residual" classes), RNL tells the model, "This uncertain image probably isn't one of these things".

This process helps the model refine its understanding of class boundaries, especially when there isn't much labeled data to learn from. By learning what a sample is _not_, the model becomes better at distinguishing between different categories, even for samples it's not initially very confident about.

![[Dynamic Class-Balanced Threshold Federated Semi-Supervised Learning.pdf]]