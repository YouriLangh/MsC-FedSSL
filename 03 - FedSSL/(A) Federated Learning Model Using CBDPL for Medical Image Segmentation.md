#FedSSL 

Adaptive thresholds? No.

Fully labeled &fully unlabeled clients? Perhaps the clients have mixed data actually.

But they merge their pseudolabeled data with labeled data i think?

They only add a subset of the generated pseudolabels per round to the dataset, where the size of the added subset increases per round.

Class Distribution Awareness: To prevent bias toward
dominant categories, class-balanced weighting is applied,
where the contribution of each class
in the pseudo-label set c is weighted by its inverse frequency

Very bad methodology explanation


idrk how the loss works, doesnt rlly matter ig?


1. **Initial Local Training on Labeled Data:** Each participating client site first trains a model using its **labeled data** to perform basic segmentation tasks and learn strong feature representations. This training uses a composite loss function, combining cross-entropy and Dice loss, and is performed locally for multiple epochs.
2. **Global Model Aggregation (ML):** After local training, each client shares its trained model weights with a central server, which aggregates them using the Federated Averaging (FedAvg) algorithm to create a global model (ML). This global model is then distributed back to the clients as an initial model for semi-supervised learning on unlabeled data.
3. **Pseudo-Label Generation for Unlabeled Data:**
    - The pre-trained global model (ML) is then used by each client to make predictions on its **unlabeled dataset**.
    - **Confidence-Based Dynamic Pseudo-Labeling (CBDPL)** is applied: instead of directly using all predictions, the mechanism assigns pseudo-labels **only if the prediction's softmax probability exceeds a predefined confidence threshold (τ)**, typically set at 0.8. This filtering step is crucial to minimise noisy labels and reduce the impact of incorrect predictions.
    - To further mitigate issues, strategies like **gradual label integration** (incorporating subsets of pseudo-labels with an adaptive threshold over rounds) and **class distribution awareness** (applying class-balanced weighting) are used.
4. **Combined Training and Refinement:** The high-confidence pseudo-labels are then merged with the client's original labeled dataset to create an enriched training set. The model is retrained using a hybrid loss function that integrates both supervised and semi-supervised objectives.
5. **Iterative Global Model Update (MU):** This process is iterative, progressively enhancing segmentation accuracy. After training on the combined dataset, each client updates its local model, shares it with the central server, and these are again aggregated via FedAvg into a new global model (MU), which integrates knowledge from both labeled and pseudo-labeled data. This updated model MU is then redistributed to clients for further rounds of federated training, progressively improving segmentation while maintaining privacy.
6. **Final Model Aggregation:** The central server finally aggregates the two global models, ML (from labeled data training) and MU (from pseudo-labeled-based training), to produce the final robust model.

This pipeline effectively leverages abundant unlabeled data to enhance model performance with limited labeled data, all while preserving patient privacy within the federated learning framework.


1. **Identify Class Frequencies:** For each class 'c' in the pseudo-label set, the system calculates how frequently that class appears (its 'count').

2. **Inverse Frequency Weighting:** It then assigns a weight (ωc) to each class. This weight is **inversely proportional to the square root of the class's frequency**. This means that classes that appear less frequently (minority classes) receive a higher weight, while classes that appear very often (dominant classes) receive a lower weight. A small constant (ϵ) is added to the denominator to avoid division by zero.

3. **Apply to Loss Function:** This calculated weight (ωc) is then used in the balanced loss function (`L(i)Balanced`) during training. Each class's contribution to the overall loss is adjusted by its corresponding weight.

![[Federated_Learning_model_using_CBDPL_for_Medical_I 1.pdf]]