#FedSSL


*Federated Semi-Supervised Learning (FSSL)*

ResNet is a decentralizeed supervised learning model, so it does not work for label scarcity


[[Pseudo-labeling]] is a fundamental concept in semi-supervised
learning (SSL) proposed by Lee et al. (2013).

[[Data augmentation]] helps create additional data based on data we already have.

Different types of noise can affect ECG (4.1.1)
Commonly used is the wavelet transform to reduce noise
--> Makes signal more periodic, can use fixed time intervals to segment signal

Have to segment ECG into multiple ECG beats

Time between R-peaks is a heartbeat cycle

(4.1.3) 2D ECG > 1D
Additionally, read this more in-depth when implementing/ looking for alternatives

So if I understand this correct:

Global model uses supervised learning to train model

Local models first use weak augmentation (with their model) to create certain pseudo-labels for the unlabelled data. If the prediction for the augmentation is above a threshold then that classification (pseudo-label) is used as its label.

Then once this is acquired: It does a strong augmentation on unlabelled data, passes it through the model and compares the labels. N then it tries to minimise the loss.

Tau is threshold for weak aug. label holding.
Lu is the unsupervised loss function.

& Then global model combines loss function with the local loss functions to update model.
 This is just for SSL i think?
 
But we cluster the clients based on their model weights?

We update weights of global model by finding the average of each cluster, and then sum those averages and divide that sum by the number of clusters.


The SSL Step 4 & the aggregation of the model, figure out how that works. & 5.2 the highlighted part, make sure to understand that

**Questions:** 
- What if we don't use K-means but smth else?
- They attribute their model as being robust to non-iid due to their clustering, is this true?

![[FedECG.pdf]]


From [NotebookLM:](https://notebooklm.google.com/notebook/1a38895a-267a-4174-8f11-eeabd01afcb8?authuser=1%3Fhl%3Den%3Fhl%3Den%3Fhl%3Den)

The *FedECG* framework employs a comprehensive pipeline for electrocardiogram (ECG) abnormalities prediction while addressing privacy protection and label scarcity. 

Here's a step-by-step explanation of their pipeline:
1. **ECG Data Processing**:
	- **Signal Denoising**: The raw ECG signals are first processed to eliminate noise, such as myoelectric noise, muscle artifacts, and instrument noise, which typically fall within the frequency range of 1 Hz3 . They utilize wavelet transform, which analyzes signal information in both time and frequency domains, effectively removing high-frequency noise while preserving essential ECG waveform features.

	- **Signal Segmentation**: The denoised continuous ECG signals are then segmented into individual ECG beats. This segmentation is based on the time interval between consecutive R wave peaks in the ECG waveform, representing a single heartbeat cycle. All extracted beats are ensured to have the same length, potentially by padding with zeros if a beat is too short, which is crucial for subsequent processing. The wavelet transform method makes the ECG signal more periodic, facilitating segmentation with a fixed time interval.

	- **Signal-to-Image Transformation**: Each segmented ECG beat is transformed into an image. Although 1D deep learning methods exist, the authors opted for a 2D representation, as prior research has shown it provides more accurate heartbeat classification. They use the Gramian Angular Fields (GAF) method to encode the 1D ECG signals as 2D images, preserving the time information of the time series. This involves: 
		- **Numerical Scaling**: The segmented ECG signals are normalized to the range using a Min-Max scaler.
		- **Coordinate Conversion**: The normalized value is encoded as an angular cosine, and the timestamp as a radius, mapping the rescaled ECG signal to polar coordinates 
		- **Trigonometric Transformation**: They employ Gramian Angular Summation Fields (GASF) and Gramian Angular Difference Fields (GADF) to capture temporal correlations by considering the trigonometric functions of samples across different timestamps.

2. **ECG Abnormalities Predicting Model**:
	- The framework utilizes a novel *ResNet-9 model* designed for predicting ECG abnormalities. This model is lightweight and consists of three main components:

		- **Convolutional Layer**: The model incorporates eight convolutional layers to extract key features from the input ECG images. A pooling layer is added after each convolutional layer, and batch normalization (BN) is applied before the pooling layer to improve training, increase convergence rate, and enhance robustness. They start with a 1D convolution of length 64 and continuously increase the length of subsequent convolutional layers to extract high-level spatial features. The kernel size for each convolutional layer is (3x3) with a stride of 114.
		- **Residual Block**: To address the vanishing or exploding gradient problem, ResNet introduces residual blocks with skip connections. These connections allow layers to be connected by skipping intermediate layers, preserving information from previous inputs and improving training efficiency15 . The residual block formula is given by: $b_i = F(a_i, w_i) + h(a_i)$ and $a_{i+1} = f(b_i)$, where $a_i$ and $a_{i+1}$ are the input and output vectors, $w_i$ is the convolution operation, $F$ is the residual function, $h(a_i)$ adjusts dimensions, and $f$ is the ReLU activation function15 .
		- Fully Connected Layer: A single fully connected layer acts as a classifier, mapping the extracted features to a 1D vector for classifying the five types of heartbeats. A softmax activation function is used in the final layer to output a probability distribution over the different classes.

3. **Semi-Supervised Learning**:
	- To address the issue of unlabeled ECG data on client devices, the FedECG framework incorporates semi-supervised learning techniques, specifically pseudo-labeling and data augmentation.
		- **Step 1 (Server)**: The central server, which has access to a small amount of labeled ECG data, performs regular supervised learning to pre-train the global model. The model is trained using the cross-entropy loss function on the labeled data.
		- **Step 2 (Clients)**: Clients, who possess only unlabeled data, use weak augmentation (e.g., standard rotation and shift operations on ECG images) and pass the augmented data through the received global model. *If the model's prediction score for an unlabeled data point is above a certain threshold, the predicted classification is used as a pseudo label for that data point.*
		`The augmentation ensures that minor variations don’t affect classification too much.
		- **Step 3 (Clients)**: Once pseudo labels are obtained, clients perform strong augmentation on the original unlabeled data. The goal is to calculate the cross-entropy loss between the generated pseudo labels and the features extracted from the strongly augmented unlabeled data.
		`This makes the model robust to real-world variations in ECG signals.`
		- **Step 4 (Server)**: The server then combines its supervised loss from the labeled data with the clients' unsupervised loss from the pseudo-labeled data to update the global model through backward gradient propagation of the total loss. The total loss is calculated as $Loss = \mathcal{L}_s + k\mathcal{L}_u$, where $\mathcal{L}_s$ is the supervised loss, $\mathcal{L}_u$ is the unsupervised loss, and $k$ is a hyperparameter to balance their contributions.

4. **Federated Learning**:
	- The FedECG framework employs federated learning to enable multiple clients to collaboratively train a global model without sharing their private ECG data. They improve upon the Federated Averaging (FedAvg) algorithm by incorporating a clustering step to address non-independent and identically distributed (non-i.i.d.) data.
		- **Model Training**: Unlike the standard FedAvg, their process begins with the server pre-training an initial global model using its labeled data. This pre-trained model is then sent to a subset of participating clients. The clients train local models using their unlabeled (and pseudo-labeled) data for a certain number of epochs and then send their updated local model parameters back to the server.
		- **Model Aggregation with Clustering**: To address the challenges posed by non-i.i.d. data, the server performs a model weights clustering algorithm. This involves using the K-means algorithm to group the received local model parameters based on the distribution of their weights. The server then aggregates the models within each cluster to create cluster-specific models. Finally, these cluster-specific models are further aggregated to form the new global model for the next communication round. This clustering aims to minimize the impact of diverse data distributions across clients and improve model convergence.
		- This entire process of client selection, local training, and global aggregation with clustering is repeated for a certain number of communication rounds until the global model converges.

By combining these steps, the FedECG framework aims to achieve accurate ECG abnormality prediction while preserving user privacy and effectively utilizing unlabeled data commonly found on smart ECG devices.



Notes: So both augmentations are added at the same time? Weak + strong to create 1 new view? Maybe for CL there is 2 diff views created. Here it is step by step




# SERVER SIDE: Initialize global model
initialize_model(global_model)

for round in range(NUM_ROUNDS):  # Federated learning rounds
    selected_clients = select_clients(CLIENTS, NUM_SELECTED)

    local_updates = []

    for client in selected_clients:
        local_model = global_model.copy()  # Send global model to client

        # CLIENT SIDE: Local Training with Pseudo-Labeling
        for epoch in range(NUM_EPOCHS):

            # Step 1: Generate Pseudo-Labels for Unlabeled Data
            for sample in client.unlabeled_data:
                weak_sample = apply_weak_augmentation(sample)
                predicted_label, confidence = local_model.predict(weak_sample)

                if confidence > CONFIDENCE_THRESHOLD:
                    pseudo_labels[sample] = predicted_label  # Accept pseudo-label
                else:
                    continue  # Discard low-confidence predictions

            # Step 2: Apply Strong Augmentation & Train on Pseudo-Labeled Data
            for sample, label in pseudo_labels.items():
                strong_sample = apply_strong_augmentation(sample)
                predicted_label = local_model.predict(strong_sample)
                
                loss = compute_loss(predicted_label, label)
                update_model(local_model, loss)

        # Step 3: Send Model Updates to Server
        local_updates.append(local_model.get_weights())

	# SERVER SIDE: Aggregate Local Models with Clustering
	clusters = cluster_models(local_updates, NUM_CLUSTERS)  # Apply K-means clustering

	clustered_updates = []
	for cluster in clusters:
	    cluster_avg = average_weights(cluster)  # Compute average for each cluster
	    clustered_updates.append(cluster_avg)

	global_model = federated_average(clustered_updates)  # Apply FedAvg to clustered models



**FedECG Vague:**
- How to cluster, flattening required or not?
- Which augmentations are performed
- Structure of their model (Pooling layer)
- Conversion to GAF + size of GAF
- Preprocessing (Segmentation is vague, do what is max size they use per image)
