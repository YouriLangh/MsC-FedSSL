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
