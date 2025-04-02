#FL #Benchmark

This paper introduces a benchmark to evaluate FL & provides 2 datasets (one for ECHO and one for ECG)

FedTD and FedDTI simulate non-iid data by altering feature distributiosn & sample sizes

Evaluates their FL pipeline with others (7 others) and also verifies it with client and central (centralized data & only 1 part of data)
FedSSL actually sees some improvement but its never close to central SSL


The FedCVD benchmark is designed as the **first real-world federated learning benchmark for cardiovascular disease (CVD) data**. It aims to address the limitations of existing FL benchmarks by using **naturally partitioned data from seven different medical institutions** instead of simulated partitions.

Here's a short explanation of how it works:

- **Data Sources:** FedCVD utilizes real-world CVD data collected from **seven medical institutions**. This data is naturally scattered across these institutions due to privacy concerns and regulations.
- **Tasks:** The benchmark includes **two main tasks** based on this data:
    - **Electrocardiogram (ECG) classification (Fed-ECG):** This task involves multi-label classification of ECG signals from four distinct datasets sourced from geographically diverse regions. The goal is to diagnose various cardiac conditions.
    - **Echocardiogram (ECHO) segmentation (Fed-ECHO):** This task focuses on the segmentation of cardiac structures in echocardiogram images derived from three different sources. The goal is to automatically segment regions like the left ventricular endocardium, epicardium, and left atrium wall.
- **Challenges:** FedCVD is specifically designed to highlight and challenge FL algorithms with three critical traits of real-world CVD data:
    - **Non-IID (Non-independently and identically distributed) data:** This includes variations in data features (e.g., imaging quality) and label distributions (e.g., disease prevalence) across institutions.
    - **Long-tail distribution of labels:** In the ECG classification task, some cardiac conditions are much more prevalent than others, creating an imbalanced label distribution that challenges model performance on less frequent conditions.
    - **Label incompleteness:** In the ECHO segmentation task, different institutions have varying annotation capabilities, leading to incomplete labeling of certain regions in the images.
- **Evaluation:** The benchmark provides **local and global evaluation sets** to assess the performance of FL algorithms. It uses metrics relevant to each task, such as micro-F1 and mean average precision (mAP) for ECG classification, and Dice similarity index and Hausdorff distance for ECHO segmentation.
- **Open Source:** FedCVD is **open-source and publicly available** at [https://github.com/SMILELab-FL/FedCVD](https://github.com/SMILELab-FL/FedCVD), allowing researchers to use it for developing and evaluating new FL algorithms in the healthcare domain.

In essence, FedCVD provides a realistic and challenging platform for evaluating federated learning techniques on real-world cardiovascular disease data with inherent heterogeneity and data quality issues.


![[FedCVD.pdf]]