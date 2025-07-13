#SEMI-SL 
### [A comprehensive overview of recent semi-supervised learning methods in Computer Vision](https://amitness.com/posts/semi-supervised-learning)

The idea is to train a model simultaneously on a batch of both labeled and unlabeled images. The model is trained on labeled images in usual supervised manner with a cross-entropy loss. The same model is used to get predictions for a batch of unlabeled images and the maximum confidence class is used as the pseudo-label. Then, cross-entropy loss is calculated by comparing model predictions and the pseudo-label for the unlabeled images .

![[99 - Assets/png/Semi-Supervised Learning in Computer Vision.png]]

This paper is amazing for an overview! Re-read if more info is needed on:

- [1. Self-Training](https://amitness.com/posts/semi-supervised-learning#self-training)
- [2. Consistency Regularization](https://amitness.com/posts/semi-supervised-learning#consistency-regularization)
    - [a. π-model](https://amitness.com/posts/semi-supervised-learning#a.-%CF%80-model)
    - [b. Temporal Ensembling](https://amitness.com/posts/semi-supervised-learning#b.-temporal-ensembling)
    - [c. Mean Teacher](https://amitness.com/posts/semi-supervised-learning#c.-mean-teacher)
    - [d. Virtual Adversarial Training](https://amitness.com/posts/semi-supervised-learning#d.-virtual-adversarial-training)
    - [e. Unsupervised Data Augmentation](https://amitness.com/posts/semi-supervised-learning#e.-unsupervised-data-augmentation) ==> Our focus
- [3. Hybrid Methods](https://amitness.com/posts/semi-supervised-learning#hybrid-methods)  ==> Important! MixMatch & FixMatch (Reread for simple explanation) FixMatch is what I'd focus on, other is too complicated.
- [Comparison of Methods](https://amitness.com/posts/semi-supervised-learning#comparison-of-methods)
- [Common Evaluation Datasets](https://amitness.com/posts/semi-supervised-learning#common-evaluation-datasets)
- [Conclusion](https://amitness.com/posts/semi-supervised-learning#conclusion)

![[Semi-Supervised Learning in Computer Vision 1.png]]

![[Semi-Supervised Learning in Computer Vision 2.png]]