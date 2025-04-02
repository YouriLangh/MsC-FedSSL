#utils 
Data augmentation is a popular machine learning approach to
address the problem of overfitting and limited labeled data.
This technique produces new data by fine-tuning the existing data or adding noise, ensuring that the labels of the new data and the original data are the same. This
approach can create additional data to compensate for the shortage
of labeled data, which can improve model performance.

There are two primary types of data augmentation methods:
weak augmentation and strong augmentation.

- Weak augmentation involves processing the data without changing the content of the data itself. Common weak augmentation methods include panning, rotating, flipping, cropping, etc. (Bin et al., 2020)
`Goal is to obtain more training data`
- Strong augmentation adds noise or perturbation to the data to change the content of the data itself. However, it can be useful in some instances, such as when the model must learn to recognize objects in a cluttered or noisy environment.
