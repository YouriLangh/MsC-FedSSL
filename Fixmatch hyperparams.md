| Hyperparameter                 | CIFAR-10 | CIFAR-100 | SVHN   | STL-10 |
| ------------------------------ | -------- | --------- | ------ | ------ |
| Confidence threshold (τ)       | 0.95     | 0.95      | 0.95   | 0.95   |
| Unlabeled loss weight (λ_u)    | 1        | 1         | 1      | 1      |
| Unlabeled batch ratio (uratio) | 7        | 7         | 7      | 7      |
| Batch size (B)                 | 64       | 64        | 64     | 64     |
| Learning rate (lr)             | 0.03     | 0.03      | 0.03   | 0.03   |
| Momentum                       | 0.9      | 0.9       | 0.9    | 0.9    |
| Nesterov                       | True     | True      | True   | True   |
| Weight decay                   | 0.0005   | 0.001     | 0.0005 | 0.0005 |


~~They weakly augment the labelled data too?~~
All the paddings and magnitudes on the augments will need to be altered depending on dataset.


Yes — labelled dataset reshuffled every time it’s repeated.