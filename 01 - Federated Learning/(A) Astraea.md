#FL 

It counters accuracy degradation caused by imbalanced datasets through **two primary strategies**:

    1. **Global data distribution based data augmentation**.

    2. **Mediator based multi-client rescheduling**.

Basically they know that some clients have labels a-f and others g-z so they combine the training of those two using mediators. They also use data augmentation somewhere


**Data augmentation** in Astraea is used for the following:

• It's the **first strategy to alleviate global imbalance** in the training data. Global imbalance occurs when the collection of distributed data across all devices is class imbalanced.

• Before model training begins, the FL server calculates the necessary augmentations for each class based on the global data distribution.

• Clients then perform data augmentation in parallel, generating additional samples for minority classes.
![[Astraea.pdf]]