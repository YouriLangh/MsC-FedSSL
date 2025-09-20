#FedSSL 

They consider both:
- case where clients have both labeled and unlabeled data (labels-at-client)
- case where the labeled data is only available at the server (labels-at-server)


We also reimplement FixMatch algorithms with strong augmentation as RandAugment (Cubuk et al.,
2019). For weak augmentation (filp-and-shift), however, as the performance has significantly dropped
when we apply the weak augmentation, we use original images rather than weakly augmenting
the images


![[(A) FedMatch.png]]


2. **Inter-Client Consistency Loss (Novel to FedMatch)** FedMatch introduces a **novel inter-client consistency loss** that extends the conventional approach. Instead of just comparing predictions from different augmentations of the _same_ input on a _single_ model, this new loss regularises the models learned at **multiple clients** to produce the same prediction for a given unlabeled instance.

    ◦ **Helper Agents**: This is achieved by selecting "helper agents" from the server. These helper agents are models from other clients, chosen based on their similarity to the current client's model.

    ◦ **Kullback-Leibler (KL) Divergence**: The inter-client consistency loss is defined using KL divergence, which measures the difference between probability distributions. Specifically, it minimises the KL divergence between the predictions of the client's local model (`pθl(y|u)`) and the predictions of the selected helper agents (`p∗ θhj (y|u)`) for an unlabeled instance `u`. The helper agents' parameters (`p∗ θhj`) are frozen during this client-side training.


PARAMETER DECOMPOSITION FOR DISJOINT LEARNING is just done bc of labels at server

![[Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning.pdf]]