#FedSSL 

Self-SL not SSL

Says its impossible to use SSL or Semi-SL at clients due to data+device heterogeneity.

All the approaches require labeled data at one of the sides, this one is adaptable.

Everyone has data, server has more. No guarantees on the existence of labels for the data

Local clients are trained in supervised fashion, each own training set, but a global validation set.
Local clients have classification head of 100 classes
Loss = cross entropy, Adam = optimizer

Server uses contrastive approach,  labels are disregarded
Slight aug + original through network --> projected to latent feature space

SimClr approach:
Initially, a stochastic crop
of the image is procured, which is subsequently subjected
to a random horizontal flip. This is then followed by arbitrary
distortion of brightness, contrast, hue, and saturation
parameters, complemented by an optional grayscale transformation.
Subsequently, a Gaussian blur filter is administered
as the terminal step of the augmentation process. The
image is ultimately resized to dimensions of 224 × 224 and
undergoes normalization.

![[FedLID_Self-Supervised_Federated_Learning_for_Leveraging_Limited_Image_Data_ICCVW_2023_paper.pdf]]