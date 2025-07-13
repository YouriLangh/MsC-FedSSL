#SSL


**SimCLR:**
1. Create 2 different augmentations of x
2. Feed these augmented samples to an encoder
3. This encoder learns representations h
4. These can be passed through a neural block (head) which converts this to projections (z) in a new latent space
5. Maximize agreement between these

**MoCo:**
MocoV2 outperformed SimCLR.
Learn good representations by looking up a large dictionary rich of negative examples
	--> less memory burden
The heads have different weights.
The second branch adds samples to the queue (dictionary)?

**Bootstrap Your Own Latent:**
Augment data 2x (as SimCLR) and use the embedding of 1 to try and predict the other.
--> Possible that only using positive pairs leads to collapsing solution (always output same constant)

<span style="color:rgb(255, 0, 0)">Various papers are discussed in  Section VIa wrt evaluation of different CL approaches & which pretext tasks are better.</span>

![[Applications of Self-Supervised Learning to Biomedical Signals. A Survey (2023).png]]

Use known variations of the PQRST complex in ECG analysis that distinguish between healthy and abnormal signals as extra information during pre-training.

![[Applications_of_Self-Supervised_Learning_to_Biomed.pdf]]