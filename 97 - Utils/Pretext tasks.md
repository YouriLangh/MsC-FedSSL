#utils 

Tasks which use automatically generated pseudolabels based on attributes of the data to train a model in learning representations of the data. Learned representation can be used for further downstream tasks

For a pretext task in contrastive learning, the original
image acts as an anchor, its augmented(transformed) version acts as a positive sample,
and the rest of the images in the batch or in the training data act as negative samples.

Examples: 
- Color transformation
- Geometric transformation
- Jigsaw puzzle
- etc...
![[Pretext tasks - Color transformation.png]]