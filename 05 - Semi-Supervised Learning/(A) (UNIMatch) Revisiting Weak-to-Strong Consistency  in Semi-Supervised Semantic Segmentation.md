#SEMI-SL 

`Basically copies fixmatch to semantic segmentation and says it needs manual tuning which isnt optimal.

They now force 2 strong perturbations to be close to the 1 weak one, similar to constrastive learning. And they also add a little nn.dropout2d on the features extracted of each weak image. They call this fp (feature perturbation) THESE 4 signals are then combined into the loss. They do CE(strong1, weak) + CE(strong2, weak) = loss_s 
CE(fp, weak) = loss_fp
loss_u = 1/2(loss_s+loss_fp)

applications:
Apart from semi-supervised classification, the methodol
ogy
in FixMatch has swept across a wide range of research
topics and achieved booming success, such as semantic seg
mentation
[19, 28, 81], object detection [42, 57, 66], unsuper
vised
domain adaptation [45], and action recognition [62,67].

They just show very big improvements like 10% over fixmatch in HEAVILY lower labeled scenarios.

![[Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf]]