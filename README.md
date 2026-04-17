# Data_augmentation_for_surgical_skill_classification

This thesis addresses the critical data bottleneck in automated surgical skill assessment by using Generative Adversarial Networks (TimeGAN) to artificially scale the highly restricted JIGSAWS dataset.
By filtering out erratic novice movements and exclusively generating expert and intermediate kinematic trajectories, the pipeline successfully synthesizes physically accurate and temporally valid surgical data.
This strategic augmentation significantly boosts the accuracy of downstream neural network classifiers, proving that synthetic data can effectively power objective, software-based surgical credentialing systems without requiring massive new clinical datasets.