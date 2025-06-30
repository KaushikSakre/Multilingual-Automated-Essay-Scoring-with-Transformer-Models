# Multilingual-Essay-Scoring

![flowchart-1](https://github.com/user-attachments/assets/6c2a5517-cee7-4db3-ae7a-7aba55581898)

**OPTIMIZATION TECHNIQUES USED**

**1. Adam Optimizer:** Used Adam optimizer for efficient adaptive learning rate adjustment during training.

**2. Learning Rate Scheduling:** Implemented dynamic learning rate reduction using schedulers (e.g., ReduceLROnPlateau) to avoid plateaus.

**3. Early Stopping:** Training stops when validation performance doesn’t improve to prevent overfitting.

**4. Batch-wise Training:** Mini-batch training improves convergence speed and stability.

**5. Weight Decay (L2 Regularization):** Used in AdamW optimizer to prevent overfitting by penalizing large weights.

**6. Seed Initialization:** Fixed random seeds ensure reproducibility of model training and evaluation.
