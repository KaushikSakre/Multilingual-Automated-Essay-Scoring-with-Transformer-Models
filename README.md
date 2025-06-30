# Multilingual-Essay-Scoring

![flowchart-1](https://github.com/user-attachments/assets/6c2a5517-cee7-4db3-ae7a-7aba55581898)

**OPTIMIZATION TECHNIQUES USED**

**1. Adam Optimizer:** Used Adam optimizer for efficient adaptive learning rate adjustment during training.

**2. Learning Rate Scheduling:** Implemented dynamic learning rate reduction using schedulers (e.g., ReduceLROnPlateau) to avoid plateaus.

**3. Early Stopping:** Training stops when validation performance doesn’t improve to prevent overfitting.

**4. Batch-wise Training:** Mini-batch training improves convergence speed and stability.

**5. Weight Decay (L2 Regularization):** Used in AdamW optimizer to prevent overfitting by penalizing large weights.

**6. Seed Initialization:** Fixed random seeds ensure reproducibility of model training and evaluation.


**RESULTS**

**1. QWK Scores of AES Model**
![image](https://github.com/user-attachments/assets/a78a97cb-03ce-4545-85be-d1fda7864207)

**2. RMSE Score of AES Model**

![image](https://github.com/user-attachments/assets/f661eda3-2f7e-4297-b692-c3844bee092c)

**3. R² Score of AES Model**
![image](https://github.com/user-attachments/assets/93d22cfd-4599-401a-b471-0ec3a38308e0)


