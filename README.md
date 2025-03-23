Backpropagation Algorithm

Neural Network Training Formulas

1. Data Normalization





2. Hidden Layer Input



3. Activation Function (Sigmoid)





4. Output Layer Input



5. Output Activation (Sigmoid)



6. Error Calculation



7. Gradient for Output Layer



8. Gradient for Hidden Layer



9. Weight Updates (Gradient Descent)





10. Bias Updates





11. Mean Squared Error (MSE)



Neural Network Training (First Iteration Calculation)

Step 1: Given Dataset

Hours of Study

Previous Exam Score (%)

Final Exam Score (%)

2

75

80

4

85

90

6

60

70

8

95

96

10

80

85

Step 1.1: Normalize Data

Formula for Normalization





Applying Normalization



X1 (Hours of Study)

X2 (Previous Score)

y (Final Score)

0.2

0.789

0.8

0.4

0.895

0.9

0.6

0.632

0.7

0.8

1.000

0.96

1.0

0.842

0.85

Step 2: Initialize Weights and Biases

Hidden Layer Weights (Wh) (2×3 matrix)



Hidden Layer Bias (Bh) (1×3 matrix)



Output Layer Weights (Wo) (3×1 matrix)



Output Bias (Bo) (1×1 matrix)



Step 3: Forward Propagation (For First Row)

Step 3.1: Compute Hidden Layer Input



Step 3.2: Apply Activation (Sigmoid)



Step 3.3: Compute Output Layer Input



Step 3.4: Apply Sigmoid Activation



Step 4: Error Calculation



Step 5: Backpropagation (Weight Update)

Step 5.1: Compute Gradient for Output Layer



Step 5.2: Compute Gradient for Hidden Layer



Step 5.3: Update Weights and Biases









Final Results After Multiple Epochs

Actual Score

Predicted Score

80%

~78%

90%

~88%

70%

~72%

96%

~95%

85%

~84%

Conclusion

This step-by-step breakdown demonstrates how Backpropagation updates weights iteratively until the model achieves accurate predictions.

