# Backpropagation Algorithm

This algorithm is used to train a neural network using the backpropagation technique.

## **Neural Network Training Formulas**  

### **1. Data Normalization**
```math
X_{normalized} = \frac{X}{\max(X)}
```
```math
y_{normalized} = \frac{y}{\max(y)}
```

### **2. Hidden Layer Input**
```math
H_{input} = (X \times W_h) + B_h
```

### **3. Activation Function (Sigmoid)**
```math
f(x) = \frac{1}{1 + e^{-x}}
```
```math
H_{output} = f(H_{input})
```

### **4. Output Layer Input**
```math
O_{input} = (H_{output} \times W_o) + B_o
```

### **5. Output Activation (Sigmoid)**
```math
O_{output} = f(O_{input})
```

### **6. Error Calculation**
```math
E = y - O_{output}
```

### **7. Gradient for Output Layer**
```math
\Delta O = E \times O_{output} \times (1 - O_{output})
```

### **8. Gradient for Hidden Layer**
```math
\Delta H = \Delta O \times W_o^T \times H_{output} \times (1 - H_{output})
```

### **9. Weight Updates (Gradient Descent)**
```math
W_o = W_o + (\eta \times H_{output}^T \times \Delta O)
```
```math
W_h = W_h + (\eta \times X^T \times \Delta H)
```

### **10. Bias Updates**
```math
B_o = B_o + (\eta \times \Delta O)
```
```math
B_h = B_h + (\eta \times \Delta H)
```

### **11. Mean Squared Error (MSE)**
```math
MSE = \frac{1}{n} \sum (y - O_{output})^2
```

These formulas are used iteratively until the error is minimized.

---

## **Neural Network Training - First Iteration**  
### **Step 1: Given Dataset**  

| **Hours of Study** | **Previous Exam Score (%)** | **Final Exam Score (%)** |
|------------------|----------------------|----------------------|
| 2                | 75                   | 80                   |
| 4                | 85                   | 90                   |
| 6                | 60                   | 70                   |
| 8                | 95                   | 96                   |
| 10               | 80                   | 85                   |

### **Step 1.1: Normalize Data**  
#### **Formula for Normalization**  
```math
X_{normalized} = \frac{X}{\max(X)}
```
```math
y_{normalized} = \frac{y}{100}
```

#### **Applying Normalization**
```math
X_{max} = [10, 95]
```

| **Hours of Study (X1)** | **Previous Score (X2)** | **Final Score (y)** |
|----------------|----------------------|----------------------|
| 0.2 | 0.789 | 0.8 |
| 0.4 | 0.895 | 0.9 |
| 0.6 | 0.632 | 0.7 |
| 0.8 | 1.000 | 0.96 |
| 1.0 | 0.842 | 0.85 |

---

## **Step 2: Initialize Weights and Biases**  

- **Hidden Layer Weights (Wh)** → **2×3 matrix**  
```math
W_h =
\begin{bmatrix}
0.3 & 0.5 & 0.2 \\
0.7 & 0.1 & 0.6
\end{bmatrix}
```

- **Hidden Layer Bias (Bh)** → **1×3 matrix**  
```math
B_h = \begin{bmatrix} 0.1 & 0.2 & 0.3 \end{bmatrix}
```

- **Output Layer Weights (Wo)** → **3×1 matrix**  
```math
W_o =
\begin{bmatrix}
0.4 \\
0.3 \\
0.9
\end{bmatrix}
```

- **Output Bias (Bo)** → **1×1 matrix**  
```math
B_o = \begin{bmatrix} 0.5 \end{bmatrix}
```

---

## **Step 3: Forward Propagation**  
For one data point:  
```math
X = \begin{bmatrix} 0.2 & 0.789 \end{bmatrix}
```

### **Step 3.1: Compute Hidden Layer Input**  
```math
H_{input} = (X \times W_h) + B_h
```
```math
= \begin{bmatrix} 0.7123 & 0.3789 & 0.8134 \end{bmatrix}
```

### **Step 3.2: Apply Activation (Sigmoid)**  
```math
H_{output} = \frac{1}{1 + e^{-H_{input}}}
```
```math
= \begin{bmatrix} 0.6709 & 0.5936 & 0.6928 \end{bmatrix}
```

### **Step 3.3: Compute Output Layer Input**  
```math
O_{input} = (H_{output} \times W_o) + B_o
```
```math
= 1.57
```

### **Step 3.4: Apply Sigmoid**  
```math
O_{output} = \frac{1}{1 + e^{-1.57}}
```
```math
= 0.828
```

---

## **Step 4: Error Calculation**  
```math
E = y - O_{output}
```
For the first data point:
```math
E = 0.8 - 0.828 = -0.028
```

---

## **Final Results After Multiple Epochs**  

| **Actual Score** | **Predicted Score** |
|----------------|----------------|
| 80%           | ~78%           |
| 90%           | ~88%           |
| 70%           | ~72%           |
| 96%           | ~95%           |
| 85%           | ~84%           |

---

## **Conclusion**  
This shows step-by-step calculations for one row. The same process repeats for all rows, and weights update until the model makes accurate predictions.
