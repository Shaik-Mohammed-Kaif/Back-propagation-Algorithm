# Back-propagation-Algorithm
This Algo to Process Back propagation Algorithm.

### **Neural Network Training Formulas**  

#### **1. Data Normalization**  
\[
X_{\text{normalized}} = \frac{X}{\max(X)}
\]
\[
y_{\text{normalized}} = \frac{y}{\max(y)}
\]

#### **2. Hidden Layer Input**  
\[
H_{\text{input}} = (X \times W_h) + B_h
\]

#### **3. Activation Function (Sigmoid)**  
\[
f(x) = \frac{1}{1 + e^{-x}}
\]
\[
H_{\text{output}} = f(H_{\text{input}})
\]

#### **4. Output Layer Input**  
\[
O_{\text{input}} = (H_{\text{output}} \times W_o) + B_o
\]

#### **5. Output Activation (Sigmoid)**  
\[
O_{\text{output}} = f(O_{\text{input}})
\]

#### **6. Error Calculation**  
\[
E = y - O_{\text{output}}
\]

#### **7. Gradient for Output Layer**  
\[
\Delta O = E \times O_{\text{output}} \times (1 - O_{\text{output}})
\]

#### **8. Gradient for Hidden Layer**  
\[
\Delta H = \Delta O \times W_o^T \times H_{\text{output}} \times (1 - H_{\text{output}})
\]

#### **9. Weight Updates (Gradient Descent)**  
\[
W_o = W_o + (\eta \times H_{\text{output}}^T \times \Delta O)
\]
\[
W_h = W_h + (\eta \times X^T \times \Delta H)
\]

#### **10. Bias Updates**  
\[
B_o = B_o + (\eta \times \sum \Delta O)
\]
\[
B_h = B_h + (\eta \times \sum \Delta H)
\]

#### **11. Mean Squared Error (MSE)**  
\[
\text{MSE} = \frac{1}{n} \sum (y - O_{\text{output}})^2
\]

These formulas are used iteratively until the error is minimized.
*******************************************************************************************

### **Neural Network Training**  
We will calculate the **first iteration** (epoch) of the **Neural Network** manually using the dataset.

---

## **Step 1: Given Dataset**
| **Hours of Study** | **Previous Exam Score (%)** | **Final Exam Score (%)** |
|------------------|----------------------|----------------------|
| 2                | 75                   | 80                   |
| 4                | 85                   | 90                   |
| 6                | 60                   | 70                   |
| 8                | 95                   | 96                   |
| 10               | 80                   | 85                   |

### **Step 1.1: Normalize Data**  
#### **Formula for Normalization**
\[
X_{\text{normalized}} = \frac{X}{\max(X)}
\]
\[
y_{\text{normalized}} = \frac{y}{100}
\]

#### **Applying Normalization**
\[
X_{\text{max}} = [10, 95]  \quad \text{(Max of each column)}
\]

| **Hours of Study (X1)** | **Previous Score (X2)** | **Final Score (y)** |
|----------------|----------------------|----------------------|
| \( \frac{2}{10} = 0.2 \)  | \( \frac{75}{95} = 0.789 \) | \( \frac{80}{100} = 0.8 \) |
| \( \frac{4}{10} = 0.4 \)  | \( \frac{85}{95} = 0.895 \) | \( \frac{90}{100} = 0.9 \) |
| \( \frac{6}{10} = 0.6 \)  | \( \frac{60}{95} = 0.632 \) | \( \frac{70}{100} = 0.7 \) |
| \( \frac{8}{10} = 0.8 \)  | \( \frac{95}{95} = 1.000 \) | \( \frac{96}{100} = 0.96 \) |
| \( \frac{10}{10} = 1.0 \)  | \( \frac{80}{95} = 0.842 \) | \( \frac{85}{100} = 0.85 \) |

---

## **Step 2: Initialize Weights and Biases**  

We randomly initialize:  

- **Hidden Layer Weights (Wh)** → **2×3 matrix** (2 inputs, 3 neurons)  
\[
Wh =
\begin{bmatrix}
0.3 & 0.5 & 0.2 \\
0.7 & 0.1 & 0.6
\end{bmatrix}
\]

- **Hidden Layer Bias (Bh)** → **1×3 matrix**  
\[
Bh = \begin{bmatrix} 0.1 & 0.2 & 0.3 \end{bmatrix}
\]

- **Output Layer Weights (Wo)** → **3×1 matrix** (3 neurons, 1 output)  
\[
Wo =
\begin{bmatrix}
0.4 \\
0.3 \\
0.9
\end{bmatrix}
\]

- **Output Bias (Bo)** → **1×1 matrix**  
\[
Bo = \begin{bmatrix} 0.5 \end{bmatrix}
\]

---

## **Step 3: Forward Propagation**  
We calculate values for **one row**:  
\[
X = \begin{bmatrix} 0.2 & 0.789 \end{bmatrix}
\]

### **Step 3.1: Compute Hidden Layer Input**  
\[
H_{\text{input}} = (X \times Wh) + Bh
\]

\[
=
\begin{bmatrix} 0.2 & 0.789 \end{bmatrix}
\begin{bmatrix}
0.3 & 0.5 & 0.2 \\
0.7 & 0.1 & 0.6
\end{bmatrix}
+
\begin{bmatrix} 0.1 & 0.2 & 0.3 \end{bmatrix}
\]

\[
=
\begin{bmatrix}
(0.2 \times 0.3) + (0.789 \times 0.7) & 
(0.2 \times 0.5) + (0.789 \times 0.1) & 
(0.2 \times 0.2) + (0.789 \times 0.6)
\end{bmatrix}
+
\begin{bmatrix} 0.1 & 0.2 & 0.3 \end{bmatrix}
\]

\[
=
\begin{bmatrix} 0.06 + 0.5523 & 0.1 + 0.0789 & 0.04 + 0.4734 \end{bmatrix} +
\begin{bmatrix} 0.1 & 0.2 & 0.3 \end{bmatrix}
\]

\[
= \begin{bmatrix} 0.7123 & 0.3789 & 0.8134 \end{bmatrix}
\]

---

### **Step 3.2: Apply Activation (Sigmoid)**
\[
H_{\text{output}} = \frac{1}{1 + e^{-H_{\text{input}}}}
\]

\[
H_{\text{output}} =
\begin{bmatrix}
\frac{1}{1 + e^{-0.7123}} & 
\frac{1}{1 + e^{-0.3789}} & 
\frac{1}{1 + e^{-0.8134}}
\end{bmatrix}
\]

\[
=
\begin{bmatrix} 0.6709 & 0.5936 & 0.6928 \end{bmatrix}
\]

---

### **Step 3.3: Compute Output Layer Input**  
\[
O_{\text{input}} = (H_{\text{output}} \times Wo) + Bo
\]

\[
=
\begin{bmatrix} 0.6709 & 0.5936 & 0.6928 \end{bmatrix}
\begin{bmatrix} 0.4 \\ 0.3 \\ 0.9 \end{bmatrix}
+
\begin{bmatrix} 0.5 \end{bmatrix}
\]

\[
=
(0.6709 \times 0.4) + (0.5936 \times 0.3) + (0.6928 \times 0.9) + 0.5
\]

\[
=
0.2684 + 0.1781 + 0.6235 + 0.5
\]

\[
=
1.57
\]

---

### **Step 3.4: Apply Sigmoid**
\[
O_{\text{output}} = \frac{1}{1 + e^{-1.57}}
\]

\[
=
\frac{1}{1 + 0.208}
\]

\[
= 0.828
\]

This is the predicted **normalized** final exam score.

---

## **Step 4: Error Calculation**
\[
E = y - O_{\text{output}}
\]

For the first data point:
\[
E = 0.8 - 0.828 = -0.028
\]

---

## **Step 5: Backpropagation (Weight Update)**
### **Step 5.1: Compute Gradient for Output Layer**
\[
\Delta O = E \times O_{\text{output}} \times (1 - O_{\text{output}})
\]

\[
= -0.028 \times 0.828 \times (1 - 0.828)
\]

\[
= -0.028 \times 0.828 \times 0.172
\]

\[
= -0.00398
\]

---

## **Final Results After Multiple Epochs**  
After repeating this process for all data points and multiple iterations, the weights adjust, and the predictions move closer to the actual values.

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
