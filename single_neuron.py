import numpy as np

#input
X = [[0,0], [0,1], [1, 0], [1,1]]
#labels
y = [0, 0, 0, 1]

#slope
w = [0] * len(X[0])
#intercept
b = 0

learning_rate = 1
prev_loss = None
tolerance = 0.000001


for epoch in range(0, 1000):
    loss_total = 0
    intercept_total = 0
    slope_contribution = np.zeros(len(w))

    for x_val, label in zip(X, y):
        #score with multiple weights
        z = b + np.dot(w, x_val)
        #prob with sigmoid
        p_fraud = 1 / (1 + np.exp(-z))

        if label == 1:
            label_prob = p_fraud
        else:
            label_prob = 1 - p_fraud
        #error
        error = p_fraud - label
        #loss per row
        row_loss = -np.log(label_prob)

        #total loss
        loss_total += row_loss
        intercept_total += error
        for i, input_val in enumerate(x_val):
            slope_contribution[i] += error * input_val

    intercept_gradient = intercept_total / len(X)
    slope_gradient = slope_contribution / len(X)
    avg_loss = loss_total / len(X)

    if prev_loss != None:
        loss_change = abs(prev_loss - avg_loss)
        if loss_change < tolerance:
            break

    #new slope
    for i, gradient in enumerate(slope_gradient):
        w[i] -= learning_rate * gradient
    #new intercept
    b = b - learning_rate * intercept_gradient
    prev_loss = avg_loss

for x_val, label in zip(X, y):
    z = b + np.dot(w, x_val)
    p = 1 / (1 + np.exp(-z))
    prediction = int(p >= 0.5)
    print(f"{x_val}  probability={p:.4f}  predicted={prediction}  expected={label}")