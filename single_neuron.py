import numpy as np

#input
X = [[0,0], [0,1], [1, 0], [1,1]]
#labels
y = [0, 0, 0, 1]

#slope
#before first epoch: w = [0,0]
#before epoch 1: w = [0,0]
w = [0] * len(X[0])
#intercept
#before first epoch: b = 0
#before epoch 1: b = -0.25
b = 0

learning_rate = 1
#before first epoch: prev_loss = None
#before epoch 1: prev_loss = 0.693
prev_loss = None
tolerance = 0.000001


for epoch in range(0, 1000):
    #first epoch (0)
    #first epoch: loss_total = 0
    #epoch (1) = loss_total = 0
    loss_total = 0
    #first epoch: intercept_total = 0
    #epoch (1): intercept_total = 0
    intercept_total = 0
    #first epoch: slope_contribution = [0., 0.]
    #epoch (1): slope_contribution = [0., 0.]
    slope_contribution = np.zeros(len(w))

    #first inner loop: x_val = [0,0], label = 0
    #inner loop iter2 (epoch0): x_val = [0,1], label = 0
    #inner loop iter3 (epoch 0): x_val = [1, 0], label = 0
    #inner loop iter4 (epoch0): x_val = [1,1], label = 1
    
    #inner loop iter1 (epoch1): x_val = [0, 0], label = 0
    #inner loop iter2 (epoch1): x_val = [0,1], label = 0
    #inner loop iter3 (epoch1): x_val = [1,0], label = 0
    #inner loop iter4 (epoch1): x_val = [1,1], label = 1
    for x_val, label in zip(X, y):
        #score with multiple weights
        #iter1 (inner loop): z = 0 + (0 * 0 + 0 * 0) = 0
        #iter2 (inner loop): z = 0 + (0 * 0 + 0 * 1) = 0
        #iter3 (inner loop): z = 0 + (0 * 1 + 0 * 0) = 0
        #iter3 (inner loop): z = 0 + (0 * 1 + 0 * 1) = 0
        
        #iter1 (inner loop epoch1): z = -0.25 + (0 * 0 + 0 * 0) = -0.25
        #iter2 (inner loop epoch1): z = -0.25 + (0 * 0 + 1 * 0) = -0.25
        #iter3 (inner loop epoch1): z = -0.25 + (1 * 0 + 0 * 0) = -0.25
        #iter3 (inner loop epoch1): z = -0.25 + (1 * 0 + 1 * 0) = -0.25
        z = b + np.dot(w, x_val)
        #prob with sigmoid
        #iter1 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        #iter2 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        #iter3 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        #iter4 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        
        #iter1 (inner loop epoch1): p_fraud = 1/(1 + e^(-(-0.25))) = 0.4378
        #iter2 (inner loop epoch1): p_fraud = 1/(1 + e^(-(-0.25))) = 0.4378
        #iter3 (inner loop epoch1): p_fraud = 1/(1 + e^(-(-0.25))) = 0.4378
        #iter3 (inner loop epoch1): p_fraud = 1/(1 + e^(-(-0.25))) = 0.4378
        p_fraud = 1 / (1 + np.exp(-z))
        
        #iter1 (inner loop): label = 0 so we skip to else
        #iter2 (inner loop): label = 0 so we skip to else
        #iter3 (inner loop): label = 0 so we skip to else
        #iter3 (inner loop): label = 1 so we go to if
        
        #iter1 (inner loop epoch1): label = 0 so we skip to else
        #iter2 (inner loop epoch1): label = 0 so we skip to else
        #iter3 (inner loop epoch1): label = 0 so we skip to else
        #iter4 (inner loop epoch1): label = 1 so we go to if
        if label == 1:
            #iter4 (inner loop): label_prob = 0.5 (p_fraud)
            
            #iter4 (inner loop epoch1):label_prob = 0.4378
            label_prob = p_fraud
        else:
        #iter1 (inner loop): label_prob = 1 - 0.5 = 0.5
        #iter2 (inner loop): label_prob = 1 - 0.5 = 0.5
        #iter3 (inner loop): label_prob = 1 - 0.5 = 0.5
        
        #iter1 (inner loop epoch1):label_prob = 1 - 0.4378 = 0.5622
        #iter2 (inner loop epoch1):label_prob = 1 - 0.4378 = 0.5622
        #iter3 (inner loop epoch1):label_prob = 1 - 0.4378 = 0.5622
            label_prob = 1 - p_fraud
        #error
        #iter1 (inner loop): error = 0.5 - 0 = 0.5
        #iter2 (inner loop): error = 0.5 - 0 = 0.5
        #iter3 (inner loop): error = 0.5 - 0 = 0.5
        #iter4 (inner loop): error = 0.5 - 1 = -0.5
        
        #iter1 (inner loop epoch1): error = 0.4378 - 0 = 0.4378
        #iter2 (inner loop epoch1): error = 0.4378 - 0 = 0.4378
        #iter3 (inner loop epoch1): error = 0.4378 - 0 = 0.4378
        #iter3 (inner loop epoch1): error = 0.4378 - 1 = -0.5622
        error = p_fraud - label
        #loss per row
        #iter1 (inner loop): row_loss = -log(0.5) = 0.693
        #iter2 (inner loop): row_loss = -log(0.5) = 0.693
        #iter3 (inner loop): row_loss = -log(0.5) = 0.693
        #iter4 (inner loop): row_loss = -log(0.5) = 0.693
        
        #iter1 (inner loop epoch1): row_loss = -log(0.5622) = 0.5759
        #iter2 (inner loop epoch1): row_loss = -log(0.5622) = 0.5759
        #iter3 (inner loop epoch1): row_loss = -log(0.5622) = 0.5759
        #iter4 (inner loop epoch1): row_loss = -log(0.4378) = 0.8260
        row_loss = -np.log(label_prob)

        #total loss
        #iter1 (inner loop) : loss_total = loss_total + row_loss = 0 + 0.693 = 0.693
        #iter2 (inner loop): loss_total = loss_total + row_loss = 0.693 + 0.693 = 1.386
        #iter3 (inner loop): loss_total = loss_total + row_loss = 1.386 + 0.693 = 2.079
        #iter4 (inner loop): loss_total = loss_total + row_loss = 2.079 + 0.693 = 2.772
        
        #iter1 (inner loop epoch1): loss_total = loss_total + row_loss = 0 + 0.5759 = 0.5759
        #iter2 (inner loop epoch1): loss_total = loss_total + row_loss =  0.5759 + 0.5759 = 1.1518
        #iter3 (inner loop epoch1): loss_total = loss_total + row_loss =  1.1518 + 0.5759 = 1.7277
        #iter4 (inner loop epoch1): loss_total = loss_total + row_loss =  1.7277 + 0.8260 = 2.5537
        loss_total += row_loss
        #iter1 (inner loop): intercept_total = intercept_total + error = 0 + 0.5 = 0.5
        #iter2 (inner loop): intercept_total = intercept_total + error = 0.5 + 0.5 = 1
        #iter3 (inner loop): intercept_total = intercept_total + erorr = 1 + 0.5 = 1.5
        #iter4 (inner loop): intercept_total = intercept_total + erorr = 1.5 + (-0.5) = 1
        
        #iter1 (inner loop epoch1): intercept_total = intercept_total + erorr = 0 + 0.4378 = 0.4378
        #iter2 (inner loop epoch1): intercept_total = intercept_total + erorr = 0.4378 + 0.4378 = 0.8756
        #iter3 (inner loop epoch1): intercept_total = intercept_total + erorr = 0.8756 + 0.4378 = 1.3134
        #iter4 (inner loop epoch1): intercept_total = intercept_total + erorr = 1.3134 + (-0.5622) = 0.7512
        intercept_total += error
        
        #iter1 (inner loop): x_val = [0, 0], error = 0.5
        #  nested i=0, input_val=0: slope_contribution[0] = 0.0 + (0.5 * 0) = 0.0
        #  nested i=1, input_val=0: slope_contribution[1] = 0.0 + (0.5 * 0) = 0.0
        #  result after iter1: slope_contribution = [0., 0.]
        #iter2 (inner loop): x_val = [0, 1], error = 0.5
        #  nested i=0, input_val=0: slope_contribution[0] = 0.0 + (0.5 * 0) = 0.0
        #  nested i=1, input_val=1: slope_contribution[1] = 0.0 + (0.5 * 1) = 0.5
        #  result after iter2: slope_contribution = [0., 0.5]
        #iter3 (inner loop): x_val = [1, 0], error = 0.5
        #  nested i=0, input_val=1: slope_contribution[0] = 0.0 + (0.5 * 1) = 0.5
        #  nested i=1, input_val=0: slope_contribution[1] = 0.0 + (0.5 * 0) = 0
        #  result after iter2: slope_contribution = [0.5, 0.5]
        #iter4 (inner loop): x_val = [1, 1], error =-0.5
        #  nested i=0, input_val=1: slope_contribution[0] = 0.5 + (-0.5 * 1) = 0
        #  nested i=1, input_val=1: slope_contribution[1] = 0.5 + (-0.5 * 1) = 0
        #  result after iter2: slope_contribution = [0, 0]
        
        #iter1 (inner loop epoch1): x_val = [0, 0], error = 0.4378
        #  nested i=0, input_val=0: slope_contribution[0] = 0.0 + (0.4378 * 0) = 0.0
        #  nested i=1, input_val=0: slope_contribution[1] = 0.0 + (0.4378 * 0) = 0.0
        #  result after iter1 (epoch1): slope_contribution = [0., 0.]
        #iter2 (inner loop epoch1): x_val = [0, 1], error = 0.4378
        #  nested i=0, input_val=0: slope_contribution[0] = 0.0 + (0.4378 * 0) = 0.0
        #  nested i=1, input_val=1: slope_contribution[1] = 0.0 + (0.4378 * 1) = 0.4378
        #  result after iter1 (epoch1): slope_contribution = [0., 0.4378]
        #iter3 (inner loop epoch1): x_val = [1, 0], error = 0.4378
        #  nested i=0, input_val=1: slope_contribution[0] = 0.0 + (0.4378 * 1) = 0.4378
        #  nested i=1, input_val=0: slope_contribution[1] = 0.0 + (0.4378 * 0) = 0
        #  result after iter1 (epoch1): slope_contribution = [0.4378, 0.4378]
        #iter4 (inner loop epoch1): x_val = [1, 1], error = -0.5622
        #  nested i=0, input_val=1: slope_contribution[0] = 0.4378 + (-0.5622 * 1) = -0.1244
        #  nested i=1, input_val=1: slope_contribution[1] = 0.4378 + (-0.5622 * 1) = -0.1244
        #  result after iter4 (epoch1): slope_contribution = [-0.1244, -0.1244]
        for i, input_val in enumerate(x_val):
            slope_contribution[i] += error * input_val

    #epoch 0: intercept_gradient = 1/4 = 0.25
    #epoch 1: intercept_gradient = 0.7512/4 = 0.1878
    intercept_gradient = intercept_total / len(X)
    #epoch 0: slope_gradient = [0,0]/4 = [0,0]
    #epoch 1: slope_gradient = [-0.1244, -0.1244]/4 = [-0.0311,-0.0311]
    slope_gradient = slope_contribution / len(X)
    #epoch 0: avg_loss = 2.772/4 = 0.693
    #epoch 1: avg_loss = 2.5537/4 = 0.638425
    avg_loss = loss_total / len(X)

    #epoch 0: prev_loss =  None so we skip
    #epoch 1: prev_loss =  0.693 so we enter if
    if prev_loss != None:
        #epoch1 loss_change = abs(0.693 - 0.638425) = 0.054575
        loss_change = abs(prev_loss - avg_loss)
        #epoch1 0.054575 is not less than tolerance so we dont break
        if loss_change < tolerance:
            break

    #new slope
    #epoch 0: w = [0, 0], slope_gradient = [0., 0.]
    #  nested i=0, gradient=0.0: w[0] = 0 - (1 * 0.0) = 0.0
    #  nested i=1, gradient=0.0: w[1] = 0 - (1 * 0.0) = 0.0
    #  result after epoch 0: w = [0.0, 0.0]
    #epoch 1: w = [0, 0], slope_gradient = [-0.0311, -0.0311]
    #  nested i=0, gradient=0.0: w[0] = 0 - (1 * -0.0311) = 0.0311
    #  nested i=1, gradient=0.0: w[1] = 0 - (1 * -0.0311) = 0.0311
    #  result after epoch 0: w = [0.0311, 0.0311]
    for i, gradient in enumerate(slope_gradient):
        w[i] -= learning_rate * gradient
    #new intercept
    #epoch 0: b = 0 - (1 * 0.25) = -0.25
    #epoch 1: b = -0.25 - (1 * 0.1878) = -0.4378
    b = b - learning_rate * intercept_gradient
    #epoch 0: prev_loss = 0.693
    #epoch 1: prev_loss = 0.638425
    prev_loss = avg_loss

for x_val, label in zip(X, y):
    z = b + np.dot(w, x_val)
    p = 1 / (1 + np.exp(-z))
    prediction = int(p >= 0.5)
    print(f"{x_val}  probability={p:.4f}  predicted={prediction}  expected={label}")