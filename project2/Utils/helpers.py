import numpy as np

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, initial_w, model_name, max_iters=0, gamma=0, lambda_=0):
    """Return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    test_idx = k_indices[k]
    idx_tr = np.arange(len(k_indices))
    train_idx = k_indices[idx_tr != k]
    train_idx = train_idx.flatten()

    # form data with polynomial degree
    #x_tr = build_poly(x[train_idx], degree)
    #x_te = build_poly(x[test_idx], degree)
    x_tr = x[train_idx]
    x_te = x[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # applying the model 
    #initial_w = np.zeros(x_tr.shape[1])
    if(model_name == 'least_squares_GD'):
                w_star, mse = least_squares_GD(y_tr, x_tr, initial_w, max_iters, gamma)
    elif(model_name == 'least_squares_SGD'):
                w_star, mse = least_squares_SGD(y_tr, x_tr, initial_w, max_iters, gamma)
    elif(model_name == 'least_squares'):
                w_star, mse = least_squares(y_tr, x_tr)
    elif(model_name == 'ridge_regression'):
                w_star, mse = ridge_regression(y_tr, x_tr, lambda_)
    elif(model_name == 'logistic_regression'):
                w_star, mse = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
    elif(model_name == 'reg_logistic_regression'):
                w_star, mse = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
                

    # calculate the loss for train and test data
    if(model_name == 'logistic_regression' or model_name == 'reg_logistic_regression'):
        #loss_tr = compute_loss_LG(y_tr, x_tr, w_star)
        y_te_ = (1+y_te)/2
        loss_te = compute_loss_LG(y_te_, x_te, w_star)
        #print(loss_te)
        mod_pred = predict_labels(w_star, x_te, logistic = True)
        #mod_pred = (1+mod_pred)/2
        acc = calculate_accuracy(mod_pred, y_te)
    else:
        #loss_tr = compute_loss_MSE(y_tr, x_tr, w_star)
        loss_te = compute_loss_MSE(y_te, x_te, w_star)
        mod_pred = predict_labels(w_star, x_te)
        acc = calculate_accuracy(mod_pred, y_te)

    return acc, loss_te