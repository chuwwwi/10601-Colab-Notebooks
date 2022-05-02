import torch

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_HW5_data():
    """
    Loads in the data from HW 5.
    """
    n = len(".nb0.nb0_utils")
    D = np.loadtxt(fname=f"{__name__[:-n]}/nb0/data/small_train_data.csv",
                   delimiter=",", dtype=np.float64) # I know, not a good solution
    Y, X = D[:, 0], D[:, 1:]
    return torch.from_numpy(X).type(torch.float), torch.from_numpy(Y).type(torch.long)

def shuffle(X: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Shuffles the training data.
    """
    idx = np.random.permutation(np.arange(len(X), dtype=int))

    return X[idx], Y[idx]

def get_top_preds(yh: np.ndarray, L):
    Y = np.array(yh)

    sorted_pairs = reversed(sorted(zip(Y, L)))
    D = {k: v for v, k in sorted_pairs}
    return D

def visualize_model(model, X, Y, idx: int, k: int=5):
    """
    Visualizes the model.
    """
    L = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]
    model.eval()

    xi = X[idx]
    im = np.reshape(xi, (16, 8))

    yi = int(Y[idx].detach())

    yh = model(xi)
    D  = get_top_preds(yh.cpu().detach().numpy(), L)

    print(f"Input image (true label '{L[yi]}'):")
    plt.imshow(im, cmap="gray")
    plt.show()

    print(f"Model top {k} predictions:")
    for i, (l, v) in enumerate(D.items()):
        if i >= k: break
        print(f"  '{l}': {round(v*100, 1)}%")

def train(model, loss_fn, optimizer, X, Y, n_epochs) -> list:
    """
    Trains the model
    """
    losses = list()

    for epoch in tqdm(range(n_epochs), smoothing=0):

        total_loss = 0

        X, Y = shuffle(X, Y)
        for x_i, y_i in zip(X, Y):
            optimizer.zero_grad()

            y_hat = model(x_i)

            loss = loss_fn(y_hat, y_i)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        losses.append(total_loss)

    return losses
