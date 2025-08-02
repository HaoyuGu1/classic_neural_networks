import torch
import numpy as np
import matplotlib.pyplot as plt
import model
import simulate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


def run_trial(trial_seed):
    device = "cpu" # the device on which the model is trained, can be "cpu", "cuda" or "mps" ("mps" is only available for mac with M-series chip)
    random_seed = trial_seed
    r2 = 0.5 # true r2 of the simulated data
    n = 1000 # simulation sample size
    dim = 112 # dimensions of the simulated images
    coord, true_beta, img_data, y = simulate.simulate_data(n, r2, dim, random_seed)

    # reshape image from 1d to 2d
    img_data_0_reshaped = img_data[0].reshape(n, dim, dim)
    img_data_1_reshaped = img_data[1].reshape(n, dim, dim)

    # stack image 1 and image 2 for each observation
    stacked_img = np.concatenate([img_data_0_reshaped, img_data_1_reshaped], axis = 1)
    stacked_img = stacked_img[:, np.newaxis, :, :]

    # create torch tensors
    y = y.reshape(-1, 1)
    y_tensor = torch.tensor(y, dtype = torch.float32).to(device)
    stacked_img_tensor = torch.tensor(stacked_img, dtype = torch.float32).to(device)

    # set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # split training and testing set and pass them into torch dataloaders
    X_train, X_test, y_train, y_test = train_test_split(stacked_img_tensor, y_tensor, test_size = 0.2, random_state = random_seed)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

    # training
    cnn = model.CNN2d().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        cnn.train()
        running_loss = 0.0
        y_pred, y_true = [], []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat_batch = cnn(X_batch)
            loss = criterion(y_hat_batch, y_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                y_pred.extend(y_hat_batch.detach().cpu().numpy().flatten())
                y_true.extend(y_batch.cpu().numpy().flatten())
        
        # print loss, training r2 and testing r2
        if (epoch + 1) % 5 == 0:
            train_r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Train R^2: {train_r2}")

        if (epoch + 1) % 5 == 0:
            cnn.eval()
            with torch.no_grad():
                y_hat_test = cnn(X_test).detach().cpu().numpy().flatten()
                test_r2 = np.corrcoef(y_test.cpu().numpy().flatten(), y_hat_test)[0, 1] ** 2
            print(f"Epoch {epoch + 1}, Test R^2: {test_r2}")

        # final evaluation per run
        cnn.eval()
        with torch.no_grad():
            y_train_pred = cnn(X_train).cpu().numpy().flatten()
            y_test_pred = cnn(X_test).cpu().numpy().flatten()

        y_train_true = y_train.cpu().numpy().flatten()
        y_test_true = y_test.cpu().numpy().flatten()

        train_mse = np.mean((y_train_true - y_train_pred) ** 2)
        test_mse = np.mean((y_test_true - y_test_pred) ** 2)
        train_r2 = np.corrcoef(y_train_true, y_train_pred)[0, 1] ** 2
        test_r2 = np.corrcoef(y_test_true, y_test_pred)[0, 1] ** 2

    return train_mse, test_mse, train_r2, test_r2


def main():
    num_trials = 10
    results = []
    for i in range(num_trials):
        print(f"=== Test {i} Running ===")
        result = run_trial(trial_seed=2025 + i)
        results.append(result)


    train_mse_list, test_mse_list, train_r2_list, test_r2_list = zip(*results)

    print("=== Summary over %.4f trials ===" % num_trials)
    print("Train MSE: mean = %.4f, std = %.4f" % (np.mean(train_mse_list), np.std(train_mse_list)))
    print("Test MSE:  mean = %.4f, std = %.4f" % (np.mean(test_mse_list), np.std(test_mse_list)))
    print("Train R²:  mean = %.4f, std = %.4f" % (np.mean(train_r2_list), np.std(train_r2_list)))
    print("Test R²:   mean = %.4f, std = %.4f" % (np.mean(test_r2_list), np.std(test_r2_list)))

if __name__ == "__main__":
    main()