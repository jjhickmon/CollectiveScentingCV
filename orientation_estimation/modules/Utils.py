import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Prep data & splitting dataset
# --------------------------------------------------------
def convert_X_for_resnet(X):
    X_np = X.numpy()
    X_np = np.tile(X_np, (1,3,1,1))
    return torch.tensor(X_np)

def shuffle_batch(batch, labels):
    idxs = np.array(range(len(batch)))
    np.random.shuffle(idxs)
    return batch[idxs], labels[idxs]

def train_test_split(bee_data, test_split=0.2):
    train_split = 1 - test_split

    # Get df
    df = bee_data.data_df

    # Separate out the scenting vs non-scenting indices
    scenting_indices = np.where(df['classification'] == 'scenting')[0]
    non_scenting_indices = np.where(df['classification'] == 'non_scenting')[0]

    # Shuffle indices
    np.random.shuffle(scenting_indices)
    np.random.shuffle(non_scenting_indices)

    # Get number of scenting, number of non scenting
    num_train_scenting = int(len(scenting_indices) * train_split)
    num_train_nonscenting = int(len(non_scenting_indices) * train_split)

    # Get train/test scenting indices
    train_scenting_indices = scenting_indices[:num_train_scenting]
    test_scenting_indices = scenting_indices[num_train_scenting:]

    # Get train/test non scenting indices
    train_non_scenting_indices = non_scenting_indices[:num_train_nonscenting]
    test_non_scenting_indices = non_scenting_indices[num_train_nonscenting:]

    # Putting together scenting and non-scenting indices
    train_idxs = np.concatenate((train_scenting_indices, train_non_scenting_indices), axis=0)
    test_idxs = np.concatenate((test_scenting_indices, test_non_scenting_indices), axis=0)

    # Error checking
    # ============================================================
    assert set(train_idxs).intersection(set(test_idxs)) == set(), 'Train and Test indices are not mutually exclusive!'
    assert len(train_idxs) + len(test_idxs) == len(df), 'Train and test idxs do not add up to full num of data idxs'

    train_idx_error_str = "Train idxs exceed len of data. "
    train_idx_error_str += f"max(train_idxs): {max(train_idxs)} -- Len data: {len(df)}"
    assert max(train_idxs) < len(df), train_idx_error_str
    assert max(test_idxs) < len(df), 'Test idxs exceed len of data'
    # ============================================================
    return train_idxs, test_idxs

# Evaluation functions
# --------------------------------------------------------
def get_labels(bee_data, preds):
    pred_strings = [bee_data.int_to_label[int(pred)] for pred in preds]
    return pred_strings

def get_prediction(logits, threshold=0.5):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds

def get_accuracy(y, preds):
    # accuracy = int(sum(y == preds)) / len(y) * 100
    # accuracy = sum(np.array(preds == y)) / len(y) * 100
    correct_sum = (preds == y).float().sum()
    accuracy = correct_sum.mul_(100.0 / len(y))
    return accuracy

def evaluate(test_loader, model, criterion, device, verbose=True):
    batch_test_accs = []
    batch_test_loss = []
    with torch.no_grad():
        model.eval()
        for batch_i, (X, y) in enumerate(test_loader):
        # ----------------------
            X = convert_X_for_resnet(X)

            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = get_prediction(logits)
            accuracy = get_accuracy(y, preds)
            batch_test_accs.append(accuracy.item())

            loss = criterion(logits, y)
            batch_test_loss.append(loss.item())

            # stdout
            if verbose:
                eval_stdout = f'\rBatch {batch_i}/{len(test_loader)} -- Accuracy: {np.mean(batch_test_accs):0.8f} -- Loss: {np.mean(batch_test_loss):0.8f}'
                sys.stdout.write(eval_stdout)
                sys.stdout.flush()

    mean_acc = sum(batch_test_accs) / len(batch_test_accs)
    mean_loss = sum(batch_test_loss) / len(batch_test_loss)
    model.train()
    return mean_acc, mean_loss

# Plotting of training results functions
# --------------------------------------------------------
def plot_loss(loader, metrics):
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=150)
    step_size = len(loader)
    ax.plot(np.array(metrics['losses'])[::step_size], c='g');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.grid(b=True, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.yaxis.grid(b=True, color=(0,0,0), alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_title('Training loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_train_test_accuracies(loader, metrics):
    step_size = len(loader)

#     max_acc = max(metrics['accs']['test'])
#     avg_acc = sum(metrics['accs']['test']) / len(metrics['accs']['test'])

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=150)
    ax.plot(np.array(metrics['accs']['train'])[::step_size], label='train', c='g')
    ax.plot(np.array(metrics['accs']['test'])[::1], label='test', c='orange')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.grid(b=True, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.yaxis.grid(b=True, color=(0,0,0), alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_title('Training and test accuracy')
    # ax.annotate(f'Max test acc: {max_acc:0.1f}% \n Avg test acc: {avg_acc:0.2f}%', xy=(30, 35), fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    plt.savefig('accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
