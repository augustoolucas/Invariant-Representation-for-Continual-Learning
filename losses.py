import torch

k_min = 0  # equal
k_max = 1  # different


def get_dissimilarity_matrix(input, sigma=1, beta=-1):
    """
    Gaussian kernel (indirect):
      k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2));
      x -> (k(c_1, x), ..., k(c_m, x)), c_i \in centers, \forall i.
    """
    input = input.sub(input.unsqueeze(1)).pow(2).view(input.size(0),
                                                      input.size(0), -1)
    input = input.sum(dim=-1).mul(-1. / (2 * sigma**2)).exp()

    return input - beta


def get_mse_dissimilarity_matrix(input, beta=0):
    """
    Using MSE to calculate the distance between the feature vectors
      k(x, y) = ||x-y||_2^2;
      x -> (k(c_1, x), ..., k(c_m, x)), c_i \in centers, \forall i.
    """
    input = input.sub(input.unsqueeze(1)).pow(2).view(input.size(0),
                                                      input.size(0), -1)
    input = input.mean(dim=-1)

    return input - beta


def get_pairwise_labels_matrix(targets, n_classes):
    onehot_targets = torch.nn.functional.one_hot(targets, n_classes)
    onehot_targets = onehot_targets.float()

    ideal = onehot_targets.mm(onehot_targets.t())

    return ideal


def nmse(input, targets, n_classes, neo=False):
    loss_fn = torch.nn.MSELoss(reduction='mean')

    x = get_dissimilarity_matrix(input, beta=0)
    x = x.view([-1] + list(x.size())[2:])

    y = get_pairwise_labels_matrix(targets, n_classes=n_classes)
    y = y.view([-1] + list(y.size())[2:])

    if neo:
        idx = y == k_max
        return loss_fn(x[idx], y[idx])

    return -loss_fn(x, y)


def contrastive(input, targets, n_classes, neo=False):
    """
    A contrastive-loss-like instantiation.
    """

    x = torch.where(
        torch.eye(len(input)).to(input.device) == 1,
        torch.tensor(-float('inf')).to(input.device),
        get_dissimilarity_matrix(input))  # removes the main diagonal
    x = x.view([-1] + list(x.size())[2:])

    y = get_pairwise_labels_matrix(targets, n_classes=n_classes)
    y = y.view([-1] + list(y.size())[2:])

    if neo:
        return torch.mean(torch.exp(x[y == k_max]))

    return -torch.sum(torch.exp(x[y == k_min])) / torch.sum(torch.exp(x))
