import torch

k_min = 0 # equal
k_max = 1 # different


def get_k_mtrx(input, sigma=2):
    input = input.sub(input.unsqueeze(1)).pow(2).view(input.size(0),
                                                      input.size(0), -1)
    input = input.sum(dim=-1).mul(-1. / (2 * sigma**2)).exp()

    return input


def get_ideal_k_mtrx(targets, n_classes):
    onehot_targets = torch.nn.functional.one_hot(targets, n_classes)
    onehot_targets = onehot_targets.float()

    ideal = onehot_targets.mm(onehot_targets.t())

    return ideal


def contrastive(input, targets, n_classes, neo=True):
    """
    A contrastive-loss-like instantiation.
    """

    x = torch.where(
        torch.eye(len(input)).to(input.device) == 1,
        torch.tensor(-float('inf')).to(input.device),
        get_k_mtrx(input))  # removes the main diagonal
    x = x.view([-1] + list(x.size())[2:])

    y = get_ideal_k_mtrx(targets, n_classes=n_classes)
    y = y.view([-1] + list(y.size())[2:])

    if neo:
        return -torch.mean(torch.exp(x[y == k_min]))

    return torch.sum(torch.exp(x[y == k_max])) / torch.sum(torch.exp(x))
