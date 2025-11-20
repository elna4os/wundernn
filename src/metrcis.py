import torch


def r2(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate R² (coefficient of determination) metric

    Args:
        preds (torch.Tensor): Predictions tensor of shape (batch_size, seq_len, feature_dim)
        targets (torch.Tensor): Targets tensor of shape (batch_size, seq_len, feature_dim)

    Returns:
        torch.Tensor: R² score
    """

    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2_score = 1 - ss_res / ss_tot

    return r2_score
