import torch


def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes Mean Squared Error loss.

    Args:
        preds (torch.Tensor): Predicted values
        targets (torch.Tensor): Ground truth values

    Returns:
        torch.Tensor: Computed MSE loss
    """

    return torch.mean((preds - targets) ** 2)
