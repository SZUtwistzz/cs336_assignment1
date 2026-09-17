import torch


def cross_entropy(logits: torch.Tensor,targets: torch.Tensor):
    max_values = logits.amax(dim = -1,keepdim=True)
    shifted = logits-max_values

    sum_exp = shifted.exp().sum(dim =-1)
    log_sum_exp = sum_exp.log()

    target_logits = shifted.gather(
        dim = -1,
        index = targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - target_logits

    return losses.mean()