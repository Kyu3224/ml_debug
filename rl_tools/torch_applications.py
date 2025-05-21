import torch


def compute_grad_norm(parameters, norm_type=2, only_non_none=True):
    """
    Compute the total gradient norm of a list of parameters.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Model parameters.
        norm_type (float): Type of the used p-norm. Default is 2.
        only_non_none (bool): If True, skip parameters with no gradient.

    Examples:
        rl_tools.compute_grad_norm(self.policy.transformer.parameters)
    """
    total_norm = 0.0
    for p in parameters():
        if p.grad is not None or not only_non_none:
            param_norm = p.grad.data.norm(norm_type) if p.grad is not None else 0.0
            total_norm += param_norm.item() ** norm_type
    print("Clipped grad norm:", total_norm ** (1. / norm_type))


def summarize_optimizer(optimizer, return_only=True) -> int:
    """
    Summarize optimizer parameter groups, showing shapes and trainable parameter counts.

    Args:
        return_only (bool): If True, don't display parameters in the optimizer group.
        optimizer (torch.optim.Optimizer): The optimizer to summarize.

    Examples:
        rl_tools.summarize_optimizer(self.policy.optimizer)

    Returns:
        total_params (int): Total number of parameters in the optimizer.
    """
    total_params = 0
    print_fn = (lambda *args, **kwargs: None) if return_only else print

    print_fn("=== Optimizer Parameter Groups Summary ===")

    for group_idx, group in enumerate(optimizer.param_groups):
        print_fn(f"\n[Group {group_idx}]")

        for p_idx, param in enumerate(group["params"]):
            shape = tuple(param.shape)
            count = param.numel()
            requires_grad = param.requires_grad

            print_fn(f"Param {p_idx}: shape={shape}, requires_grad={requires_grad}")

            if requires_grad:
                total_params += count
                if len(shape) not in (1, 2):
                    print_fn(f"  [INFO] Unusual shape (rank {len(shape)}): counted {count} parameters")
            else:
                print_fn("  [INFO] Skipped (requires_grad=False)")

    print_fn(f"\n✅ Total trainable parameters in optimizer: {total_params}")

    return total_params

def validate_tensor(tensor_dict):
    """
    Validates that tensors do not contain NaNs or Infs.
    Please Call before computing loss.

    Args:
        tensor_dict (dict): A dictionary of tensors to validate.

    Examples:
       rl_tools.validate_tensor({
                "state_batch": state_batch,
                "obs_batch": obs_batch,
                "actions_batch": actions_batch,
            })

    Raises:
        AssertionError: If any tensor contains NaN or Inf.
    """
    for name, tensor in tensor_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} is not a torch.Tensor")

        if torch.isnan(tensor).any():
            raise ValueError(f"[NaN Detected] Tensor '{name}' contains NaNs.")

        if torch.isinf(tensor).any():
            raise ValueError(f"[Inf Detected] Tensor '{name}' contains Infs.")


def check_param_grad(named_parameters):
    """
    Check gradients of model parameters and print their status.
    Please Call after the optimizer step.

    Args:
        named_parameters (Iterable[(str, torch.nn.Parameter)]): Named parameters from a model.

    Examples:
        rl_tools.check_param_grad(self.policy.named_parameters)
    """
    for name, param in named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"[WARNING] {name} has no gradient!")
            elif torch.all(param.grad == 0):
                print(f"[INFO] {name} has all-zero gradients.")
        else:
            print(f"[INFO] {name} does not require gradient.")


def get_cloned_params(parameters):
    """
    Clone the list of parameters for later comparison.
    Please Call before the optimizer step.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Parameters to clone.

    Returns:
        List[torch.Tensor]: Cloned parameter tensors.

    Examples:
        before_params = rl_tools.get_cloned_params(self.policy.parameters)
    """
    return [p.detach().clone() for p in parameters()]


def report_param_changes(
    named_parameters,
    before_params: list[torch.Tensor],
    threshold: float = 1e-6,
    return_only: bool = True
) -> int:
    """
    Report which parameters have changed after an optimizer step.

    Args:
        named_parameters (Callable[[], Iterable[Tuple[str, torch.nn.Parameter]]]):
            A callable returning named parameters (e.g., model.named_parameters).
        before_params (List[torch.Tensor]): Cloned tensors before the optimizer step.
        threshold (float): Minimum absolute difference to count as changed.
        return_only (bool): If True, suppress print output.

    Returns:
        int: Total number of changed individual parameters.
    """
    print_fn = (lambda *args, **kwargs: None) if return_only else print
    changed_param_cnt = 0
    changed_param_groups = 0
    unchanged_params = []

    for (name, after_param), before_param in zip(named_parameters(), before_params):
        diff_mask = (before_param - after_param).abs() > threshold
        num_changed = diff_mask.sum().item()

        if num_changed > 0:
            changed_param_cnt += num_changed
            changed_param_groups += 1
            print_fn(f"[Changed] {name} — {num_changed} elements changed "
                     f"out of {before_param.numel()} "
                     f"({100 * num_changed / before_param.numel():.2f}%)")
        else:
            unchanged_params.append(name)

    if not return_only:
        for name in unchanged_params:
            print_fn(f"[Unchanged]: {name}")
        print_fn(f"\n✅ Num of changed parameter groups: {changed_param_groups}")
        print_fn(f"✅ Total number of changed parameters: {changed_param_cnt}")

    return changed_param_cnt
