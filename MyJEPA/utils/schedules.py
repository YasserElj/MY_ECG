"""Learning rate and other training schedules."""

import math


def update_learning_rate_(optimizer, lr):
    """Update learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_weight_decay_(optimizer, wd):
    """Update weight decay for parameter groups that use it."""
    for param_group in optimizer.param_groups:
        if param_group.get('use_weight_decay', False):
            param_group['weight_decay'] = wd


def cosine_schedule(
    total_steps: int,
    start_value: float,
    final_value: float,
    warmup_steps: int = 0,
    warmup_start_value: float = 0.,
    step: int = 0
):
    """
    Cosine decay schedule with optional warmup.
    
    Args:
        total_steps: Total number of training steps
        start_value: Value after warmup
        final_value: Final value after decay
        warmup_steps: Number of warmup steps
        warmup_start_value: Initial value during warmup
        step: Starting step
    
    Yields:
        Scheduled values
    """
    while True:
        if step < warmup_steps:
            value = warmup_start_value + (start_value - warmup_start_value) * step / warmup_steps
        elif step > total_steps:
            value = final_value
        else:
            decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
            coefficient = 0.5 * (1. + math.cos(math.pi * decay_ratio))
            value = final_value + coefficient * (start_value - final_value)
        yield value
        step += 1


def linear_schedule(
    total_steps: int,
    start_value: float,
    final_value: float,
    step: int = 0
):
    """
    Linear schedule from start to final value.
    
    Args:
        total_steps: Total number of training steps
        start_value: Initial value
        final_value: Final value
        step: Starting step
    
    Yields:
        Scheduled values
    """
    while True:
        if step < total_steps:
            value = start_value + (final_value - start_value) * step / total_steps
        else:
            value = final_value
        yield value
        step += 1

