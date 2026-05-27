from torch.optim.lr_scheduler import LambdaLR

def get_lr_multiplier(step_num, d_model=512, warmup_steps=4000):
    """
    Noam learning rate schedule multiplier.
    Avoids dividing by zero by setting step_num to at least 1.
    """
    step_num = max(step_num, 1)
    return (d_model ** -0.5) * min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

def build_scheduler(optimizer, d_model=512, warmup_steps=4000):
    """
    Build LambdaLR scheduler with Noam schedule.
    """
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(step, d_model=d_model, warmup_steps=warmup_steps)
    )
