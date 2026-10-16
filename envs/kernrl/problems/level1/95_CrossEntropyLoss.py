# [MODULE DOC]
# kernrl problem (level 1): mean cross-entropy loss over 4096 samples and 10
# classes.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model computing cross-entropy loss. No parameters.
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        # [FUNCTION DOC]
        # Returns the scalar mean cross-entropy between logits predictions
        # (batch, num_classes) and integer class targets (batch,).
        return torch.nn.functional.cross_entropy(predictions, targets)


batch_size = 4096
num_classes = 10
input_shape = (num_classes,)  # Output for each class
dim = 1


def get_inputs():
    # [FUNCTION DOC]
    # Returns random logits of shape (batch_size, num_classes) and random int64
    # targets in [0, num_classes).
    return [
        torch.randn(batch_size, *input_shape),
        torch.randint(0, num_classes, (batch_size,)),
    ]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
