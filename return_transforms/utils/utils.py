import torch

"""ESPER utils."""


def get_past_indices(x, seq_len):
    """
    Note: this assumes that padding is actually before the sequence.

    Often we want to get a tensor of indices for another tensor of shape
    (bsz, T, ...). These indices (bsz, T) should be between the start of the
    non-padded inputs and T. This function returns such an index tensor.
    """
    bsz, t = x.shape[:2]

    idxs = torch.randint(0, t, (bsz, t)).to(x)
    ts = torch.arange(0, t).view(1, t).expand(bsz, t).to(x)
    # Denotes how much padding is before each sequence
    pad_lens = t - seq_len.view(bsz, 1).expand(bsz, t)
    ts = ts - pad_lens + 1  # Shifts the indices so that the first non-padded index is 0

    # If ts == 0, then set idxs to 0. Otherwise, use the remainder of the division.
    idxs = torch.where(ts == 0, torch.zeros_like(idxs), idxs % ts)

    # Now add back the padding lengths
    idxs = idxs + pad_lens

    return idxs.long()
