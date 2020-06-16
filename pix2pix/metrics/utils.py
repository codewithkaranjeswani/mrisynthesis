from typing import Optional, Union, Tuple, List
import torch

def _adjust_dimensions(input_tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    r"""Expands input tensors dimensions to 4D
    """
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    resized_tensors = []
    for tensor in input_tensors:
        tmp = tensor.clone()
        if tmp.dim() == 2:
            tmp = tmp.unsqueeze(0)
        if tmp.dim() == 3:
            tmp = tmp.unsqueeze(0)
        if tmp.dim() != 4 and tmp.dim() != 5:
            raise ValueError(f'Expected 2, 3, 4 or 5 dimensions (got {tensor.dim()})')
        resized_tensors.append(tmp)

    if len(resized_tensors) == 1:
        return resized_tensors[0]
    return tuple(resized_tensors)

def _validate_input(
        input_tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        allow_5d: bool,
        kernel_size: Optional[int] = None,
        scale_weights: Union[Optional[Tuple[float]], Optional[List[float]], Optional[torch.Tensor]] = None) -> None:

    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    assert isinstance(input_tensors, tuple)
    assert 0 < len(input_tensors) < 3, f'Expected one or two input tensors, got {len(input_tensors)}'

    min_n_dim = 2
    max_n_dim = 5 if allow_5d else 4
    for tensor in input_tensors:
        assert isinstance(tensor, torch.Tensor), f'Expected input to be torch.Tensor, got {type(tensor)}.'
        assert min_n_dim <= tensor.dim() <= max_n_dim, \
            f'Input images must be {min_n_dim}D - {max_n_dim}D tensors, got images of shape {tensor.size()}.'
        if tensor.dim() == 5:
            assert tensor.size(-1) == 2, f'Expected Complex 5D tensor with (N,C,H,W,2) size, got {tensor.size()}'

    if len(input_tensors) == 2:
        assert input_tensors[0].size() == input_tensors[1].size(), \
            f'Input images must have the same dimensions, got {input_tensors[0].size()} and {input_tensors[1].size()}.'

    if kernel_size is not None:
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got {kernel_size}.'
    if scale_weights is not None:
        assert isinstance(scale_weights, (list, tuple, torch.Tensor)), \
            f'Scale weights must be of type list, tuple or torch.Tensor, got {type(scale_weights)}.'
        assert (torch.tensor(scale_weights).dim() == 1), \
            f'Scale weights must be one dimensional, got {torch.tensor(scale_weights).dim()}.'

