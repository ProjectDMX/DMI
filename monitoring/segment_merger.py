# NOTE: Currently not merging across shard_ranks.
# TODO: Merge across shard_ranks && layer_no to easily support distributed.
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import torch

def parse_internal_id(internal_id: str) -> Tuple[int, str]:
    if internal_id.startswith("blocks."):
        internal_id_list = internal_id.split('.')
        return (int(internal_id_list[1]), '.'.join(['blocks'] + internal_id_list[2:]))
    else:
        return (-1, internal_id)
    
def get_delta_token_len(shape: tuple, act_name: str, have_batch_dim: bool = False) -> int:
    diff_dim = 1 if not have_batch_dim else 0
    if act_name.endswith('attn.hook_attn_scores') or act_name.endswith('attn.hook_pattern'):
        return shape[2 - diff_dim]
    else:
        return shape[1 - diff_dim]


class OffloadedSegments(ABC):
    @abstractmethod
    def append(self, value) -> int:
        pass
    @abstractmethod
    def extend(self, segments):
        pass
    @abstractmethod
    def read_and_merge(self):
        pass
    @property
    @abstractmethod
    def size(self):
        pass

class TorchOffloadedSegmentsOnDim(OffloadedSegments):
    def __init__(self, token_dim: int, drop_token_cnt_to: Optional[int] = None):
        self._tensor_list = []
        self._token_dim = token_dim
        self._drop_token_cnt_to = drop_token_cnt_to
        assert self._drop_token_cnt_to is None or self._drop_token_cnt_to >= 0
    def append(self, value: torch.Tensor) -> int:
        self._tensor_list.append(value)
        return value.numel() * value.element_size()
    def extend(self, segments):
        if hasattr(segments, "_tensor_list"):
            self._tensor_list.extend(segments._tensor_list)
        else:
            self._tensor_list.extend(segments)
    def read_and_merge(self):
        if len(self._tensor_list) == 0:
            return None
        t = torch.cat(self._tensor_list, dim=self._token_dim)
        keep = min(t.shape[self._token_dim], self._drop_token_cnt_to) if self._drop_token_cnt_to is not None else t.shape[self._token_dim]
        keep = int(keep)
        if keep < t.shape[self._token_dim]:
            t = t.narrow(self._token_dim, 0, keep)
        self._tensor_list = [t]
        return t
    @property
    def size(self):
        return sum([t.numel() * t.element_size() for t in self._tensor_list])
    

# TODO: Store as Sparse matrix.
class TorchOffloadedSegmentsAttnMatrix(OffloadedSegments):
    def __init__(self, token_dim_incremental: int, token_dim_sum_to_now: int, fill_nan_value: float, drop_token_cnt_to: Optional[int] = None):
        self._tensor_list = []
        self._td_inc = token_dim_incremental
        self._td_sum = token_dim_sum_to_now
        self._fill_val = fill_nan_value # Can be 0 after softmax and -inf before.
        self._drop_token_cnt_to = drop_token_cnt_to
        assert self._drop_token_cnt_to is None or self._drop_token_cnt_to >= 0
    def append(self, value: torch.Tensor) -> int:
        self._tensor_list.append(value)
        return value.numel() * value.element_size()
    def extend(self, segments):
        if hasattr(segments, "_tensor_list"):
            self._tensor_list.extend(segments._tensor_list)
        else:
            self._tensor_list.extend(segments)
    def read_and_merge(self):
        if len(self._tensor_list) == 0:
            return None
        total_token_cnt = self._tensor_list[-1].shape[self._td_sum]
        keep_token_cnt = min(total_token_cnt, self._drop_token_cnt_to) if self._drop_token_cnt_to is not None else total_token_cnt
        t_list = []
        for t in self._tensor_list:
            if t.shape[self._td_sum] > keep_token_cnt:
                t = t.narrow(self._td_sum, 0, keep_token_cnt)
            t_shape = list(t.shape)
            t_shape[self._td_sum] = keep_token_cnt - t_shape[self._td_sum]
            if t_shape[self._td_sum] > 0:
                pad_t = torch.full(
                    size=t_shape, 
                    fill_value=self._fill_val, 
                    dtype=t.dtype, 
                    device=t.device
                )
                t = torch.cat([t, pad_t], dim=self._td_sum)
            t_list.append(t)
        t = torch.cat(t_list, dim=self._td_inc)
        if keep_token_cnt < total_token_cnt:
            keep_inc = keep_token_cnt
            t = t.narrow(self._td_inc, 0, keep_inc)
        self._tensor_list = [t]
        return t
    @property
    def size(self):
        return sum([t.numel() * t.element_size() for t in self._tensor_list])
    
# Now no batch_dim.
def segment_manager(act_name: str, drop_token_cnt_to: Optional[int] = None, have_batch_dim: bool = False):
    # NOTE: This assumes parse_id before.
    dim_diff = 1 if not have_batch_dim else 0
    if act_name.endswith('attn.hook_attn_scores'):
        return TorchOffloadedSegmentsAttnMatrix(token_dim_incremental=2-dim_diff, token_dim_sum_to_now=3-dim_diff, fill_nan_value=float('-inf'), drop_token_cnt_to=drop_token_cnt_to)
    elif act_name.endswith('attn.hook_pattern'):
        return TorchOffloadedSegmentsAttnMatrix(token_dim_incremental=2-dim_diff, token_dim_sum_to_now=3-dim_diff, fill_nan_value=0.0, drop_token_cnt_to=drop_token_cnt_to)
    else:
        return TorchOffloadedSegmentsOnDim(token_dim=1-dim_diff, drop_token_cnt_to=drop_token_cnt_to)

def merge_segments(tensor_list: list, act_name: str, drop_token_cnt_to: Optional[int] = None, have_batch_dim: bool = False) -> torch.Tensor:
    manager = segment_manager(act_name, drop_token_cnt_to, have_batch_dim)
    manager.extend(tensor_list)
    return manager.read_and_merge()
    