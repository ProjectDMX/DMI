from typing import List, Tuple, Union
import threading
import torch
import json
import numpy
try:
    from .batching_queue import SizedWorkItem
except ImportError:
    from batching_queue import SizedWorkItem
BytesLike = Union[bytes, bytearray, memoryview]

TORCH_DTYPES_NAME2TYPE = {"torch.float32": torch.float32, "torch.float": torch.float, 
                     "torch.float64": torch.float64, "torch.double": torch.double,
                     "torch.complex32": torch.complex32, "torch.chalf": torch.chalf, 
                     "torch.complex64": torch.complex64, "torch.cfloat": torch.cfloat,
                     "torch.complex128": torch.complex128, "torch.cdouble": torch.cdouble,
                     "torch.float16": torch.float16, "torch.half": torch.half,
                     "torch.bfloat16": torch.bfloat16, "torch.uint8": torch.uint8,
                     "torch.int8": torch.int8, "torch.int16": torch.int16,
                     "torch.short": torch.short, "torch.int32": torch.int32, 
                     "torch.int": torch.int, "torch.int64": torch.int64, 
                     "torch.long": torch.long, "torch.bool": torch.bool}
TORCH_DTYPES_TYPE2NAME = {v: k for k, v in TORCH_DTYPES_NAME2TYPE.items()}

def torch_encode_numpy(tensor: torch.Tensor) -> Tuple[memoryview, numpy.ndarray]:
    # Returns (metadata, data).
    dtype_str = TORCH_DTYPES_TYPE2NAME[tensor.dtype]
    shape_tuple = tuple(tensor.shape)
    meta_json_dict = {"dtype": dtype_str, "shape": shape_tuple}
    json_str = json.dumps(meta_json_dict)
    # TODO: More compact json.
    json_bytes = json_str.encode()
    metadata = memoryview(json_bytes).cast('B')
    # Return memoryview.
    # NOTE: Assume a detached, on CPU, contiguous tensor
    data = tensor.flatten().numpy()
    return metadata, data

def torch_encode(tensor: torch.Tensor) -> Tuple[memoryview, memoryview]:
    metadata, numpy_data = torch_encode_numpy(tensor)
    return metadata, memoryview(numpy_data).cast('B')

def torch_decode_numpy(metadata: BytesLike, numpy_data: numpy.ndarray):
    json_str = metadata.decode()
    meta_json_dict = json.loads(json_str)
    dtype_str = meta_json_dict["dtype"]
    shape_tuple = meta_json_dict["shape"]
    dtype = TORCH_DTYPES_NAME2TYPE[dtype_str]
    numpy_data = numpy_data.reshape(shape_tuple)
    tensor = torch.from_numpy(numpy_data).to(dtype)
    return tensor

def torch_decode_list(metadata: BytesLike, list_data: list):
    json_str = metadata.decode()
    meta_json_dict = json.loads(json_str)
    dtype_str = meta_json_dict["dtype"]
    shape_tuple = meta_json_dict["shape"]
    dtype = TORCH_DTYPES_NAME2TYPE[dtype_str]
    tensor = torch.tensor(list_data, dtype=dtype)
    tensor = tensor.reshape(shape_tuple)
    return tensor

def torch_decode(metadata: BytesLike, data: BytesLike) -> torch.Tensor:
    json_str = metadata.decode()
    meta_json_dict = json.loads(json_str)
    dtype_str = meta_json_dict["dtype"]
    shape_tuple = meta_json_dict["shape"]
    dtype = TORCH_DTYPES_NAME2TYPE[dtype_str]
    tensor = torch.frombuffer(data, dtype=dtype).view(*shape_tuple)
    return tensor

class StageOneItemForFuture(SizedWorkItem):
    def __init__(self, row):
        self._row = row
    @property
    def row(self):
        return self._row
    def size(self):
        # Now the first queue is bounded by rowcnt.
        # TODO: Change to real tensor size.
        return 1
    
class StageTwoItemForDB(SizedWorkItem):
    # Here it defines the format.
    # Format is hard-coded here.
    def __init__(self, row, size: int):
        self._row = row
        self._size = size
    def size(self):
        return self._size

def input_handler_v1(
    list_of_tuple_keys: List[tuple],
    list_of_start_token_idx: List[int],
    list_of_cache_dict: List[dict],
) -> List[SizedWorkItem]:
    n = len(list_of_tuple_keys)
    assert n == len(list_of_start_token_idx) == len(list_of_cache_dict)
    
    total = 0
    for cd in list_of_cache_dict:
        total += len(cd)

    out: List[SizedWorkItem] = [None] * total  # type: ignore[list-item]
    j = 0

    for tuple_key, start_idx, cache_dict in zip(
        list_of_tuple_keys, list_of_start_token_idx, list_of_cache_dict
    ):
        for k, v in cache_dict.items():
            out[j] = StageOneItemForFuture((*tuple_key, start_idx, k, v))
            j += 1

    return out


def parse_internal_id(internal_id: str) -> Tuple[int, str]:
    if internal_id.startswith("blocks."):
        internal_id_list = internal_id.split('.')
        return (int(internal_id_list[1]), '.'.join(['blocks'] + internal_id_list[2:]))
    else:
        return (-1, internal_id)
    
def get_delta_token_len(shape: tuple, act_name: str) -> int:
    if act_name.endswith('attn.hook_attn_scores') or act_name.endswith('attn.hook_pattern'):
        # print(f"{act_name}: {shape[2]}")
        return shape[2]
    else:
        # print(f"{act_name}: {shape[1]}")
        return shape[1]

_tls_stage_one = threading.local()
def stage_one_thread_init(thread_idx: int, thread_init_config):
    _tls_stage_one.worker_id = thread_idx

def stage_one_thread_cleanup():
    pass
        
def stage_one_parsing_and_wait(list_of_items: List[StageOneItemForFuture]) -> List[StageTwoItemForDB]:
    output_list = []
    for item in list_of_items:
        row = item.row
        # NOTE: Hard-coded format.
        model_id, request_id, start_token_idx, act_name, tensor_future = row
        if tensor_future is None:
            continue
        tensor = tensor_future.result()
        if tensor is None:
            continue
        layer_no, act_name = parse_internal_id(act_name)
        delta_token_len = get_delta_token_len(tuple(tensor.shape), act_name)
        end_token_idx = start_token_idx + delta_token_len
        assert delta_token_len > 0
        metadata, data = torch_encode(tensor)
        # NOTE: Hard-coded format.
        formatted_row = (model_id, request_id, act_name, layer_no, start_token_idx, end_token_idx, metadata, data)
        size = tensor.numel() * tensor.element_size()
        output_list.append(StageTwoItemForDB(formatted_row, size))
    return output_list

