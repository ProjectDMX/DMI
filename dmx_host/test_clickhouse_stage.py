import clickhouse_client

from dmx_interface import StageTwoItemForDB
from dmx_interface import torch_encode
import torch


cfg = clickhouse_client.ClickHouseClientConfig()
cfg.host = "localhost"
cfg.port = 9000
cfg.username = "default"
cfg.password = ""
cfg.database = "default"
cfg.table = "offload"
cfg.secure = False
cfg.client_side_compress = False   # or True / "lz4" / "zstd"
cfg.client_settings = None         # or {"async_insert": 1, "wait_for_async_insert": 0}
cfg.create_database_if_missing = True
cfg.drop_existing_database = True
cfg.index_granularity = 8192

# In each worker thread:
clickhouse_client.clickhouse_init(thread_idx=0, thread_init_config=cfg)

a = torch.zeros(1, 10, 20)
b = torch.zeros(1, 10, 30)
json_a, bytes_a = torch_encode(a)
size_a = a.numel() * a.element_size()
json_b, bytes_b = torch_encode(b)
size_b = b.numel() * b.element_size()
combined = ['gpt2', '0', 'fake', 1, 0, 10]
items = [StageTwoItemForDB(tuple(combined + [json_a, bytes_a]), size_a), StageTwoItemForDB(tuple(combined + [json_b, bytes_b]), size_b)]
clickhouse_client.clickhouse_insert(items)

# Once per thread:
clickhouse_client.clickhouse_cleanup()
