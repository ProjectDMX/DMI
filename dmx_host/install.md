# Install
## clickhouse-cpp
```
git clone --recursive https://github.com/ClickHouse/clickhouse-cpp.git
cd clickhouse-cpp
makedir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_OPENSSL=ON
make
```
## clickhouse_client
My custom module.  
Modify Makefile.  
CLICKHOUSE_INCLUDE and CLICKHOUSE_LIBDIR should match the directory with clickhouse-cpp headers / clickhouse-cpp libraries.  
```
CLICKHOUSE_INCLUDE ?= /usr/local/include/
CLICKHOUSE_LIBDIR  ?= /usr/local/lib/
```
```
make
```
# Test run
After install:  
Setup python env of offloading backend(venv or conda).  
Then 
```
MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 MON_NATIVE_BATCH=1 PYTHONPATH=<offloading_backend_folder>/transformers/src:$PYTHONPATH python3 test_prefill.py
```
