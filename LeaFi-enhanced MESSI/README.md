# LeaFi-enhanced MESSI

### requirements

* LibTorch (<https://pytorch.org/cppdocs/installing.html>)
* GSL (<https://www.gnu.org/software/gsl/>)

### build

LeaFi-enhanced MESSI can be built with CMake.

```bash
cd /path/to/code
mkdir build
cd build
cmake ..
cmake --build .
```

### run

ere is an example code to run LeaFi-enhanced MESSI.
The full list of parameters and their explanation can be found in src/utils/config.c

```bash
build/leafi_messi --database_filepath /path/to/data --database_size 25000000 --series_length 256 --query_filepath /path/to/query --query_size 1000 --leaf_size 10000 --exact_search --on_disk --cpu_cores 16 --require_neural_filter --filter_conformal_recall 0.99 --filter_num_synthetic_query_global 1500 --filter_num_synthetic_query_local 500 --filter_train_val_split 0.8 --log_filepath /path/to/log --dump_index --index_dump_folderpath /path/to/index/dump
```
