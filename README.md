# Enable Fast CPU Training and Inference of YOLOX
This repo is an example on using [BigDL-Nano](https://bigdl.readthedocs.io/en/latest/doc/Nano/index.html) to enable fast CPU training and inference of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
## Environmental Setup
```
conda create -n yolox_cpu python=3.7 setuptools=58.0.4
conda activate yolox_cpu
pip install --pre --upgrade bigdl-nano[pytorch]
pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install --upgrade intel-extension-for-pytorch
pip install openvino-dev
pip install neural-compressor==1.12
pip install onnx==1.8.1 onnxruntime-extensions
pip install -v -e .
source bigdl-nano-init
```
## Training
### Single-process training
```
python -m yolox.tools.train -n yolox_nano -b 64 --precision 32 [--use_ipex]
```
### Multi-process training
```
python -m yolox.tools.train -n yolox_nano -b 64 --num_processes 4 --strategy subprocess --precision 32 [--use_ipex]
```
[]: optional
* num_processes – number of processes in distributed training, defaults to 1
* use_ipex – whether use [ipex acceleration](https://github.com/intel/intel-extension-for-pytorch), defaults to False
* strategy – use which backend in distributed mode, defaults to ‘subprocess’, now avaiable backends are ‘spawn’, ‘subprocess’ and ‘ray’
* precision – Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16), defaults to 32. Enable ipex bfloat16 weight prepack when use_ipex=True and precision=’bf16’
### Accelerating results
Below accelerations were obtained on 4 Intel Xeon Cooper Lake CPUs (each has 56 logical cores) by counting the average execution time of first two epochs (batch size=64).
* single process: 20200s
* single process with ipex: 16329s, __80.8%__ of single process cost
* 2 processes: 7466s, __37.0%__ of single process cost
* 4 processes: 4590s, __22.7%__ of single process cost
* 8 processes: 3389s, __16.8%__ of single process cost

## Inference Pipeline Demo
```
python -m yolox.tools.inference_demo -n yolox_nano -b 64 -c /path/to/your/model
```
### Summarization of optimizing approaches
|method|status|latency(ms)[^1]|accuracy|
|----|----|----|----|
|            original            |      successful      |   213.539    |        0.411         |
|           fp32_ipex            |      successful      |   225.013    |    not recomputed    |
|              bf16              |      successful      |   153.385    |        0.406         |
|           bf16_ipex            |      successful      |   452.874    |         0.41         |
|              int8              |      successful      |   202.647    |         0.0          |
|            jit_fp32            |   fail to convert    |     None     |         None         |
|         jit_fp32_ipex          |   fail to convert    |     None     |         None         |
|  jit_fp32_ipex_channels_last   |   fail to convert    |     None     |         None         |
|         openvino_fp32          |      successful      |    173.06    |    not recomputed    |
|         openvino_int8          |   fail to convert    |     None     |         None         |
|        onnxruntime_fp32        |      successful      |   969.888    |    not recomputed    |
|    onnxruntime_int8_qlinear    |   fail to convert    |     None     |         None         |
|    onnxruntime_int8_integer    |   fail to convert    |     None     |         None         |

[^1]: includes a batch of 64 samples.
### Known issue
* GPU-CPU patch tool cannot support jit
* int8 quantization will lead to zero accuracy