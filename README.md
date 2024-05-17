# HunyuanDiT-TensorRT-libtorch

用`TensorRT`和`libtorch`简单实现了`HunyuanDiT`模型的`pipeline`推理。

## 准备

- 安装`TensorRT`, `TensorRT10`的api变化了, 建议用`TensorRT8`以下的版本
- 从[huggingface](`https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main`)下载模型
- 安装`pytorch`, `onnx`等依赖

## 导出4个onnx模型用于pipeline

修改[export.py](export.py)中的`args`
执行:

```shell
python export.py
```

你会得到`bert`, `t5`, `hunyuan`, `vae`四个onnx模型
你可以用[onnxsim](`https://github.com/daquexian/onnx-simplifier`)将它们简化
执行:

```shell
onnxsim bert.onnx bert-sim.onnx
onnxsim t5.onnx t5-sim.onnx
onnxsim hunyuan.onnx hunyuan-sim.onnx
onnxsim vae.onnx vae-sim.onnx
```

onnx很大的情况下, 简化的耗时也很长

## onnx转换到tensorrt

这里我用了trtexec转化, 比较省事

```shell
trtexec --onnx=bert-sim.onnx --saveEngine=bert.plan --fp16
trtexec --onnx=t5-sim.onnx --saveEngine=t5.plan --fp16
trtexec --onnx=hunyuan-sim.onnx --saveEngine=hunyuan.plan --fp16
trtexec --onnx=vae-sim.onnx --saveEngine=vae.plan --fp16
```

tensorrt转换的过程也很慢

## 编译安装python包

执行:

```shell
python setup.py install
```

包名是: `py_hunyuan_dit`

## 推理一个文生图

修改[run.py](run.py)中的4个模型路径, 修改推理步数, 默认100比较慢

执行:

```shell
python run.py
```

你会看到生成的图片


