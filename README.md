# tflite-nanodet

This is a implement of nanodet inference on tflite. 

Offical code: [Pytorch](https://github.com/RangiLyu/nanodet)

## How to use:
### Requirements

* Python >= 3.6
* Pytorch >= 1.7
* tensorflow >= 2.6.0

You can convert tensorflow SavedModel to tensorflow Lite model by convert.py. 

The inference code is in inference.py, includes preprocess and postprocess.

## Others:
Nanodet pytorch model convert to onnx, see this [guide](https://github.com/RangiLyu/nanodet/blob/main/demo_openvino/README.md):
```bash
python ./tools/export_onnx.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
```
When you convert pytorch model to onnx, you may face this question:

```bash
RuntimeError: Resize coordinate_transformation_mode=pytorch_half_pixel is not supported in Tensorflow.
```

You should modify line 56 in tools/export_onnx.py in [nanodet](https://github.com/RangiLyu/nanodet) to：
 ```python
opset_version=10,
 ```

For more details, you can find in this [link](https://github.com/onnx/onnx-tensorflow/issues/632)

How to convert onnx to tensorflow: [onnx-tf](https://github.com/onnx/onnx-tensorflow).

