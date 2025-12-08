# inpherens
Small application for c++ onnx based inference

## Get latest onnxruntime release
https://github.com/microsoft/onnxruntime/releases

## To build
```
mkdir build
cd build
cmake .. -DONNXRUNTIME_ROOT_DIR="C:/path/to/extracted/onnxruntime-win-xxx"
cmake .. -DONNXRUNTIME_ROOT_DIR=D:\Github\inpherens\onnxruntime-win-x64-1.23.2
cmake --build . --config Release
```

## Using onnxruntime-extensions to tokenize input
https://github.com/microsoft/onnxruntime-extensions

See TEST(CApiTest, test_enable_ort_customops_stringlower):
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/shared_lib/test_inference.cc#L824

This error
```
ONNX Runtime Error: Load model from D:\Github\inpherens\tokenizer_convert\tokenizer.onnx failed:Fatal error: ai.onnx.contrib:CLIPTokenizer(-1) is not a registered function/op
```
requires custom op registration
```
session_options.RegisterCustomOpsLibrary(L"ortextensions.dll");
```
