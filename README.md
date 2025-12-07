# inpherens
Small application for c++ onnx based inference

## Get latest onnxruntime release
https://github.com/microsoft/onnxruntime/releases

##
```
mkdir build
cd build
cmake .. -DONNXRUNTIME_ROOT_DIR="C:/path/to/extracted/onnxruntime-win-xxx"
cmake --build . --config Release
```