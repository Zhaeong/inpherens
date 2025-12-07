#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    // 1. Setup Logging and Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    
    // Optional: Optimize for current system
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // 2. Load the Model
    // Ensure "model.onnx" is in the same directory as the binary or provide full path
#ifdef _WIN32
    const wchar_t* model_path = L"D:\\Github\\inpherens\\resnet50-v2-7.onnx";
#else
    const char* model_path = "model.onnx";
#endif

    std::cout << "Loading model..." << std::endl;
    try {
        Ort::Session session(env, model_path, session_options);

        // 3. Define Input/Output Info
        // In this example, we know the names from the Python script
        const char* input_names[] = {"data"};
        const char* output_names[] = {"resnetv24_dense0_fwd"};

        // 4. Create Input Data (1x5 matrix)
        //std::vector<int64_t> input_shape = {1, 5};
        //std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        //size_t input_tensor_size = 5; // 1 * 5

        // Prepare input tensor buffers
        // ResNet expects: 1 Batch, 3 Channels, 224 Height, 224 Width
        std::vector<int64_t> input_node_dims = { 1, 3, 224, 224 };
        size_t input_tensor_size = 1 * 3 * 224 * 224;
        std::vector<float> input_tensor_values(input_tensor_size);

        for (int i = 0; i < input_tensor_size; i++) {
            input_tensor_values[i] = 0.2f;
        }

        // Create memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, 
            OrtMemType::OrtMemTypeDefault
        );

        // 4. Create Tensor objects

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_tensor_values.data(), 
            input_tensor_size,
            input_node_dims.data(), 
            input_node_dims.size());


        // 5. Run Inference
        std::cout << "Running inference..." << std::endl;
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, // Number of inputs
            output_names, 
            1  // Number of outputs
        );

        // 6. Get Output Result
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        
        std::cout << "Input:  [1, 2, 3, 4, 5]" << std::endl;
        std::cout << "Output: [";
        for (int i = 0; i < 5; i++) {
            std::cout << floatarr[i] << (i < 4 ? ", " : "");
        }
        std::cout << "]" << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}