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

    session_options.RegisterCustomOpsLibrary(L"ortextensions.dll");

    // 2. Load the Model
    // Ensure "model.onnx" is in the same directory as the binary or provide full path
#ifdef _WIN32
    const wchar_t* model_path = L"D:\\Github\\inpherens\\resnet50-v2-7.onnx";
    const wchar_t* tokenizer_path = L"D:\\Github\\inpherens\\tokenizer_convert\\tokenizer.onnx";
#else
    const char* model_path = "model.onnx";
#endif

    std::cout << "Loading model..." << std::endl;
    try {
        //// The line loads the customop library into ONNXRuntime engine to load the ONNX model with the custom op
        //Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, custom_op_library_filename, &handle));

        Ort::Session session_tokenizer(env, tokenizer_path, session_options);

    // The model typically expects a 1D tensor of strings (batch_size)
        std::vector<int64_t> input_shape = { 1 }; // Batch size of 1

        Ort::AllocatorWithDefaultOptions allocator;

        //Ort::Value::CreateTensor<float>


        std::string input_data{ "I am a cat what is going on" };
        const char* const input_strings[] = { input_data.c_str() };
        std::vector<int64_t> input_dims = { 1, 1 };

        Ort::Value input_tensor = Ort::Value::CreateTensor(allocator, input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
        input_tensor.FillStringTensor(input_strings, 1U);

        // 5. Run Inference
        const char* input_names[] = { "input_text" }; // Default name from gen_processing_models
        const char* output_names[] = { "input_ids", "attention_mask" }; // Typical outputs

        auto output_tensors = session_tokenizer.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1,
            output_names,
            2 // We are requesting 2 outputs
        );

        // 6. Process Output (Token IDs)
        // "input_ids" is usually the first output (int64 tensor)
        int64_t* token_ids = output_tensors[0].GetTensorMutableData<int64_t>();
        size_t num_tokens = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::cout << "Original Prompt: \"" << input_data << "\"" << std::endl;
        std::cout << "Token IDs: [ ";
        for (size_t i = 0; i < num_tokens; i++) {
            std::cout << token_ids[i] << " ";
        }
        std::cout << "]" << std::endl;

        return 0;

        
        /*
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

        */

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}