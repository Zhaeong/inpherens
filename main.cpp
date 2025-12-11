#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

std::vector<int64_t> tokenize(std::string input)
{
    const wchar_t* tokenizer_path = L"D:\\Github\\inpherens\\tokenizer_convert\\tokenizer.onnx";
    try 
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inpherens");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        // Required for tokenizer custom op
        session_options.RegisterCustomOpsLibrary(L"ortextensions.dll");

        Ort::Session session_tokenizer(env, tokenizer_path, session_options);
        const char* const input_strings[] = { input.c_str() };
        std::vector<int64_t> input_shape = { 1 }; // Batch size of 1

        Ort::AllocatorWithDefaultOptions allocator;

        Ort::Value input_tensor = Ort::Value::CreateTensor(allocator, 
                                                           input_shape.data(), 
                                                           input_shape.size(), 
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
        input_tensor.FillStringTensor(input_strings, 1U);

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

        // "input_ids" is usually the first output (int64 tensor)
        int64_t* token_ids = output_tensors[0].GetTensorMutableData<int64_t>();
        size_t num_tokens = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<int64_t> output;
        output.reserve(num_tokens);

        std::cout << "Original Prompt: \"" << input << "\"" << std::endl;
        std::cout << "Token IDs: [ ";
        for (size_t i = 0; i < num_tokens; i++) {
            std::cout << token_ids[i] << " ";
            output.push_back(token_ids[i]);
        }
        std::cout << "]" << std::endl;

        return output;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    }
}

void text_encode(std::vector<int32_t> input_ids, std::vector<Ort::Float16_t>& last_hidden_state, std::vector<Ort::Float16_t>& pooler_output)
{
    const wchar_t* tokenizer_path = L"D:\\Models\\stable-diffusion-1.5_io16_amdgpu\\text_encoder\\model.onnx";
    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inpherens");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        Ort::Session session(env, tokenizer_path, session_options);
        std::vector<int64_t> input_shape = { 1, (int64_t)input_ids.size()}; // Batch size of 1

        // Create memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor(
            memory_info,
            input_ids.data(),
            input_ids.size() * sizeof(int32_t),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);


        const char* input_names[] = { "input_ids" };
        const char* output_names[] = { "last_hidden_state", "pooler_output" };

        std::cout << "Running inference..." << std::endl;
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1, // Number of inputs
            output_names,
            2  // Number of outputs
        );

        

        Ort::Float16_t* last_hidden_state_ptr        = output_tensors[0].GetTensorMutableData<Ort::Float16_t>();
        std::vector<int64_t> last_hidden_state_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t last_hidden_state_num_tokens          = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();


        for (size_t i = 0; i < last_hidden_state_num_tokens; i++) {
            last_hidden_state.push_back(last_hidden_state_ptr[i]);
        }


        Ort::Float16_t* pooler_output_ptr        = output_tensors[1].GetTensorMutableData<Ort::Float16_t>();
        std::vector<int64_t> pooler_output_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        size_t pooler_output_num_tokens          = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();

        for (size_t i = 0; i < pooler_output_num_tokens; i++) {
            pooler_output.push_back(pooler_output_ptr[i]);
        }

        std::cout << "finish inference..." << std::endl;


    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    }

}
std::vector<int32_t> convert_int64_to_int32(std::vector<int64_t> input)
{
    std::vector<int32_t> output;
    output.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output.push_back(input[i]);
    }
    return output;
}

int main() {


    std::vector<int64_t> tokenizer_output = tokenize("a big cat");

    std::vector<int32_t> tokenizer_convert = convert_int64_to_int32(tokenizer_output);

    std::vector<Ort::Float16_t> last_hidden_state;
    std::vector<Ort::Float16_t> pooler_output;

    text_encode(tokenizer_convert, last_hidden_state, pooler_output);


    return 0;
}