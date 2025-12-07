#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>

// ONNX Runtime Header
#include <onnxruntime_cxx_api.h>

// STB Image for loading (Header-only library)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Helper to load ImageNet labels
std::vector<std::string> loadLabels(const std::string& filename) {
    std::vector<std::string> labels;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) labels.push_back(line);
    return labels;
}

// Softmax function to convert logits to probabilities
void softmax(std::vector<float>& input) {
    float max = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (auto& val : input) {
        val = std::exp(val - max);
        sum += val;
    }
    for (auto& val : input) val /= sum;
}

int main() {
    // 1. Settings
    const std::string model_path = "resnet50-v2-7.onnx";
    const std::string image_path = "dog.jpg"; // Must be 224x224 for this simple example
    const std::string label_path = "imagenet_classes.txt";

    // 2. Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNet50");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    // 3. Load and Preprocess Image
    int width, height, channels;
    // Load image (force 3 channels: R, G, B)
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    
    if (!img_data) {
        std::cerr << "Failed to load image. Make sure " << image_path << " exists." << std::endl;
        return 1;
    }
    if (width != 224 || height != 224) {
        std::cerr << "Error: Image must be 224x224 for this simple example." << std::endl;
        stbi_image_free(img_data);
        return 1;
    }

    // Prepare input tensor buffers
    // ResNet expects: 1 Batch, 3 Channels, 224 Height, 224 Width
    std::vector<int64_t> input_node_dims = {1, 3, 224, 224};
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);

    // Normalization parameters for ResNet (ImageNet standards)
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[]  = {0.229f, 0.224f, 0.225f};

    // HWC to CHW Conversion + Normalization
    // stbi_load loads pixels as RGBRGB... (HWC)
    // ONNX needs RRR...GGG...BBB... (CHW)
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int pixel_index = (h * width + w) * 3;
            
            // Normalize and store in planar (CHW) format
            for (int c = 0; c < 3; c++) {
                float val = img_data[pixel_index + c] / 255.0f; // Scale 0-1
                val = (val - mean[c]) / std[c];                 // Normalize
                
                // Calculate index for CHW layout
                // Index = (Channel * Height * Width) + (HeightIndex * Width) + WidthIndex
                int target_index = (c * height * width) + (h * width) + w;
                input_tensor_values[target_index] = val;
            }
        }
    }
    stbi_image_free(img_data);

    // 4. Create Tensor objects
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, 
        input_node_dims.data(), input_node_dims.size());

    // 5. Run Inference
    const char* input_names[] = {"data"};   // Ensure this matches your ONNX model input name
    const char* output_names[] = {"resnetv24_dense0_fwd"}; // Ensure this matches your ONNX model output name
    
    // NOTE: If you are unsure of names, use Neutron (netron.app) to inspect your .onnx file
    
    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
    );

    // 6. Post-process (Get Top-1 Class)
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    
    // Copy to vector to use softmax and algorithms
    std::vector<float> output_probs(floatarr, floatarr + output_count);
    
    // Apply Softmax (optional, makes it a percentage)
    softmax(output_probs);

    // Find the index of the max element
    auto max_it = std::max_element(output_probs.begin(), output_probs.end());
    int max_index = std::distance(output_probs.begin(), max_it);
    float max_prob = *max_it;

    // Load labels and print result
    auto labels = loadLabels(label_path);
    std::cout << "Predicted Class ID: " << max_index << std::endl;
    if (max_index < labels.size()) {
        std::cout << "Label: " << labels[max_index] << std::endl;
    }
    std::cout << "Confidence: " << max_prob * 100.0f << "%" << std::endl;

    return 0;
}