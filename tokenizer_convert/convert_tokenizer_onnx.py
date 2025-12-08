# pip install onnx onnxruntime onnxruntime-extensions transformers
from onnxruntime_extensions import gen_processing_models
from transformers import AutoTokenizer, CLIPTokenizer

# 1. Load your desired tokenizer (e.g., BERT, GPT-2, Llama)
model_name = "D:\\Models\\stable-diffusion-1.5_io16_amdgpu\\tokenizer\\" 
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)


text = "A photo of a cat"
res = tokenizer.encode(text)
print(res)
# help(gen_processing_models)
# 2. Convert and save it as an ONNX model
# This creates a model that takes text strings and outputs token IDs
# gen_processing_models(tokenizer, pre_kwargs={}, post_kwargs={}, output_dir=".")


tok_encode, _ = gen_processing_models(tokenizer, pre_kwargs={})

onnx_tokenizer_path = "tokenizer.onnx"
with open(onnx_tokenizer_path, "wb") as f:
    f.write(tok_encode.SerializeToString())

print("Created 'tokenizer.onnx' in the current directory.")