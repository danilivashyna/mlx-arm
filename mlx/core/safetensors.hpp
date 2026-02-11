// Simple Safetensors Parser for MLX-ARM
// Loads weights from HuggingFace safetensors format
// Reference: https://huggingface.co/docs/safetensors/index

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <memory>

namespace mlx {
namespace safetensors {

enum class DType {
    F32,  // float32
    F16,  // float16
    I32,  // int32
    I8,   // int8
    U8    // uint8
};

struct Tensor {
    std::string name;
    DType dtype;
    std::vector<size_t> shape;
    size_t data_offset;  // Offset in file
    size_t data_size;    // Size in bytes
    std::shared_ptr<uint8_t[]> data;  // Loaded data
    
    size_t num_elements() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    
    size_t element_size() const {
        switch (dtype) {
            case DType::F32: return 4;
            case DType::F16: return 2;
            case DType::I32: return 4;
            case DType::I8: return 1;
            case DType::U8: return 1;
            default: return 0;
        }
    }
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    
    // Load safetensors file
    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open: %s\n", path.c_str());
            return false;
        }
        
        // Read header size (first 8 bytes, little-endian)
        uint64_t header_size;
        file.read(reinterpret_cast<char*>(&header_size), 8);
        
        // Read header JSON
        std::vector<char> header_json(header_size);
        file.read(header_json.data(), header_size);
        std::string header_str(header_json.begin(), header_json.end());
        
        // Parse header (simple JSON parsing for tensor metadata)
        if (!parse_header(header_str)) {
            fprintf(stderr, "Failed to parse header\n");
            return false;
        }
        
        // Data starts after 8-byte size + header
        size_t data_offset_base = 8 + header_size;
        
        // Load tensor data
        for (auto& [name, tensor] : tensors_) {
            file.seekg(data_offset_base + tensor.data_offset);
            tensor.data = std::shared_ptr<uint8_t[]>(new uint8_t[tensor.data_size]);
            file.read(reinterpret_cast<char*>(tensor.data.get()), tensor.data_size);
        }
        
        file.close();
        printf("✅ Loaded %zu tensors from %s\n", tensors_.size(), path.c_str());
        return true;
    }
    
    // Get tensor by name
    const Tensor* get_tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    // List all tensor names
    std::vector<std::string> list_tensors() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : tensors_) {
            names.push_back(name);
        }
        return names;
    }
    
    // Get tensor data as float (with conversion if needed)
    std::vector<float> get_tensor_float(const std::string& name) const {
        auto tensor = get_tensor(name);
        if (!tensor || !tensor->data) return {};
        
        std::vector<float> result(tensor->num_elements());
        
        if (tensor->dtype == DType::F32) {
            // Direct copy
            memcpy(result.data(), tensor->data.get(), result.size() * sizeof(float));
        } else if (tensor->dtype == DType::F16) {
            // Convert FP16 to FP32
            const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(tensor->data.get());
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = fp16_to_fp32(fp16_data[i]);
            }
        }
        
        return result;
    }
    
private:
    std::unordered_map<std::string, Tensor> tensors_;
    
    // Simple FP16 to FP32 conversion
    static float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1f;
        uint32_t mantissa = h & 0x3ff;
        
        if (exp == 0) {
            if (mantissa == 0) return sign ? -0.0f : 0.0f;
            // Subnormal
            float f = mantissa / 1024.0f * powf(2.0f, -14.0f);
            return sign ? -f : f;
        }
        if (exp == 31) return sign ? -INFINITY : INFINITY;
        
        float f = (1.0f + mantissa / 1024.0f) * powf(2.0f, (int)exp - 15);
        return sign ? -f : f;
    }
    
    // Parse JSON header (simplified - handles safetensors format)
    bool parse_header(const std::string& json) {
        // Simple JSON parser for safetensors metadata
        // Format: {"tensor_name": {"dtype": "F32", "shape": [M, N], "data_offsets": [start, end]}, ...}
        
        size_t pos = 0;
        while (pos < json.size()) {
            // Find tensor name
            size_t name_start = json.find("\"", pos);
            if (name_start == std::string::npos) break;
            name_start++;
            
            size_t name_end = json.find("\"", name_start);
            if (name_end == std::string::npos) break;
            
            std::string name = json.substr(name_start, name_end - name_start);
            
            // Skip metadata tensor
            if (name == "__metadata__") {
                pos = json.find("}", name_end) + 1;
                continue;
            }
            
            Tensor tensor;
            tensor.name = name;
            
            // Find dtype
            size_t dtype_pos = json.find("\"dtype\"", name_end);
            if (dtype_pos != std::string::npos) {
                size_t dtype_start = json.find("\"", dtype_pos + 7) + 1;
                size_t dtype_end = json.find("\"", dtype_start);
                std::string dtype_str = json.substr(dtype_start, dtype_end - dtype_start);
                
                if (dtype_str == "F32") tensor.dtype = DType::F32;
                else if (dtype_str == "F16") tensor.dtype = DType::F16;
                else if (dtype_str == "I32") tensor.dtype = DType::I32;
                else if (dtype_str == "I8") tensor.dtype = DType::I8;
                else if (dtype_str == "U8") tensor.dtype = DType::U8;
            }
            
            // Find shape
            size_t shape_pos = json.find("\"shape\"", dtype_pos);
            if (shape_pos != std::string::npos) {
                size_t shape_start = json.find("[", shape_pos);
                size_t shape_end = json.find("]", shape_start);
                std::string shape_str = json.substr(shape_start + 1, shape_end - shape_start - 1);
                
                // Parse dimensions
                size_t dim_pos = 0;
                while (dim_pos < shape_str.size()) {
                    size_t comma = shape_str.find(",", dim_pos);
                    if (comma == std::string::npos) comma = shape_str.size();
                    
                    std::string dim_str = shape_str.substr(dim_pos, comma - dim_pos);
                    // Trim whitespace
                    size_t first = dim_str.find_first_not_of(" \t\n\r");
                    size_t last = dim_str.find_last_not_of(" \t\n\r");
                    if (first != std::string::npos) {
                        dim_str = dim_str.substr(first, last - first + 1);
                        tensor.shape.push_back(std::stoull(dim_str));
                    }
                    
                    dim_pos = comma + 1;
                }
            }
            
            // Find data_offsets
            size_t offsets_pos = json.find("\"data_offsets\"", shape_pos);
            if (offsets_pos != std::string::npos) {
                size_t offsets_start = json.find("[", offsets_pos);
                size_t offsets_end = json.find("]", offsets_start);
                std::string offsets_str = json.substr(offsets_start + 1, offsets_end - offsets_start - 1);
                
                size_t comma = offsets_str.find(",");
                std::string start_str = offsets_str.substr(0, comma);
                std::string end_str = offsets_str.substr(comma + 1);
                
                // Trim whitespace
                size_t first = start_str.find_first_not_of(" \t\n\r");
                size_t last = start_str.find_last_not_of(" \t\n\r");
                if (first != std::string::npos) {
                    start_str = start_str.substr(first, last - first + 1);
                }
                
                first = end_str.find_first_not_of(" \t\n\r");
                last = end_str.find_last_not_of(" \t\n\r");
                if (first != std::string::npos) {
                    end_str = end_str.substr(first, last - first + 1);
                }
                
                tensor.data_offset = std::stoull(start_str);
                size_t data_end = std::stoull(end_str);
                tensor.data_size = data_end - tensor.data_offset;
            }
            
            tensors_[name] = tensor;
            pos = json.find("}", offsets_pos) + 1;
        }
        
        return !tensors_.empty();
    }
};

} // namespace safetensors
} // namespace mlx
