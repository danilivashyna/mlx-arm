// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/safetensors.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdint>

namespace mlx::core {

// Helper to parse minimalistic JSON structure for Safetensors
// WARN: This is NOT a full JSON parser. It's tailored for flat safetensors headers.
struct TensorInfo {
    std::vector<int> shape;
    size_t start_offset;
    size_t end_offset;
    std::string dtype;
};

std::string extract_string(const std::string& json, size_t& pos) {
    size_t start = json.find('"', pos);
    if (start == std::string::npos) return "";
    size_t end = json.find('"', start + 1);
    pos = end + 1;
    return json.substr(start + 1, end - start - 1);
}

void skip_whitespace(const std::string& json, size_t& pos) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) {
        pos++;
    }
}

std::map<std::string, TensorInfo> parse_header(const std::string& header) {
    std::map<std::string, TensorInfo> info;
    size_t pos = 0;
    
    skip_whitespace(header, pos);
    if (header[pos] != '{') return info;
    pos++;

    while (pos < header.size()) {
        skip_whitespace(header, pos);
        if (header[pos] == '}') break;

        // Key (tensor name)
        std::string key = extract_string(header, pos);
        if (key.empty() || key == "__metadata__") {
            // Skip metadata or invalid keys
            // Simple skip logic: find matching brace
            skip_whitespace(header, pos);
            if (header[pos] == ':') pos++;
            skip_whitespace(header, pos);
            if (header[pos] == '{') {
                int depth = 1;
                pos++;
                while (depth > 0 && pos < header.size()) {
                    if (header[pos] == '{') depth++;
                    if (header[pos] == '}') depth--;
                    pos++;
                }
            }
            skip_whitespace(header, pos);
            if (header[pos] == ',') pos++;
            continue;
        }

        skip_whitespace(header, pos);
        if (header[pos] != ':') break; // Error
        pos++;
        skip_whitespace(header, pos);
        
        // Value object
        if (header[pos] != '{') break; // Error
        pos++;

        TensorInfo ti;
        while (pos < header.size()) {
            skip_whitespace(header, pos);
            if (header[pos] == '}') { pos++; break; } 
            
            std::string field = extract_string(header, pos);
            skip_whitespace(header, pos);
            if (header[pos] == ':') pos++;
            skip_whitespace(header, pos);

            if (field == "dtype") {
                ti.dtype = extract_string(header, pos);
            } else if (field == "shape") {
                if (header[pos] == '[') {
                    pos++;
                    while (pos < header.size() && header[pos] != ']') {
                        skip_whitespace(header, pos);
                        if (header[pos] == ']') break;
                        if (header[pos] == ',') { pos++; continue; }
                        // Parse number
                        size_t next_comma = header.find_first_of(",]", pos);
                        std::string num_str = header.substr(pos, next_comma - pos);
                        ti.shape.push_back(std::stoi(num_str));
                        pos = next_comma;
                    }
                    pos++; // skip ]
                }
            } else if (field == "data_offsets") {
                if (header[pos] == '[') {
                    pos++;
                    skip_whitespace(header, pos);
                    size_t next_comma = header.find(',', pos);
                    ti.start_offset = std::stoll(header.substr(pos, next_comma - pos));
                    pos = next_comma + 1;
                    skip_whitespace(header, pos);
                    size_t next_bracket = header.find(']', pos);
                    ti.end_offset = std::stoll(header.substr(pos, next_bracket - pos));
                    pos = next_bracket + 1;
                }
            } else {
                // Skip other fields
                // Hacky skip value
                while(pos < header.size() && header[pos] != ',' && header[pos] != '}') pos++;
            }
            
            skip_whitespace(header, pos);
            if (header[pos] == ',') pos++;
        }
        
        info[key] = ti;
        
        skip_whitespace(header, pos);
        if (header[pos] == ',') pos++;
    }
    return info;
}

std::map<std::string, Array> load_safetensors(const std::string& filename) {
    std::map<std::string, Array> weights;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return weights;
    }

    // Read header size (8 bytes, little endian uint64)
    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 8);
    
    // Read header
    std::string header_str(header_len, ' ');
    file.read(&header_str[0], header_len);
    
    auto tensors = parse_header(header_str);
    printf("DEBUG: Found %zu tensors in safetensors header.\n", tensors.size());

    // Base offset for data
    size_t data_begin = 8 + header_len;

    for (auto& [name, info] : tensors) {
        size_t size_bytes = info.end_offset - info.start_offset;
        
        // We only support F32 for now in Array.
        // If data is F16 or BF16, we need to convert.
        // For simplicity now, let's assume we load into float vector.
        
        std::vector<float> float_data;
        size_t num_elements = 1;
        for(int d : info.shape) num_elements *= d;
        float_data.resize(num_elements);

        file.seekg(data_begin + info.start_offset);
        
        if (info.dtype == "F32") {
            file.read(reinterpret_cast<char*>(float_data.data()), size_bytes);
        } else if (info.dtype == "F16") {
            // Very naive F16 to F32 conversion
            std::vector<uint16_t> fp16_data(num_elements);
            file.read(reinterpret_cast<char*>(fp16_data.data()), size_bytes);
            for(size_t i=0; i<num_elements; ++i) {
                // TODO: Proper FP16 conversion. For now, just cast if it were integer (WRONG)
                // Actually, without a library, proper F16 conversion is tedious.
                // Let's implement a minimal converter or just load as zeros if we can't.
                
                // Let's try to do a basic conversion using bit manipulation if possible
                // Or... since we are on ARM, maybe we have __fp16 support?
                // But standard C++ doesn't.
                
                // Placeholder: Load as small random to allow running, or 0.
                // Or implement a tiny lookup table/converter.
                
                // Let's implement a mini-converter for half-precision.
                uint16_t h = fp16_data[i];
                uint32_t s = (h >> 15) & 0x00000001;
                uint32_t e = (h >> 10) & 0x0000001f;
                uint32_t m = h & 0x000003ff;
                
                uint32_t val;
                if (e == 0) {
                    if (m == 0) {
                        val = s << 31;
                    } else {
                        // Denormal
                        while (!(m & 0x00000400)) {
                            m <<= 1;
                            e -= 1;
                        }
                        e += 1;
                        m &= ~0x00000400;
                        e = e + (127 - 15);
                        m <<= 13;
                        val = (s << 31) | (e << 23) | m;
                    }
                } else if (e == 31) {
                    if (m == 0) {
                        val = (s << 31) | 0x7f800000; // Inf
                    } else {
                        val = (s << 31) | 0x7f800000 | (m << 13); // NaN
                    }
                } else {
                    e = e + (127 - 15);
                    m = m << 13;
                    val = (s << 31) | (e << 23) | m;
                }
                
                float_data[i] = *reinterpret_cast<float*>(&val);
            }
        } else if (info.dtype == "BF16") {
             // BF16 is easy: just shift 16 bits
             std::vector<uint16_t> bf16_data(num_elements);
             file.read(reinterpret_cast<char*>(bf16_data.data()), size_bytes);
             for(size_t i=0; i<num_elements; ++i) {
                 uint32_t val = static_cast<uint32_t>(bf16_data[i]) << 16;
                 float_data[i] = *reinterpret_cast<float*>(&val);
             }
        } else {
            printf("WARNING: Unsupported dtype %s for tensor %s\n", info.dtype.c_str(), name.c_str());
        }

        weights[name] = Array(float_data, info.shape);
    }
    
    return weights;
}

} // namespace mlx::core
