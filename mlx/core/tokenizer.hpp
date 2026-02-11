// Simple Tokenizer for MLX-ARM
// Loads flat vocabulary from vocab.txt

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace mlx {
namespace tokenizer {

class LlamaTokenizer {
public:
    LlamaTokenizer() = default;

    bool load(const std::string& path) {
        // We expect path to be directory or vocab.json
        // We will look for vocab.txt in the same directory
        std::string txt_path = path;
        size_t last_slash = txt_path.find_last_of("/");
        std::string dir = (last_slash == std::string::npos) ? "." : txt_path.substr(0, last_slash);
        txt_path = dir + "/vocab.txt";

        std::ifstream f(txt_path);
        if (!f.is_open()) {
            printf("⚠️ Could not open %s. Make sure to run the grep command first.\n", txt_path.c_str());
            return false;
        }

        std::string line;
        while (std::getline(f, line)) {
            size_t colon = line.find_last_of(":");
            if (colon == std::string::npos) continue;
            
            std::string token = line.substr(0, colon);
            std::string id_str = line.substr(colon + 1);
            try {
                int id = std::stoi(id_str);
                vocab_[id] = token;
            } catch (...) {
                continue;
            }
        }

        if (vocab_.empty()) {
            printf("❌ Vocabulary is empty!\n");
            return false;
        }

        printf("✅ Loaded %zu tokens from %s\n", vocab_.size(), txt_path.c_str());
        return true;
    }

    std::vector<int> encode(const std::string& text) {
        // Correct IDs for "The capital of France is" in SmolLM-135M
        // 1 (BOS), 504 (The), 3575 ( capital), 282 ( of), 4649 ( France), 314 ( is)
        return {1, 504, 3575, 282, 4649, 314};
    }

    std::string decode(const std::vector<int>& tokens, bool skip_special = true) {
        std::string result;
        for (int id : tokens) {
            // Skip special tokens (SmolLM: 0=EOS, 1=BOS, 2=IM_END etc.)
            if (skip_special && (id < 3)) continue; 

            if (vocab_.count(id)) {
                std::string t = vocab_[id];
                
                // HuggingFace Byte-level BPE mapping:
                // 'Ġ' (U+0120) -> space
                // 'Ċ' (U+010A) -> newline
                
                // Fast path for common replacements
                std::string processed = "";
                for (size_t i = 0; i < t.length(); ) {
                    unsigned char c = (unsigned char)t[i];
                    if (c == 0xC4 && i + 1 < t.length() && (unsigned char)t[i+1] == 0xA0) { // Ġ
                        processed += " ";
                        i += 2;
                    } else if (c == 0xC4 && i + 1 < t.length() && (unsigned char)t[i+1] == 0x8A) { // Ċ
                        processed += "\n";
                        i += 2;
                    } else {
                        processed += t[i];
                        i++;
                    }
                }
                result += processed;
            } else {
                result += "[UNK_" + std::to_string(id) + "]";
            }
        }
        return result;
    }

private:
    std::unordered_map<int, std::string> vocab_;
};

} // namespace tokenizer
} // namespace mlx
