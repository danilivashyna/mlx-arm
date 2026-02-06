// Simple Tokenizer for LLaMA models
// Initially supports basic BPE-style encoding/decoding
// TODO: Integrate full SentencePiece implementation

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace mlx {
namespace tokenizer {

class LlamaTokenizer {
public:
    LlamaTokenizer() {
        // Initialize special tokens
        special_tokens_["<s>"] = 1;       // BOS
        special_tokens_["</s>"] = 2;      // EOS
        special_tokens_["<unk>"] = 0;     // Unknown
        
        // Common tokens (mock - real tokenizer uses SentencePiece)
        vocab_[1] = "<s>";
        vocab_[2] = "</s>";
        vocab_[0] = "<unk>";
        
        // Some common words (for demo)
        vocab_[29871] = "▁";  // Space
        vocab_[15043] = "Hello";
        vocab_[29892] = ",";
        vocab_[920] = "how";
        vocab_[526] = "are";
        vocab_[366] = "you";
        vocab_[29973] = "?";
        vocab_[306] = "I";
        vocab_[29915] = "'";
        vocab_[29885] = "m";
        vocab_[263] = "a";
        vocab_[4086] = "language";
        vocab_[1904] = "model";
        vocab_[2020] = "AI";
        vocab_[20255] = "assistant";
        vocab_[1576] = "The";
        vocab_[7375] = "cat";
        vocab_[3786] = "dog";
        vocab_[338] = "is";
        vocab_[6350] = "running";
    }
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text, bool add_bos = true) {
        std::vector<int> tokens;
        
        if (add_bos) {
            tokens.push_back(1);  // BOS
        }
        
        // Simple word-based tokenization (mock)
        // Real implementation would use SentencePiece BPE
        size_t pos = 0;
        std::string current_word;
        
        while (pos < text.size()) {
            char c = text[pos];
            
            if (c == ' ') {
                if (!current_word.empty()) {
                    int token_id = find_token_id(current_word);
                    if (token_id >= 0) tokens.push_back(token_id);
                    current_word.clear();
                }
                // Space token
                tokens.push_back(29871);
            } else if (c == ',' || c == '.' || c == '!' || c == '?') {
                if (!current_word.empty()) {
                    int token_id = find_token_id(current_word);
                    if (token_id >= 0) tokens.push_back(token_id);
                    current_word.clear();
                }
                // Punctuation
                if (c == ',') tokens.push_back(29892);
                else if (c == '?') tokens.push_back(29973);
                else tokens.push_back(0);  // Unknown
            } else {
                current_word += c;
            }
            
            pos++;
        }
        
        if (!current_word.empty()) {
            int token_id = find_token_id(current_word);
            if (token_id >= 0) tokens.push_back(token_id);
        }
        
        return tokens;
    }
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens, bool skip_special = true) {
        std::string result;
        
        for (int token_id : tokens) {
            if (skip_special && (token_id == 1 || token_id == 2)) {
                continue;  // Skip BOS/EOS
            }
            
            auto it = vocab_.find(token_id);
            if (it != vocab_.end()) {
                std::string token_str = it->second;
                
                // Handle special space token
                if (token_str == "▁") {
                    result += " ";
                } else {
                    result += token_str;
                }
            } else {
                result += "<UNK>";
            }
        }
        
        return result;
    }
    
    // Get vocabulary size
    int vocab_size() const {
        return 32000;  // LLaMA vocab size
    }
    
    // Get BOS token
    int bos_token() const {
        return 1;
    }
    
    // Get EOS token
    int eos_token() const {
        return 2;
    }
    
private:
    std::unordered_map<std::string, int> special_tokens_;
    std::unordered_map<int, std::string> vocab_;
    
    int find_token_id(const std::string& word) {
        // Reverse lookup (inefficient but ok for mock)
        for (const auto& [id, token] : vocab_) {
            if (token == word) return id;
        }
        return 0;  // Unknown
    }
};

} // namespace tokenizer
} // namespace mlx
