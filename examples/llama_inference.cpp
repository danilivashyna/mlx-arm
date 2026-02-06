// LLaMA Inference Test - FIRST TOKEN GENERATION!
// Simple inference pipeline with random weights as proof-of-concept
// Next step: Load real LLaMA weights from safetensors

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>

// Simple config (TinyLlama-like for testing)
struct LlamaConfig {
    int vocab_size = 32000;
    int hidden_dim = 512;      // Small for testing
    int num_layers = 1;        // Start with 1 layer
    int num_heads = 8;
    int head_dim = hidden_dim / num_heads;  // 64
    int intermediate_dim = 1024;  // FFN intermediate
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

// Mock tokenizer (hardcoded for testing)
std::vector<int> tokenize(const char* text) {
    // For now, just return some token IDs
    // Real implementation would use SentencePiece/BPE
    if (strcmp(text, "Hello") == 0) return {1, 15043};  // BOS + "Hello"
    if (strcmp(text, "The") == 0) return {1, 450};      // BOS + "The"
    return {1, 1000, 2000};  // Default: BOS + random tokens
}

const char* detokenize(int token_id) {
    // Mock detokenizer
    if (token_id == 1) return "<BOS>";
    if (token_id == 15043) return "Hello";
    if (token_id == 3186) return " world";
    if (token_id == 450) return "The";
    if (token_id == 4799) return " cat";
    return "<UNK>";
}

// RMSNorm (CPU reference)
void rms_norm_cpu(std::vector<float>& x, const std::vector<float>& weight, 
                  int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / size + eps);
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] / rms) * weight[i];
    }
}

// Simple matmul (CPU reference)
void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// SiLU activation
void silu_cpu(std::vector<float>& x) {
    for (auto& v : x) {
        v = v / (1.0f + expf(-v));
    }
}

// Element-wise multiply
void elemwise_mul_cpu(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); i++) {
        a[i] *= b[i];
    }
}

// Softmax
void softmax_cpu(std::vector<float>& x, int size) {
    float max_val = *std::max_element(x.begin(), x.begin() + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Simple RoPE (CPU)
void rope_cpu(std::vector<float>& x, int seq_len, int num_heads, int head_dim, 
              float theta_base) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < head_dim; i += 2) {
                int idx = (pos * num_heads + h) * head_dim + i;
                float x0 = x[idx];
                float x1 = x[idx + 1];
                
                float freq = powf(theta_base, -(float)i / head_dim);
                float theta = pos * freq;
                float cos_theta = cosf(theta);
                float sin_theta = sinf(theta);
                
                x[idx] = x0 * cos_theta - x1 * sin_theta;
                x[idx + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

// Attention (simplified single-head for now)
void attention_cpu(const std::vector<float>& Q, const std::vector<float>& K,
                   const std::vector<float>& V, std::vector<float>& out,
                   int seq_len, int head_dim) {
    // Q×K^T
    std::vector<float> scores(seq_len * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                sum += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            scores[i * seq_len + j] = sum / sqrtf((float)head_dim);
        }
    }
    
    // Softmax each row
    for (int i = 0; i < seq_len; i++) {
        softmax_cpu(scores, seq_len);
    }
    
    // Attention × V
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                sum += scores[i * seq_len + k] * V[k * head_dim + j];
            }
            out[i * head_dim + j] = sum;
        }
    }
}

// Initialize random weights
void init_random_weights(std::vector<float>& w, size_t size) {
    for (size_t i = 0; i < size; i++) {
        w[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;  // Small random values
    }
}

// Transformer layer (simplified)
void transformer_layer(std::vector<float>& x, int seq_len, const LlamaConfig& cfg,
                      const std::vector<float>& attn_norm_weight,
                      const std::vector<float>& q_weight,
                      const std::vector<float>& k_weight,
                      const std::vector<float>& v_weight,
                      const std::vector<float>& o_weight,
                      const std::vector<float>& ffn_norm_weight,
                      const std::vector<float>& gate_weight,
                      const std::vector<float>& up_weight,
                      const std::vector<float>& down_weight) {
    
    printf("  Transformer Layer:\n");
    auto layer_start = std::chrono::high_resolution_clock::now();
    
    // === ATTENTION ===
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> residual = x;
    
    // Pre-norm
    rms_norm_cpu(x, attn_norm_weight, cfg.hidden_dim, cfg.rms_norm_eps);
    
    // Q, K, V projections (simplified - single head)
    std::vector<float> Q(seq_len * cfg.head_dim);
    std::vector<float> K(seq_len * cfg.head_dim);
    std::vector<float> V(seq_len * cfg.head_dim);
    
    matmul_cpu(x, q_weight, Q, seq_len, cfg.head_dim, cfg.hidden_dim);
    matmul_cpu(x, k_weight, K, seq_len, cfg.head_dim, cfg.hidden_dim);
    matmul_cpu(x, v_weight, V, seq_len, cfg.head_dim, cfg.hidden_dim);
    
    // RoPE
    rope_cpu(Q, seq_len, 1, cfg.head_dim, cfg.rope_theta);
    rope_cpu(K, seq_len, 1, cfg.head_dim, cfg.rope_theta);
    
    // Attention
    std::vector<float> attn_out(seq_len * cfg.head_dim);
    attention_cpu(Q, K, V, attn_out, seq_len, cfg.head_dim);
    
    // Output projection
    std::vector<float> attn_final(seq_len * cfg.hidden_dim);
    matmul_cpu(attn_out, o_weight, attn_final, seq_len, cfg.hidden_dim, cfg.head_dim);
    
    // Residual
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = residual[i] + attn_final[i];
    }
    
    auto attn_end = std::chrono::high_resolution_clock::now();
    printf("    Attention: %.2f ms\n", 
           std::chrono::duration<double, std::milli>(attn_end - start).count());
    
    // === FFN ===
    start = std::chrono::high_resolution_clock::now();
    residual = x;
    
    // Pre-norm
    rms_norm_cpu(x, ffn_norm_weight, cfg.hidden_dim, cfg.rms_norm_eps);
    
    // Gate + Up projections
    std::vector<float> gate(seq_len * cfg.intermediate_dim);
    std::vector<float> up(seq_len * cfg.intermediate_dim);
    
    matmul_cpu(x, gate_weight, gate, seq_len, cfg.intermediate_dim, cfg.hidden_dim);
    matmul_cpu(x, up_weight, up, seq_len, cfg.intermediate_dim, cfg.hidden_dim);
    
    // SiLU on gate
    silu_cpu(gate);
    
    // Element-wise multiply
    elemwise_mul_cpu(gate, up);
    
    // Down projection
    std::vector<float> ffn_out(seq_len * cfg.hidden_dim);
    matmul_cpu(gate, down_weight, ffn_out, seq_len, cfg.hidden_dim, cfg.intermediate_dim);
    
    // Residual
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = residual[i] + ffn_out[i];
    }
    
    auto ffn_end = std::chrono::high_resolution_clock::now();
    printf("    FFN: %.2f ms\n", 
           std::chrono::duration<double, std::milli>(ffn_end - start).count());
    
    auto layer_end = std::chrono::high_resolution_clock::now();
    printf("    Total Layer: %.2f ms\n\n", 
           std::chrono::duration<double, std::milli>(layer_end - layer_start).count());
}

// Argmax sampling
int sample_token(const std::vector<float>& logits, int vocab_size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void generate_tokens(const char* prompt, int num_new_tokens) {
    printf("🚀 LLaMA Inference Test\n");
    printf("======================\n\n");
    
    LlamaConfig cfg;
    printf("Config:\n");
    printf("  hidden_dim: %d\n", cfg.hidden_dim);
    printf("  num_heads: %d\n", cfg.num_heads);
    printf("  head_dim: %d\n", cfg.head_dim);
    printf("  intermediate_dim: %d\n", cfg.intermediate_dim);
    printf("  vocab_size: %d\n\n", cfg.vocab_size);
    
    srand(42);
    
    // Initialize random weights
    printf("Initializing random weights...\n");
    std::vector<float> embed_weight(cfg.vocab_size * cfg.hidden_dim);
    std::vector<float> attn_norm_weight(cfg.hidden_dim);
    std::vector<float> q_weight(cfg.hidden_dim * cfg.head_dim);
    std::vector<float> k_weight(cfg.hidden_dim * cfg.head_dim);
    std::vector<float> v_weight(cfg.hidden_dim * cfg.head_dim);
    std::vector<float> o_weight(cfg.head_dim * cfg.hidden_dim);
    std::vector<float> ffn_norm_weight(cfg.hidden_dim);
    std::vector<float> gate_weight(cfg.hidden_dim * cfg.intermediate_dim);
    std::vector<float> up_weight(cfg.hidden_dim * cfg.intermediate_dim);
    std::vector<float> down_weight(cfg.intermediate_dim * cfg.hidden_dim);
    std::vector<float> lm_head_weight(cfg.hidden_dim * cfg.vocab_size);
    std::vector<float> final_norm_weight(cfg.hidden_dim);
    
    init_random_weights(embed_weight, embed_weight.size());
    init_random_weights(attn_norm_weight, attn_norm_weight.size());
    init_random_weights(q_weight, q_weight.size());
    init_random_weights(k_weight, k_weight.size());
    init_random_weights(v_weight, v_weight.size());
    init_random_weights(o_weight, o_weight.size());
    init_random_weights(ffn_norm_weight, ffn_norm_weight.size());
    init_random_weights(gate_weight, gate_weight.size());
    init_random_weights(up_weight, up_weight.size());
    init_random_weights(down_weight, down_weight.size());
    init_random_weights(lm_head_weight, lm_head_weight.size());
    init_random_weights(final_norm_weight, final_norm_weight.size());
    
    for (size_t i = 0; i < attn_norm_weight.size(); i++) attn_norm_weight[i] = 1.0f;
    for (size_t i = 0; i < ffn_norm_weight.size(); i++) ffn_norm_weight[i] = 1.0f;
    for (size_t i = 0; i < final_norm_weight.size(); i++) final_norm_weight[i] = 1.0f;
    
    printf("Done!\n\n");
    
    // Tokenize prompt
    std::vector<int> tokens = tokenize(prompt);
    printf("Prompt: \"%s\"\n", prompt);
    printf("Tokens: [");
    for (size_t i = 0; i < tokens.size(); i++) {
        printf("%d%s", tokens[i], i < tokens.size()-1 ? ", " : "");
    }
    printf("]\n\n");
    
    printf("Generating %d new tokens:\n", num_new_tokens);
    printf("==========================\n\n");
    
    // Generation loop
    for (int gen_idx = 0; gen_idx < num_new_tokens; gen_idx++) {
        printf("Generation step %d:\n", gen_idx + 1);
        auto gen_start = std::chrono::high_resolution_clock::now();
        
        int seq_len = tokens.size();
        
        // Embedding lookup
        std::vector<float> x(seq_len * cfg.hidden_dim);
        for (int i = 0; i < seq_len; i++) {
            int token_id = tokens[i];
            for (int j = 0; j < cfg.hidden_dim; j++) {
                x[i * cfg.hidden_dim + j] = embed_weight[token_id * cfg.hidden_dim + j];
            }
        }
        
        // Transformer layer
        transformer_layer(x, seq_len, cfg,
                         attn_norm_weight, q_weight, k_weight, v_weight, o_weight,
                         ffn_norm_weight, gate_weight, up_weight, down_weight);
        
        // Final norm (last token only for generation)
        std::vector<float> last_hidden(cfg.hidden_dim);
        for (int i = 0; i < cfg.hidden_dim; i++) {
            last_hidden[i] = x[(seq_len - 1) * cfg.hidden_dim + i];
        }
        rms_norm_cpu(last_hidden, final_norm_weight, cfg.hidden_dim, cfg.rms_norm_eps);
        
        // LM head
        std::vector<float> logits(cfg.vocab_size);
        matmul_cpu(last_hidden, lm_head_weight, logits, 1, cfg.vocab_size, cfg.hidden_dim);
        
        // Sample
        int next_token = sample_token(logits, cfg.vocab_size);
        tokens.push_back(next_token);
        
        auto gen_end = std::chrono::high_resolution_clock::now();
        double gen_time = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
        
        printf("  Generated token: %d (%s)\n", next_token, detokenize(next_token));
        printf("  Time: %.2f ms\n", gen_time);
        printf("  Tokens/sec: %.2f\n\n", 1000.0 / gen_time);
    }
    
    // Print full sequence
    printf("\n🎉 INFERENCE COMPLETE!\n");
    printf("======================\n");
    printf("Full sequence: ");
    for (int token : tokens) {
        printf("%s ", detokenize(token));
    }
    printf("\n");
}

int main() {
    generate_tokens("Hello", 3);
    
    printf("\n✅ First token generation successful!\n");
    printf("Next step: Load real LLaMA weights!\n");
    
    return 0;
}
