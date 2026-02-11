#pragma once

#include <vector>
#include <cstdint>
#include <cstdio>

namespace llm {

/**
 * KV-Cache для autoregressive LLM generation
 * 
 * Проблема без cache:
 *   Step 1: Compute K,V для tokens [0]
 *   Step 2: Compute K,V для tokens [0,1]  <- пересчёт!
 *   Step 3: Compute K,V для tokens [0,1,2] <- пересчёт!
 *   Complexity: O(N²)
 * 
 * С cache:
 *   Step 1: Compute K,V для [0] → save в cache
 *   Step 2: Compute K,V для [1] → append к cache, reuse [0]
 *   Step 3: Compute K,V для [2] → append к cache, reuse [0,1]
 *   Complexity: O(N)
 * 
 * Expected speedup: 5-10x для generation!
 */
class KVCache {
public:
    /**
     * @param num_layers Количество transformer layers
     * @param num_heads Количество attention heads
     * @param head_dim Размерность каждого head
     * @param max_seq_len Максимальная длина sequence
     */
    KVCache(uint32_t num_layers, 
            uint32_t num_heads,
            uint32_t head_dim,
            uint32_t max_seq_len)
        : num_layers_(num_layers)
        , num_heads_(num_heads)
        , head_dim_(head_dim)
        , max_seq_len_(max_seq_len)
        , current_length_(0)
    {
        // Allocate cache для всех layers
        k_cache_.resize(num_layers);
        v_cache_.resize(num_layers);
        
        size_t cache_size = max_seq_len * num_heads * head_dim;
        
        for (uint32_t l = 0; l < num_layers; l++) {
            // Reserve max capacity: [max_seq_len, num_heads, head_dim]
            k_cache_[l].reserve(cache_size);
            v_cache_[l].reserve(cache_size);
        }
        
        printf("✅ KV-cache allocated: %u layers, %u heads, %u head_dim, max_seq=%u\n",
               num_layers, num_heads, head_dim, max_seq_len);
        printf("   Memory per layer: %.2f MB\n",
               (cache_size * sizeof(float) * 2) / (1024.0 * 1024.0));
        fflush(stdout);
    }
    
    /**
     * Append new K, V vectors для одного token на указанном layer
     * @param layer_idx Index of transformer layer (0..num_layers-1)
     * @param k_new K vector для нового token [num_heads * head_dim]
     * @param v_new V vector для нового token [num_heads * head_dim]
     */
    void append(uint32_t layer_idx,
                const std::vector<float>& k_new,
                const std::vector<float>& v_new) {
        
        // ✅ BOUNDS CHECK
        if (layer_idx >= num_layers_) {
            printf("❌ [KVCache] Invalid layer_idx: %u >= %u\n",
                   layer_idx, num_layers_);
            return;
        }
        
        size_t expected_size = num_heads_ * head_dim_;
        
        if (k_new.size() != expected_size) {
            printf("❌ [KVCache] k_new size: got %zu, expected %zu\n",
                   k_new.size(), expected_size);
            return;
        }
        
        if (v_new.size() != expected_size) {
            printf("❌ [KVCache] v_new size: got %zu, expected %zu\n",
                   v_new.size(), expected_size);
            return;
        }
        
        if (current_length_ >= max_seq_len_) {
            printf("❌ [KVCache] Cache full: %u >= %u\n",
                   current_length_, max_seq_len_);
            return;
        }
        
        // Append к cache
        k_cache_[layer_idx].insert(
            k_cache_[layer_idx].end(),
            k_new.begin(), k_new.end()
        );
        
        v_cache_[layer_idx].insert(
            v_cache_[layer_idx].end(),
            v_new.begin(), v_new.end()
        );
        
        // Update length только на первом layer (один раз за шаг)
        if (layer_idx == 0) {
            current_length_++;
        }
    }
    
    /**
     * Append batch K,V (для prefill с несколькими tokens)
     * @param layer_idx Index of transformer layer
     * @param k_batch K matrix [batch_size, num_heads * head_dim]
     * @param v_batch V matrix [batch_size, num_heads * head_dim]
     * @param batch_size Количество tokens в batch
     */
    void append_batch(uint32_t layer_idx,
                      const float* k_batch,
                      const float* v_batch,
                      uint32_t batch_size) {
        
        if (layer_idx >= num_layers_) {
            fprintf(stderr, "⚠️ KVCache::append_batch: invalid layer_idx %u\n", layer_idx);
            return;
        }
        
        size_t kv_size = num_heads_ * head_dim_;
        size_t total_size = batch_size * kv_size;
        
        // Append весь batch
        k_cache_[layer_idx].insert(
            k_cache_[layer_idx].end(),
            k_batch, k_batch + total_size
        );
        
        v_cache_[layer_idx].insert(
            v_cache_[layer_idx].end(),
            v_batch, v_batch + total_size
        );
        
        if (layer_idx == 0) {
            current_length_ += batch_size;
            // Logging removed for speed
        }
    }
    
    /**
     * Get полный K cache для указанного layer
     * @return Pointer to K cache: [current_length, num_heads, head_dim]
     */
    const float* get_k(uint32_t layer_idx) const {
        if (layer_idx >= num_layers_) return nullptr;
        return k_cache_[layer_idx].data();
    }
    
    /**
     * Get полный V cache для указанного layer
     * @return Pointer to V cache: [current_length, num_heads, head_dim]
     */
    const float* get_v(uint32_t layer_idx) const {
        if (layer_idx >= num_layers_) return nullptr;
        return v_cache_[layer_idx].data();
    }
    
    /**
     * Get K cache как vector (для удобства)
     */
    const std::vector<float>& get_k_vec(uint32_t layer_idx) const {
        return k_cache_[layer_idx];
    }
    
    const std::vector<float>& get_v_vec(uint32_t layer_idx) const {
        return v_cache_[layer_idx];
    }
    
    /**
     * Текущая длина sequence в cache
     */
    uint32_t length() const { return current_length_; }
    
    /**
     * Clear весь cache (перед новой генерацией)
     */
    void clear() {
        for (auto& k : k_cache_) k.clear();
        for (auto& v : v_cache_) v.clear();
        current_length_ = 0;
        // Logging removed for speed
    }
    
    /**
     * Print cache statistics
     */
    void print_stats() const {
        printf("\n📊 KV-Cache Stats:\n");
        printf("   Current length: %u / %u tokens\n", current_length_, max_seq_len_);
        printf("   Num layers: %u\n", num_layers_);
        printf("   Memory used: %.2f MB\n",
               (current_length_ * num_heads_ * head_dim_ * sizeof(float) * 2 * num_layers_) 
               / (1024.0 * 1024.0));
        fflush(stdout);
    }

private:
    // Cache storage: [num_layers][seq_len * num_heads * head_dim]
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;
    
    // Configuration
    uint32_t num_layers_;
    uint32_t num_heads_;
    uint32_t head_dim_;
    uint32_t max_seq_len_;
    
    // Current state
    uint32_t current_length_;
};

} // namespace llm
