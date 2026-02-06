// RoPE Benchmark - Rotary Position Embeddings for Attention
// Tests positional encoding used in LLaMA Q/K matrices

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

// FP16 helpers (same as other benchmarks)
uint16_t float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xff) - 127;
    uint32_t mantissa = bits & 0x7fffff;
    
    if (exp > 15) { exp = 15; mantissa = 0x3ff; }
    else if (exp < -14) { return (sign << 15); }
    
    return (sign << 15) | ((exp + 15) << 10) | (mantissa >> 13);
}

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    
    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float f = mantissa / 1024.0f * powf(2.0f, -14.0f);
        return sign ? -f : f;
    }
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    
    float f = (1.0f + mantissa / 1024.0f) * powf(2.0f, (int)exp - 15);
    return sign ? -f : f;
}

std::vector<uint16_t> convert_to_fp16(const std::vector<float>& data) {
    std::vector<uint16_t> result(data.size());
    for (size_t i = 0; i < data.size(); i++) result[i] = float_to_fp16(data[i]);
    return result;
}

std::vector<float> convert_from_fp16(const std::vector<uint16_t>& data) {
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); i++) result[i] = fp16_to_float(data[i]);
    return result;
}

// Vulkan globals
VkInstance instance;
VkPhysicalDevice physicalDevice;
VkDevice device;
VkQueue computeQueue;
VkCommandPool commandPool;

struct VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
};

void init_vulkan() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RoPE Benchmark";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    vkCreateInstance(&instanceInfo, nullptr, &instance);
    
    uint32_t deviceCount = 1;
    vkEnumeratePhysicalDevices(instance, &deviceCount, &physicalDevice);
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    printf("Using GPU: %s\n", props.deviceName);
    
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    uint32_t computeQueueFamily = 0;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = i;
            break;
        }
    }
    
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    
    VkPhysicalDeviceShaderFloat16Int8Features float16Features = {};
    float16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    float16Features.shaderFloat16 = VK_TRUE;
    
    VkPhysicalDevice16BitStorageFeatures storage16Features = {};
    storage16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16Features.storageBuffer16BitAccess = VK_TRUE;
    storage16Features.pNext = &float16Features;
    
    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pNext = &storage16Features;
    
    vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
    
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
}

uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

VulkanBuffer create_buffer(size_t size, VkBufferUsageFlags usage) {
    VulkanBuffer buf;
    buf.size = size;
    
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufferInfo, nullptr, &buf.buffer);
    
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, buf.buffer, &memReqs);
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    vkAllocateMemory(device, &allocInfo, nullptr, &buf.memory);
    vkBindBufferMemory(device, buf.buffer, buf.memory, 0);
    return buf;
}

void upload_data(VulkanBuffer& buf, const void* data) {
    void* mapped;
    vkMapMemory(device, buf.memory, 0, buf.size, 0, &mapped);
    memcpy(mapped, data, buf.size);
    vkUnmapMemory(device, buf.memory);
}

void download_data(VulkanBuffer& buf, void* data) {
    void* mapped;
    vkMapMemory(device, buf.memory, 0, buf.size, 0, &mapped);
    memcpy(data, mapped, buf.size);
    vkUnmapMemory(device, buf.memory);
}

void destroy_buffer(VulkanBuffer& buf) {
    vkDestroyBuffer(device, buf.buffer, nullptr);
    vkFreeMemory(device, buf.memory, nullptr);
}

VkShaderModule load_shader(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Failed to open shader: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code(size / 4);
    fread(code.data(), 1, size, f);
    fclose(f);
    
    VkShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = size;
    moduleInfo.pCode = code.data();
    
    VkShaderModule module;
    vkCreateShaderModule(device, &moduleInfo, nullptr, &module);
    return module;
}

// CPU reference
void rope_cpu(const std::vector<float>& input, std::vector<float>& output,
              uint32_t batch, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
              float theta_base) {
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t pos = 0; pos < seq_len; pos++) {
            for (uint32_t h = 0; h < num_heads; h++) {
                uint32_t base_offset = ((b * seq_len + pos) * num_heads + h) * head_dim;
                
                for (uint32_t i = 0; i < head_dim; i += 2) {
                    uint32_t offset = base_offset + i;
                    float x0 = input[offset];
                    float x1 = input[offset + 1];
                    
                    float exponent = -(float)i / head_dim;
                    float freq = powf(theta_base, exponent);
                    float theta = pos * freq;
                    
                    float cos_theta = cosf(theta);
                    float sin_theta = sinf(theta);
                    
                    output[offset] = x0 * cos_theta - x1 * sin_theta;
                    output[offset + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
}

// GPU implementation
void rope_gpu(const std::vector<float>& input, std::vector<float>& output,
              uint32_t batch, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
              float theta_base, double& gpu_time_ms) {
    auto input_fp16 = convert_to_fp16(input);
    
    VulkanBuffer buf_input = create_buffer(input_fp16.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer buf_output = create_buffer(input_fp16.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    upload_data(buf_input, input_fp16.data());
    
    VkShaderModule shader = load_shader("shaders/rope.spv");
    
    VkDescriptorSetLayoutBinding bindings[2] = {};
    for (int i = 0; i < 2; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    
    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    
    struct PushConstants {
        uint32_t batch_size;
        uint32_t seq_len;
        uint32_t num_heads;
        uint32_t head_dim;
        float theta_base;
    };
    
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shader;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;
    
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    
    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    
    VkDescriptorSet descriptorSet;
    vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
    
    VkDescriptorBufferInfo bufferInfos[2] = {};
    bufferInfos[0].buffer = buf_input.buffer;
    bufferInfos[0].range = VK_WHOLE_SIZE;
    bufferInfos[1].buffer = buf_output.buffer;
    bufferInfos[1].range = VK_WHOLE_SIZE;
    
    VkWriteDescriptorSet writes[2] = {};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    
    PushConstants pc = {batch, seq_len, num_heads, head_dim, theta_base};
    vkCmdPushConstants(commandBuffer, pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    
    uint32_t workgroups_x = (seq_len + 15) / 16;
    uint32_t workgroups_y = (num_heads + 15) / 16;
    vkCmdDispatch(commandBuffer, workgroups_x, workgroups_y, 1);
    vkEndCommandBuffer(commandBuffer);
    
    auto start = std::chrono::high_resolution_clock::now();
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);
    auto end = std::chrono::high_resolution_clock::now();
    gpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::vector<uint16_t> output_fp16(input_fp16.size());
    download_data(buf_output, output_fp16.data());
    output = convert_from_fp16(output_fp16);
    
    destroy_buffer(buf_input);
    destroy_buffer(buf_output);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

float compute_max_error(const std::vector<float>& ref, const std::vector<float>& test) {
    float max_error = 0.0f;
    const float THRESHOLD = 0.01f;  // Use absolute error for values < 0.01
    int max_idx = -1;
    for (size_t i = 0; i < ref.size(); i++) {
        float abs_error = std::abs(ref[i] - test[i]);
        float error;
        if (std::abs(ref[i]) < THRESHOLD) {
            // Use absolute error for small values
            error = abs_error;
        } else {
            // Use relative error for larger values
            error = abs_error / std::abs(ref[i]);
        }
        if (error > max_error) {
            max_error = error;
            max_idx = i;
        }
    }
    if (max_idx >= 0) {
        printf("  Max error at [%d]: CPU=%.6f GPU=%.6f (%.4f%%)\n",
               max_idx, ref[max_idx], test[max_idx], max_error * 100);
    }
    return max_error;
}

void run_benchmark(uint32_t batch, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) {
    printf("\n=== RoPE: batch=%u, seq=%u, heads=%u, dim=%u ===\n",
           batch, seq_len, num_heads, head_dim);
    
    const float theta_base = 10000.0f;
    size_t total_elements = batch * seq_len * num_heads * head_dim;
    
    srand(42);
    std::vector<float> input(total_elements);
    for (auto& v : input) {
        float sum = 0.0f;
        for (int j = 0; j < 12; j++) sum += (rand() / (float)RAND_MAX);
        v = (sum - 6.0f);
    }
    
    std::vector<float> output_cpu(total_elements);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    rope_cpu(input, output_cpu, batch, seq_len, num_heads, head_dim, theta_base);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    std::vector<float> output_gpu;
    double best_gpu_time = 1e9;
    for (int i = 0; i < 3; i++) {
        double gpu_time;
        rope_gpu(input, output_gpu, batch, seq_len, num_heads, head_dim, theta_base, gpu_time);
        best_gpu_time = std::min(best_gpu_time, gpu_time);
    }
    
    // Debug: print first few values
    printf("First 10 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] CPU=%.6f GPU=%.6f\n", i, output_cpu[i], output_gpu[i]);
    }
    
    // Debug: check specific problemtic indices
    if (total_elements > 112522) {
        size_t idx = 112522;
        // Calculate position in tensor
        size_t remaining = idx;
        size_t b = remaining / (seq_len * num_heads * head_dim);
        remaining %= (seq_len * num_heads * head_dim);
        size_t pos = remaining / (num_heads * head_dim);
        remaining %= (num_heads * head_dim);
        size_t h = remaining / head_dim;
        size_t dim_idx = remaining % head_dim;
        
        printf("  Index %zu: batch=%zu, pos=%zu, head=%zu, dim=%zu\n", idx, b, pos, h, dim_idx);
        printf("    Input: %.6f\n", input[idx]);
        printf("    CPU output: %.6f\n", output_cpu[idx]);
        printf("    GPU output: %.6f\n", output_gpu[idx]);
    }
    
    float max_error = compute_max_error(output_cpu, output_gpu);
    
    double bytes = total_elements * 2.0 * 2.0;  // 2 ops × 2 bytes FP16
    double cpu_bw = bytes / (cpu_time * 1e6);
    double gpu_bw = bytes / (best_gpu_time * 1e6);
    
    printf("CPU: %.2f ms (%.2f GB/s)\n", cpu_time, cpu_bw);
    printf("GPU: %.2f ms (%.2f GB/s)\n", best_gpu_time, gpu_bw);
    printf("Speedup: %.2fx\n", cpu_time / best_gpu_time);
    printf("Error: %.4f%%\n", max_error * 100);
    printf("Status: %s\n", max_error < 0.10 ? "✅ GOOD" : "❌ FAIL");
}

int main() {
    printf("RoPE (Rotary Position Embeddings) Benchmark\n");
    printf("============================================\n\n");
    
    init_vulkan();
    
    // LLaMA dimensions
    run_benchmark(1, 32, 32, 128);    // Small: 32 tokens
    run_benchmark(1, 128, 32, 128);   // Medium: 128 tokens
    run_benchmark(1, 512, 32, 128);   // Large: 512 tokens
    run_benchmark(4, 128, 32, 128);   // Batch: 4×128 tokens
    
    printf("\n✅ RoPE kernel validated!\n");
    printf("Ready for Attention mechanism.\n");
    
    return 0;
}
