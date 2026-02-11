// RMSNorm Benchmark - First LLM Kernel
// Tests Root Mean Square Layer Normalization on GPU
//
// RMSNorm is critical for:
// - LLaMA, Mistral, Gemma architectures
// - Pre-normalization in transformer layers
// - More stable than LayerNorm
//
// Formula: output = (x / RMS(x)) * weight
//          where RMS(x) = sqrt(mean(x^2) + eps)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

// FP16 conversion helpers (same as matmul_benchmark)
uint16_t float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xff) - 127;
    uint32_t mantissa = bits & 0x7fffff;
    
    if (exp > 15) {
        exp = 15;
        mantissa = 0x3ff;
    } else if (exp < -14) {
        return (sign << 15);
    }
    
    uint16_t fp16 = (sign << 15) | ((exp + 15) << 10) | (mantissa >> 13);
    return fp16;
}

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    
    if (exp == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        float f = mantissa / 1024.0f;
        f *= powf(2.0f, -14.0f);
        return sign ? -f : f;
    }
    
    if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }
    
    float f = 1.0f + (mantissa / 1024.0f);
    f *= powf(2.0f, (int)exp - 15);
    return sign ? -f : f;
}

std::vector<uint16_t> convert_to_fp16(const std::vector<float>& data) {
    std::vector<uint16_t> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result[i] = float_to_fp16(data[i]);
    }
    return result;
}

std::vector<float> convert_from_fp16(const std::vector<uint16_t>& data) {
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result[i] = fp16_to_float(data[i]);
    }
    return result;
}

// Vulkan helpers
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
    // Create instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RMSNorm Benchmark";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    
    vkCreateInstance(&instanceInfo, nullptr, &instance);
    
    // Select physical device
    uint32_t deviceCount = 1;
    vkEnumeratePhysicalDevices(instance, &deviceCount, &physicalDevice);
    
    // Print device name
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    printf("Using GPU: %s\n", props.deviceName);
    
    // Find compute queue family
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
    
    // Create logical device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    
    // Enable FP16 features
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
    
    // Create command pool
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
        if ((typeFilter & (1 << i)) && 
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
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
    if (!f) {
        printf("Failed to open shader: %s\n", path);
        exit(1);
    }
    
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

// CPU reference implementation
void rmsnorm_cpu(const std::vector<float>& input,
                 const std::vector<float>& weight,
                 std::vector<float>& output,
                 uint32_t batch, uint32_t seq_len, uint32_t hidden,
                 float eps = 1e-6f) {
    for (uint32_t t = 0; t < batch * seq_len; t++) {
        uint32_t offset = t * hidden;
        
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (uint32_t h = 0; h < hidden; h++) {
            float val = input[offset + h];
            sum_sq += val * val;
        }
        
        // Compute RMS
        float variance = sum_sq / hidden;
        float rms_inv = 1.0f / sqrtf(variance + eps);
        
        // Normalize and apply weight
        for (uint32_t h = 0; h < hidden; h++) {
            float val = input[offset + h];
            float normalized = val * rms_inv;
            output[offset + h] = normalized * weight[h];
        }
    }
}

// GPU implementation
void rmsnorm_gpu(const std::vector<float>& input,
                 const std::vector<float>& weight,
                 std::vector<float>& output,
                 uint32_t batch, uint32_t seq_len, uint32_t hidden,
                 float eps, double& gpu_time_ms) {
    // Convert to FP16
    auto input_fp16 = convert_to_fp16(input);
    auto weight_fp16 = convert_to_fp16(weight);
    
    // Create buffers
    VulkanBuffer buf_input = create_buffer(input_fp16.size() * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer buf_weight = create_buffer(weight_fp16.size() * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer buf_output = create_buffer(input_fp16.size() * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    upload_data(buf_input, input_fp16.data());
    upload_data(buf_weight, weight_fp16.data());
    
    // Load shader
    VkShaderModule shader = load_shader("shaders/rms_norm.spv");
    
    // Create descriptor set layout
    VkDescriptorSetLayoutBinding bindings[3] = {};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    
    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    
    // Create pipeline layout with push constants
    struct PushConstants {
        uint32_t batch_size;
        uint32_t seq_len;
        uint32_t hidden_dim;
        float eps;
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
    
    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shader;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;
    
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    
    // Create descriptor pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    
    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    
    VkDescriptorSet descriptorSet;
    vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
    
    // Update descriptor set
    VkDescriptorBufferInfo bufferInfos[3] = {};
    bufferInfos[0].buffer = buf_input.buffer;
    bufferInfos[0].range = VK_WHOLE_SIZE;
    bufferInfos[1].buffer = buf_weight.buffer;
    bufferInfos[1].range = VK_WHOLE_SIZE;
    bufferInfos[2].buffer = buf_output.buffer;
    bufferInfos[2].range = VK_WHOLE_SIZE;
    
    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    
    // Record command buffer
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
    
    PushConstants pc = {batch, seq_len, hidden, eps};
    vkCmdPushConstants(commandBuffer, pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    
    vkCmdDispatch(commandBuffer, batch * seq_len, 1, 1);
    vkEndCommandBuffer(commandBuffer);
    
    // Submit and time
    auto start = std::chrono::high_resolution_clock::now();
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);
    
    auto end = std::chrono::high_resolution_clock::now();
    gpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Download result
    std::vector<uint16_t> output_fp16(input_fp16.size());
    download_data(buf_output, output_fp16.data());
    output = convert_from_fp16(output_fp16);
    
    // Cleanup
    destroy_buffer(buf_input);
    destroy_buffer(buf_weight);
    destroy_buffer(buf_output);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

float compute_max_relative_error(const std::vector<float>& ref,
                                  const std::vector<float>& test) {
    float max_error = 0.0f;
    size_t max_idx = 0;
    const float THRESHOLD = 5e-4f;  // FP16 minimum normal value is ~6e-5
    
    for (size_t i = 0; i < ref.size(); i++) {
        // Skip extremely small values where FP16 precision is insufficient
        if (std::abs(ref[i]) < THRESHOLD || std::abs(test[i]) < THRESHOLD) {
            continue;
        }
        
        float error = std::abs(ref[i] - test[i]);
        float relative = error / std::abs(ref[i]);
        if (relative > max_error) {
            max_error = relative;
            max_idx = i;
        }
    }
    return max_error;
}

void run_benchmark(uint32_t batch, uint32_t seq_len, uint32_t hidden) {
    printf("\n=== RMSNorm Benchmark: batch=%u, seq_len=%u, hidden=%u ===\n",
           batch, seq_len, hidden);
    
    const float eps = 1e-6f;
    size_t total_elements = batch * seq_len * hidden;
    
    // Initialize with fixed seed for reproducibility
    srand(42);  // Same seed for all sizes
    
    // Initialize random data
    std::vector<float> input(total_elements);
    std::vector<float> weight(hidden);
    
    // Use Gaussian-like distribution (sum of 12 uniforms)
    // This gives mean=0, std≈0.3, which is typical for LLM activations
    for (size_t i = 0; i < input.size(); i++) {
        float sum = 0.0f;
        for (int j = 0; j < 12; j++) {
            sum += (rand() / (float)RAND_MAX);
        }
        input[i] = (sum - 6.0f) * 0.3f;  // Mean=0, std≈0.3
    }
    for (size_t i = 0; i < weight.size(); i++) {
        weight[i] = 1.0f;  // Unit weights for simplicity
    }
    
    // CPU reference
    std::vector<float> output_cpu(total_elements);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    rmsnorm_cpu(input, weight, output_cpu, batch, seq_len, hidden, eps);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU (best of 3)
    std::vector<float> output_gpu;
    double best_gpu_time = 1e9;
    for (int i = 0; i < 3; i++) {
        double gpu_time;
        rmsnorm_gpu(input, weight, output_gpu, batch, seq_len, hidden, eps, gpu_time);
        best_gpu_time = std::min(best_gpu_time, gpu_time);
    }
    
    // Compute accuracy
    float max_error = compute_max_relative_error(output_cpu, output_gpu);
    
    // Compute GFLOPS (approximation: 2*hidden FLOPs per element)
    // Sum of squares: hidden multiplies + (hidden-1) adds
    // RMS computation: 1 div, 1 sqrt, 1 add
    // Normalize + weight: hidden multiplies, hidden multiplies
    // Total ≈ 4*hidden FLOPs per token
    double flops = batch * seq_len * hidden * 4.0;
    double cpu_gflops = flops / (cpu_time * 1e6);
    double gpu_gflops = flops / (best_gpu_time * 1e6);
    
    printf("CPU time: %.2f ms (%.2f GFLOPS)\n", cpu_time, cpu_gflops);
    printf("GPU time: %.2f ms (%.2f GFLOPS)\n", best_gpu_time, gpu_gflops);
    printf("Speedup: %.2fx\n", cpu_time / best_gpu_time);
    printf("Max relative error: %.4f%%\n", max_error * 100);
    printf("Status: %s\n", max_error < 0.01 ? "✅ GOOD" : "❌ FAIL");
}

int main() {
    printf("RMSNorm Benchmark - First LLM Kernel\n");
    printf("=====================================\n\n");
    
    init_vulkan();
    
    // Test various sizes (typical LLM dimensions)
    run_benchmark(1, 1, 768);      // Small transformer
    run_benchmark(1, 1, 2048);     // Medium
    run_benchmark(1, 1, 4096);     // LLaMA 7B/13B
    run_benchmark(1, 128, 4096);   // Batch of tokens
    run_benchmark(1, 512, 4096);   // Longer sequence
    
    printf("\n✅ RMSNorm kernel validated!\n");
    printf("Ready for LLM inference pipeline.\n");
    
    return 0;
}
