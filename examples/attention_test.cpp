// Simple Attention Pipeline Test
// Tests: Q×K^T → scale(1/√d) → softmax → ×V
// Proof-of-concept that all attention components work together

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>

// FP16 conversion helpers
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
    appInfo.pApplicationName = "Attention Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    vkCreateInstance(&instanceInfo, nullptr, &instance);
    
    uint32_t deviceCount = 1;
    vkEnumeratePhysicalDevices(instance, &deviceCount, &physicalDevice);
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    printf("Using GPU: %s\n\n", props.deviceName);
    
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
    
    VkPhysicalDevice16BitStorageFeatures storageFeatures = {};
    storageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storageFeatures.storageBuffer16BitAccess = VK_TRUE;
    storageFeatures.pNext = &float16Features;
    
    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pNext = &storageFeatures;
    
    vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
    
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
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
    if (!f) {
        printf("Failed to open shader: %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> code(size);
    fread(code.data(), 1, size, f);
    fclose(f);
    
    VkShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode = (uint32_t*)code.data();
    
    VkShaderModule module;
    vkCreateShaderModule(device, &moduleInfo, nullptr, &module);
    return module;
}

// CPU reference: simple matmul for testing
void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, uint32_t M, uint32_t N, uint32_t K) {
    // C[M×N] = A[M×K] × B[K×N]
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Transpose matrix B for B^T multiplication
std::vector<float> transpose(const std::vector<float>& B, uint32_t rows, uint32_t cols) {
    std::vector<float> BT(rows * cols);
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            BT[j * rows + i] = B[i * cols + j];
        }
    }
    return BT;
}

// GPU softmax (reuse from softmax_benchmark)
void softmax_gpu(std::vector<float>& data, uint32_t batch_size, uint32_t seq_len) {
    auto data_fp16 = convert_to_fp16(data);
    
    VulkanBuffer buf = create_buffer(data_fp16.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    upload_data(buf, data_fp16.data());
    
    VkShaderModule shader = load_shader("shaders/softmax.spv");
    
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    
    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    
    struct PushConstants { uint32_t batch_size; uint32_t seq_len; };
    
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
    poolSize.descriptorCount = 1;
    
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
    
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = buf.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;
    
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    
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
    
    PushConstants pc = {batch_size, seq_len};
    vkCmdPushConstants(commandBuffer, pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    
    vkCmdDispatch(commandBuffer, batch_size, 1, 1);
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);
    
    std::vector<uint16_t> output_fp16(data_fp16.size());
    download_data(buf, output_fp16.data());
    data = convert_from_fp16(output_fp16);
    
    destroy_buffer(buf);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void test_attention() {
    printf("=== Simple Attention Pipeline Test ===\n");
    printf("Testing: Q×K^T → scale → softmax → ×V\n\n");
    
    // Simple single-head attention
    const uint32_t seq_len = 64;    // Sequence length
    const uint32_t head_dim = 128;  // Head dimension
    
    printf("Config: seq_len=%u, head_dim=%u\n\n", seq_len, head_dim);
    
    // Generate random Q, K, V
    srand(42);
    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    std::vector<float> V(seq_len * head_dim);
    
    for (auto& v : Q) v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (auto& v : K) v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (auto& v : V) v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    
    printf("Step 1: Q×K^T (scores)\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Q×K^T: [seq_len×head_dim] × [head_dim×seq_len] = [seq_len×seq_len]
    std::vector<float> K_T = transpose(K, seq_len, head_dim);
    std::vector<float> scores(seq_len * seq_len);
    matmul_cpu(Q, K_T, scores, seq_len, seq_len, head_dim);
    
    auto step1 = std::chrono::high_resolution_clock::now();
    printf("  Time: %.2f ms\n", std::chrono::duration<double, std::milli>(step1 - start).count());
    printf("  Output shape: [%u × %u]\n", seq_len, seq_len);
    printf("  Sample values: %.4f, %.4f, %.4f\n", scores[0], scores[1], scores[2]);
    
    printf("\nStep 2: Scale by 1/√head_dim\n");
    float scale = 1.0f / sqrtf((float)head_dim);
    for (auto& v : scores) v *= scale;
    
    auto step2 = std::chrono::high_resolution_clock::now();
    printf("  Scale factor: %.6f\n", scale);
    printf("  Time: %.2f ms\n", std::chrono::duration<double, std::milli>(step2 - step1).count());
    printf("  Scaled values: %.4f, %.4f, %.4f\n", scores[0], scores[1], scores[2]);
    
    printf("\nStep 3: Softmax (GPU)\n");
    softmax_gpu(scores, seq_len, seq_len);  // Each row is independent
    
    auto step3 = std::chrono::high_resolution_clock::now();
    printf("  Time: %.2f ms\n", std::chrono::duration<double, std::milli>(step3 - step2).count());
    printf("  Attention weights: %.4f, %.4f, %.4f\n", scores[0], scores[1], scores[2]);
    
    // Verify sum ≈ 1.0 for first row
    float sum = 0.0f;
    for (uint32_t i = 0; i < seq_len; i++) sum += scores[i];
    printf("  First row sum: %.6f (expected 1.0)\n", sum);
    
    printf("\nStep 4: Attention × V\n");
    // scores × V: [seq_len×seq_len] × [seq_len×head_dim] = [seq_len×head_dim]
    std::vector<float> output(seq_len * head_dim);
    matmul_cpu(scores, V, output, seq_len, head_dim, seq_len);
    
    auto step4 = std::chrono::high_resolution_clock::now();
    printf("  Time: %.2f ms\n", std::chrono::duration<double, std::milli>(step4 - step3).count());
    printf("  Output shape: [%u × %u]\n", seq_len, head_dim);
    printf("  Sample outputs: %.4f, %.4f, %.4f\n", output[0], output[1], output[2]);
    
    double total_time = std::chrono::duration<double, std::milli>(step4 - start).count();
    printf("\n=== Total Attention Time: %.2f ms ===\n", total_time);
    printf("✅ Attention pipeline works!\n");
    
    // Sanity check: output should be weighted combination of V
    printf("\nSanity check:\n");
    printf("  Input Q range: [%.3f, %.3f]\n", 
           *std::min_element(Q.begin(), Q.end()), 
           *std::max_element(Q.begin(), Q.end()));
    printf("  Input V range: [%.3f, %.3f]\n",
           *std::min_element(V.begin(), V.end()),
           *std::max_element(V.begin(), V.end()));
    printf("  Output range: [%.3f, %.3f]\n",
           *std::min_element(output.begin(), output.end()),
           *std::max_element(output.begin(), output.end()));
    printf("  Status: %s\n", 
           (std::abs(*std::max_element(output.begin(), output.end())) < 10.0f) ? "✅ GOOD" : "❌ FAIL");
}

int main() {
    printf("Simple Attention Pipeline Test\n");
    printf("==============================\n\n");
    
    init_vulkan();
    test_attention();
    
    printf("\n🎉 Ready for full LLaMA Attention implementation!\n");
    
    return 0;
}
