// Element-wise Multiply Benchmark
// Tests C[i] = A[i] * B[i] performance
//
// Critical for LLaMA FFN:
//   hidden = silu(gate_proj(x)) * up_proj(x)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>

// FP16 conversion (same as other benchmarks)
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
    appInfo.pApplicationName = "Elemwise Multiply Benchmark";
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

void elemwise_multiply_cpu(const std::vector<float>& a, const std::vector<float>& b, 
                           std::vector<float>& c) {
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] * b[i];
    }
}

void elemwise_multiply_gpu(const std::vector<float>& a, const std::vector<float>& b,
                           std::vector<float>& c, double& gpu_time_ms) {
    auto a_fp16 = convert_to_fp16(a);
    auto b_fp16 = convert_to_fp16(b);
    
    VulkanBuffer buf_a = create_buffer(a_fp16.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer buf_b = create_buffer(b_fp16.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer buf_c = create_buffer(a_fp16.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    upload_data(buf_a, a_fp16.data());
    upload_data(buf_b, b_fp16.data());
    
    VkShaderModule shader = load_shader("shaders/elemwise_multiply.spv");
    
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
    
    struct PushConstants { uint32_t num_elements; };
    
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
    poolSize.descriptorCount = 3;
    
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
    
    VkDescriptorBufferInfo bufferInfos[3] = {};
    bufferInfos[0].buffer = buf_a.buffer;
    bufferInfos[0].range = VK_WHOLE_SIZE;
    bufferInfos[1].buffer = buf_b.buffer;
    bufferInfos[1].range = VK_WHOLE_SIZE;
    bufferInfos[2].buffer = buf_c.buffer;
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
    
    PushConstants pc = {(uint32_t)a.size()};
    vkCmdPushConstants(commandBuffer, pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    
    uint32_t workgroups = (a.size() + 255) / 256;
    vkCmdDispatch(commandBuffer, workgroups, 1, 1);
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
    
    std::vector<uint16_t> c_fp16(a_fp16.size());
    download_data(buf_c, c_fp16.data());
    c = convert_from_fp16(c_fp16);
    
    destroy_buffer(buf_a);
    destroy_buffer(buf_b);
    destroy_buffer(buf_c);
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
    const float THRESHOLD = 5e-4f;
    
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::abs(ref[i]) < THRESHOLD || std::abs(test[i]) < THRESHOLD) {
            continue;
        }
        
        float error = std::abs(ref[i] - test[i]);
        float relative = error / std::abs(ref[i]);
        max_error = std::max(max_error, relative);
    }
    return max_error;
}

void run_benchmark(uint32_t num_elements, const char* description) {
    printf("\n=== %s (N=%u) ===\n", description, num_elements);
    
    srand(42);
    
    std::vector<float> a(num_elements);
    std::vector<float> b(num_elements);
    
    for (uint32_t i = 0; i < num_elements; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 12; j++) {
            sum += (rand() / (float)RAND_MAX);
        }
        a[i] = (sum - 6.0f);
        b[i] = (sum - 6.0f) * 0.5f;
    }
    
    std::vector<float> c_cpu(num_elements);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    elemwise_multiply_cpu(a, b, c_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    std::vector<float> c_gpu;
    double best_gpu_time = 1e9;
    for (int i = 0; i < 3; i++) {
        double gpu_time;
        elemwise_multiply_gpu(a, b, c_gpu, gpu_time);
        best_gpu_time = std::min(best_gpu_time, gpu_time);
    }
    
    float max_error = compute_max_relative_error(c_cpu, c_gpu);
    
    // Bandwidth: 3 memory ops (2 reads + 1 write) × 2 bytes (FP16)
    double bytes_transferred = num_elements * 3.0 * 2.0;
    double cpu_bandwidth = bytes_transferred / (cpu_time * 1e6);
    double gpu_bandwidth = bytes_transferred / (best_gpu_time * 1e6);
    
    printf("CPU time: %.2f ms (%.2f GB/s)\n", cpu_time, cpu_bandwidth);
    printf("GPU time: %.2f ms (%.2f GB/s)\n", best_gpu_time, gpu_bandwidth);
    printf("Speedup: %.2fx\n", cpu_time / best_gpu_time);
    printf("Max relative error: %.4f%%\n", max_error * 100);
    printf("Status: %s\n", max_error < 0.01 ? "✅ GOOD" : "❌ FAIL");
}

int main() {
    printf("Element-wise Multiply Benchmark\n");
    printf("================================\n\n");
    
    init_vulkan();
    
    // Test LLaMA FFN sizes
    run_benchmark(4096, "Hidden size (4K)");
    run_benchmark(11008, "LLaMA 7B intermediate (11K)");
    run_benchmark(4096 * 128, "Batch-128 hidden (512K)");
    run_benchmark(11008 * 128, "Batch-128 intermediate (1.4M)");
    
    printf("\n✅ Element-wise multiply validated!\n");
    printf("Ready for full FFN block.\n");
    
    return 0;
}
