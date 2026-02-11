#include "vulkan_pipeline.h"
#include "vulkan_device.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

// Cross-platform logging macros
#ifdef __ANDROID__
    #include <android/log.h>
    #define LOGD(fmt, ...) __android_log_print(ANDROID_LOG_DEBUG, "MLX-Pipeline", fmt, ##__VA_ARGS__)
    #define LOGI(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "MLX-Pipeline", fmt, ##__VA_ARGS__)
    #define LOGE(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, "MLX-Pipeline", fmt, ##__VA_ARGS__)
#else
    #include <cstdio>
    #define LOGD(fmt, ...) do { printf("DEBUG: " fmt "\n", ##__VA_ARGS__); } while(0)
    #define LOGI(fmt, ...) do { printf("INFO: " fmt "\n", ##__VA_ARGS__); } while(0)
    #define LOGE(fmt, ...) do { fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__); } while(0)
#endif

namespace mlx {
namespace backend {
namespace vulkan {

VulkanPipeline::VulkanPipeline(
    const VulkanDevice& device,
    const std::string& spirvPath,
    uint32_t pushConstantSize
) : device_(&device), push_constant_size_(pushConstantSize) {
    
    // Load SPIR-V
    auto spirv = loadSPIRV(spirvPath);
    LOGI("Loaded SPIR-V: %s (%zu bytes)", spirvPath.c_str(), spirv.size() * 4);
    
    // Create pipeline components
    createShaderModule(spirv);
    createDescriptorSetLayout();
    createDescriptorPool();
    createPipelineLayout();
    createPipeline();
    
    LOGI("Created Vulkan compute pipeline");
}

VulkanPipeline::~VulkanPipeline() {
    cleanup();
}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept
    : device_(other.device_),
      pipeline_(other.pipeline_),
      layout_(other.layout_),
      descriptor_set_layout_(other.descriptor_set_layout_),
      descriptor_pool_(other.descriptor_pool_),
      shader_module_(other.shader_module_),
      push_constant_size_(other.push_constant_size_) {
    other.pipeline_ = VK_NULL_HANDLE;
    other.layout_ = VK_NULL_HANDLE;
    other.descriptor_set_layout_ = VK_NULL_HANDLE;
    other.descriptor_pool_ = VK_NULL_HANDLE;
    other.shader_module_ = VK_NULL_HANDLE;
}

VulkanPipeline& VulkanPipeline::operator=(VulkanPipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        device_ = other.device_;
        pipeline_ = other.pipeline_;
        layout_ = other.layout_;
        descriptor_set_layout_ = other.descriptor_set_layout_;
        descriptor_pool_ = other.descriptor_pool_;
        shader_module_ = other.shader_module_;
        push_constant_size_ = other.push_constant_size_;
        
        other.pipeline_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
        other.descriptor_set_layout_ = VK_NULL_HANDLE;
        other.descriptor_pool_ = VK_NULL_HANDLE;
        other.shader_module_ = VK_NULL_HANDLE;
    }
    return *this;
}

VkDescriptorSet VulkanPipeline::createDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptor_pool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptor_set_layout_;
    
    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(device_->device(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }
    
    return descriptorSet;
}

void VulkanPipeline::updateDescriptorSet(
    VkDescriptorSet descriptorSet,
    VkBuffer* buffers,
    size_t bufferCount
) {
    std::vector<VkDescriptorBufferInfo> bufferInfos(bufferCount);
    std::vector<VkWriteDescriptorSet> descriptorWrites(bufferCount);
    
    for (size_t i = 0; i < bufferCount; i++) {
        bufferInfos[i].buffer = buffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
        
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    
    vkUpdateDescriptorSets(device_->device(), 
                          static_cast<uint32_t>(descriptorWrites.size()), 
                          descriptorWrites.data(), 
                          0, nullptr);
}

std::vector<uint32_t> VulkanPipeline::loadSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    
    return buffer;
}

void VulkanPipeline::createShaderModule(const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();
    
    if (vkCreateShaderModule(device_->device(), &createInfo, nullptr, &shader_module_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
}

void VulkanPipeline::createDescriptorSetLayout() {
    // 3 storage buffers: input A, input B, output result
    VkDescriptorSetLayoutBinding bindings[3] = {};
    
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    
    if (vkCreateDescriptorSetLayout(device_->device(), &layoutInfo, nullptr, 
                                   &descriptor_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void VulkanPipeline::createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 30;  // Support 10 descriptor sets with 3 buffers each
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 10;
    
    if (vkCreateDescriptorPool(device_->device(), &poolInfo, nullptr, 
                              &descriptor_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void VulkanPipeline::createPipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_set_layout_;
    
    // Add push constants if needed
    VkPushConstantRange pushConstantRange{};
    if (push_constant_size_ > 0) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = push_constant_size_;
        
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    }
    
    if (vkCreatePipelineLayout(device_->device(), &pipelineLayoutInfo, nullptr, 
                              &layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
}

void VulkanPipeline::createPipeline() {
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shader_module_;
    shaderStageInfo.pName = "main";
    
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = layout_;
    
    if (vkCreateComputePipelines(device_->device(), VK_NULL_HANDLE, 1, 
                                 &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
}

void VulkanPipeline::cleanup() {
    if (device_) {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_->device(), pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        
        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_->device(), layout_, nullptr);
            layout_ = VK_NULL_HANDLE;
        }
        
        if (descriptor_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_->device(), descriptor_pool_, nullptr);
            descriptor_pool_ = VK_NULL_HANDLE;
        }
        
        if (descriptor_set_layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_->device(), descriptor_set_layout_, nullptr);
            descriptor_set_layout_ = VK_NULL_HANDLE;
        }
        
        if (shader_module_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_->device(), shader_module_, nullptr);
            shader_module_ = VK_NULL_HANDLE;
        }
    }
}

} // namespace vulkan
} // namespace backend
} // namespace mlx
