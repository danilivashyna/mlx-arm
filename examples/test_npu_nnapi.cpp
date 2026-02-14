
#include <iostream>
#include <vector>
#include <dlfcn.h>
#include <android/NeuralNetworks.h>

typedef int (*ANeuralNetworks_getDeviceCount_ptr)(uint32_t* numDevices);
typedef int (*ANeuralNetworks_getDevice_ptr)(uint32_t devIndex, ANeuralNetworksDevice** device);
typedef int (*ANeuralNetworksDevice_getName_ptr)(const ANeuralNetworksDevice* device, const char** name);
typedef int (*ANeuralNetworksDevice_getType_ptr)(const ANeuralNetworksDevice* device, int32_t* type);

int main() {
    void* handle = dlopen("libneuralnetworks.so", RTLD_NOW);
    if (!handle) {
        std::cerr << "Failed to load libneuralnetworks.so" << std::endl;
        return 1;
    }

    auto getDeviceCount = (ANeuralNetworks_getDeviceCount_ptr)dlsym(handle, "ANeuralNetworks_getDeviceCount");
    auto getDevice = (ANeuralNetworks_getDevice_ptr)dlsym(handle, "ANeuralNetworks_getDevice");
    auto getName = (ANeuralNetworksDevice_getName_ptr)dlsym(handle, "ANeuralNetworksDevice_getName");
    auto getType = (ANeuralNetworksDevice_getType_ptr)dlsym(handle, "ANeuralNetworksDevice_getType");

    if (!getDeviceCount || !getDevice || !getName || !getType) {
        std::cerr << "Failed to find NNAPI symbols" << std::endl;
        return 1;
    }

    uint32_t numDevices = 0;
    getDeviceCount(&numDevices);
    std::cout << "Found " << numDevices << " NNAPI devices:" << std::endl;

    for (uint32_t i = 0; i < numDevices; ++i) {
        ANeuralNetworksDevice* device = nullptr;
        getDevice(i, &device);
        const char* name = nullptr;
        getName(device, &name);
        int32_t type = 0;
        getType(device, &type);

        std::cout << "Device [" << i << "]: " << name << " (Type: " << type << ")" << std::endl;
    }

    return 0;
}
