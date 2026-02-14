
#include <iostream>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    std::cout << "=== Hexagon FastRPC Discovery ===" << std::endl;

    // Try to load the compute DSP RPC library locally first
    const char* lib_path = "./libcdsprpc.so";
    void* handle = dlopen(lib_path, RTLD_NOW);

    if (!handle) {
        std::cout << "ℹ️  Local libcdsprpc.so not found, trying system path..." << std::endl;
        lib_path = "/vendor/lib64/libcdsprpc.so";
        handle = dlopen(lib_path, RTLD_NOW);
    }

    if (handle) {
        std::cout << "✅ Successfully loaded libcdsprpc.so!" << std::endl;
        
        // Try to find the remote handle open function
        void* open_func = dlsym(handle, "remote_handle_open");
        if (open_func) {
            std::cout << "✅ Found remote_handle_open symbol. FastRPC is accessible!" << std::endl;
        } else {
            std::cout << "⚠️  Could not find remote_handle_open. Library might be restricted." << std::endl;
        }
        
        dlclose(handle);
    } else {
        std::cout << "❌ Hexagon cDSP is not accessible through standard RPC library." << std::endl;
    }

    // Also check for the device node
    int fd = open("/dev/cdsp-passthrough", O_RDONLY);
    if (fd >= 0) {
        std::cout << "✅ Device node /dev/cdsp-passthrough is accessible!" << std::endl;
        close(fd);
    } else {
        std::cout << "ℹ️  Device node /dev/cdsp-passthrough not found (standard for user apps)." << std::endl;
    }

    return 0;
}
