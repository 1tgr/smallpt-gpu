#include <fstream>
#include <iostream>
#include <sstream>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "OpenCL/cl2.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "render.h"

static const std::string kernel_source = {
#include "kernel_gen.h"
};

static cl_float3 vec(float x = 0.0, float y = 0.0, float z = 0.0) {
    cl_float3 vec = { .x = x, .y = y, .z = z};
    return vec;
}

static cl_float3 operator*(cl_float3 a, float b) {
    cl_float3 vec = { .x = a.x * b, .y = a.y * b, .z = a.z * b };
    return vec;
}

static Sphere scene[] = {//Scene: radius, position, emission, color, material
    Sphere(1e5, vec( 1e5+1,40.8,81.6), vec(),vec(.75,.25,.25),DIFF),//Left
    Sphere(1e5, vec(-1e5+99,40.8,81.6),vec(),vec(.25,.25,.75),DIFF),//Rght
    Sphere(1e5, vec(50,40.8, 1e5),     vec(),vec(.75,.75,.75),DIFF),//Back
    Sphere(1e5, vec(50,40.8,-1e5+170), vec(),vec(),           DIFF),//Frnt
    Sphere(1e5, vec(50, 1e5, 81.6),    vec(),vec(.75,.75,.75),DIFF),//Botm
    Sphere(1e5, vec(50,-1e5+81.6,81.6),vec(),vec(.75,.75,.75),DIFF),//Top
    Sphere(16.5,vec(27,16.5,47),       vec(),vec(1,1,1)*.999, SPEC),//Mirr
    Sphere(16.5,vec(73,16.5,78),       vec(),vec(1,1,1)*.999, REFR),//Glas
    Sphere(600, vec(50,681.6-.27,81.6),vec(12,12,12),  vec(), DIFF) //Lite
};

int main(void) {
    auto platforms = std::vector<cl::Platform>();
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cout << "No OpenCL platform found.";
        return -1;
    }

    auto platform = cl::Platform::setDefault(platforms[0]);
    auto device = cl::Device::getDefault();
    std::cout << "Using OpenCL device '" << device.getInfo<CL_DEVICE_NAME>()
              << "' on platform '" << platform.getInfo<CL_PLATFORM_NAME>() << "'" << std::endl;

    auto renderProgram = cl::Program(kernel_source);
    try {
        renderProgram.build();
    }
    catch (...) {
        // Print build info for all devices
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = renderProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        return 1;
    }

    const auto width = 640;
    const auto height = 480;
    const auto stride = width * 4;
    auto output = std::vector<cl_char4>(static_cast<size_t>(stride) * height);
    auto sceneBuffer = cl::Buffer(std::begin(scene), std::end(scene), true);
    auto outputBuffer = cl::Buffer(std::begin(output), std::end(output), false);
    auto renderKernel = cl::KernelFunctor<int, int, int, cl::Buffer, int, cl::Buffer>(renderProgram, "renderKernel");
    renderKernel(
        cl::EnqueueArgs(cl::NDRange(width, height)),
        width,
        height,
        stride,
        sceneBuffer,
        sizeof(scene) / sizeof(scene[0]),
        outputBuffer
    );

    cl::copy(outputBuffer, begin(output), end(output));
    stbi_write_png("image.png", width, height, 4, &output[0], stride);
    return 0;
}
