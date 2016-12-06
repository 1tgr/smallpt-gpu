#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "OpenCL/cl2.hpp"
#define USE_GPU

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "render.h"

#ifdef USE_GPU

static const std::string kernel_source = {
#include "kernel_gen.h"
};

#else

#include "render.cl"

#endif

static num clamp(num x) {
    return x<0 ? 0 : x>1 ? 1 : x;
}

static unsigned char to_byte(num x) {
    return static_cast<unsigned char>(pow(clamp(x), 1.0f / 2.2f) * 255.f + .5f);
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
    const auto width = 640;
    const auto height = 480;
    const auto samples = 100;
    auto output = std::vector<Vec>(width * height);

#ifdef USE_GPU
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
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = renderProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        return 1;
    }

    auto sceneBuffer = cl::Buffer(std::begin(scene), std::end(scene), true);
    auto outputBuffer = cl::Buffer(std::begin(output), std::end(output), false);
    auto renderKernel = cl::KernelFunctor<int, int, int, cl::Buffer, int, cl::Buffer>(renderProgram, "renderKernel");
    renderKernel(
        cl::EnqueueArgs(cl::NDRange(width, height, samples)),
        width,
        height,
        samples,
        sceneBuffer,
        sizeof(scene) / sizeof(scene[0]),
        outputBuffer
    );

    cl::copy(outputBuffer, begin(output), end(output));
#else
    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            for (auto samp = 0; samp < samples; samp++) {
                renderKernel(width, height, samples, scene, sizeof(scene) / sizeof(scene[0]), &output[0], x, y, samp);
            }
        }
    }
#endif

    auto data = std::vector<unsigned char>();
    data.reserve(width * height * 3);
    for (auto color : output) {
        data.push_back(to_byte(color.x));
        data.push_back(to_byte(color.y));
        data.push_back(to_byte(color.z));
    }

    stbi_write_png("image.png", width, height, 3, &data[0], width * 3);
    return 0;
}
