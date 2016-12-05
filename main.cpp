#include <fstream>
#include <iostream>
#include <sstream>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "OpenCL/cl2.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

std::string get_file_contents(const char *filename) {
    std::ifstream in(filename, std::ios::in);
    if (!in)
        throw errno;

    std::string contents;
    in.seekg(0, std::ios::end);
    contents.reserve(in.tellg());
    in.seekg(0, std::ios::beg);
    contents.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    return contents;
}

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

    auto source = get_file_contents("/Users/Tim/Git/render-gpu/render.cl");
    auto squareProgram = cl::Program(source);
    try {
        squareProgram.build();
    }
    catch (...) {
        // Print build info for all devices
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = squareProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        return 1;
    }

    const auto width = 640;
    const auto height = 480;
    auto input = std::vector<float>(width * height);
    for (auto& f : input) {
        f = rand() / (float) RAND_MAX;
    }

    auto output = std::vector<cl_char4>(input.size());
    auto inputBuffer = cl::Buffer(begin(input), end(input), true);
    auto outputBuffer = cl::Buffer(begin(output), end(output), false);
    auto squareKernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, size_t>(squareProgram, "square");
    cl_int error;
    squareKernel(
        cl::EnqueueArgs(cl::NDRange(input.size())),
        inputBuffer,
        outputBuffer,
        input.size()
    );

    cl::copy(outputBuffer, begin(output), end(output));
    stbi_write_png("image.png", width, height, 4, &output[0], width * 4);
    return 0;
}
