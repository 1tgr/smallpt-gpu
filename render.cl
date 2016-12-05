__kernel void square(__global float *input, __global char4 *output, const unsigned int count) {
    int i = get_global_id(0);
    if (i >= count)
        return;

    float n = input[i] * input[i];
    char b = (char) (n * 255);
    output[i].x = b;
    output[i].y = b;
    output[i].z = b;
    output[i].w = 255;
}