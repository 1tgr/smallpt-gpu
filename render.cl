#include "render.h"

#ifdef __OPENCL_VERSION__

#define sqrtf sqrt

#else

#include <math.h>
#define __global
#define __kernel

#endif

static Vec vec(float x, float y, float z) {
    Vec vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    return vec;
}

static Ray ray(Vec o, Vec d) {
    Ray ray;
    ray.o = o;
    ray.d = d;
    return ray;
}

static float intersect_sphere(__global const Sphere *sphere, const Ray *r) { // returns distance, 0 if nohit
    Vec op = sphere->p - r->o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t,
        eps = 1e-4f,
        b = dot(op, r->d),
        det = b * b - dot(op, op) + sphere->rad * sphere->rad;
    if (det < 0) {
        return 0;
    }

    det = sqrtf(det);
    t = b - det;
    if (t > eps) {
        return t;
    }

    t = b + det;
    if (t > eps) {
        return t;
    }

    return 0;
}

static bool intersect_scene(__global const Sphere *scene, const int sceneSize, const Ray *r, float *t, int *id){
    float d, inf = *t = 1e20f;

    for(int i=sceneSize;i--;) {
        if ((d = intersect_sphere(&scene[i], r)) && d < *t) {
            *t = d;
            *id = i;
        }
    }

    return *t < inf;
}

static int rand_next(long long *seed, int bits) {
    *seed = (*seed * 0x5DEECE66DLL + 0xBLL) & ((1LL << 48) - 1);
    return (int)(*seed >> (48 - bits));
}

static float randf(long long *seed) {
    return rand_next(seed, 24) / ((float)(1 << 24));
}

__kernel void renderKernel(
    const int width,
    const int height,
    __global const Sphere *scene,
    const int sceneSize,
    __global float3 *output
) {
    Ray cam = ray(vec(50, 52, 295.6f), normalize(vec(0, -0.042612f, -1))); // cam pos, dir
    Vec cx = vec(width * .5135f / height, 0, 0), cy = normalize(cross(cx, cam.d)) * .5135f;
    int x = get_global_id(0);
    int y = get_global_id(1);
    int samp = get_global_id(2);
    long long seed = x * y * samp;
    float r1=2*randf(&seed), dx=r1<1 ? sqrtf(r1)-1: 1-sqrtf(2-r1);
    float r2=2*randf(&seed), dy=r2<1 ? sqrtf(r2)-1: 1-sqrtf(2-r2);
    Vec d
        = cx*(((dx + .5f) / 2 + x) / width - .5f)
        + cy*(((dy + .5f) / 2 + y) / height - .5f)
        + cam.d;
    //r = r + radiance(Ray(cam.o+d*140,d.norm()),0,Xi)*(1./samps);
    Ray r = ray(cam.o + d * 140.f, normalize(d));
    float t;
    int id;
    Vec color;
    if (intersect_scene(scene, sceneSize, &r, &t, &id)) {
        switch (id) {
            case 0:
                color = vec(1, 0, 0);
                break;

            case 1:
                color = vec(0, 1, 0);
                break;

            case 2:
                color = vec(0, 0, 1);
                break;

            case 3:
                color = vec(0.5f, 0, 0);
                break;

            case 4:
                color = vec(0, 0.5f, 0);
                break;

            case 5:
                color = vec(0, 0, 0.5f);
                break;

            case 6:
                color = vec(0.25f, 0, 0);
                break;

            case 7:
                color = vec(0, 0.25f, 0);
                break;

            case 8:
                color = vec(0, 0, 0.25f);
                break;

            default:
                color = vec(1, 1, 0);
                break;
        }
    } else {
        color = vec(0, 0, 0);
    }

    output[(height - y - 1) * width + x] += color;
}