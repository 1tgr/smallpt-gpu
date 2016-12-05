#include "render.h"

#ifndef __OPENCL_VERSION__
#include <math.h>
#include <cl_kernel.h>
#define __global
#define __kernel
typedef Vec char4;
#endif

static Vec vec(float x, float y, float z) {
    Vec vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    return vec;
}

static char4 color(char r, char g, char b) {
    char4 color;
    color.x = r;
    color.y = g;
    color.z = b;
    color.w = 255;
    return color;
}

static Ray ray(Vec o, Vec d) {
    Ray ray;
    ray.o = o;
    ray.d = d;
    return ray;
}

static int toInt(float x) {
    return int(pow(clamp(x, 0.0f, 1.0f), 1.0f / 2.2f) * 255.f + .5f);
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

    det = sqrt(det);
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

__kernel void renderKernel(
    const int width,
    const int height,
    const int stride,
    __global const Sphere *scene,
    const int sceneSize,
    __global char4 *output
) {
    Ray cam = ray(vec(50, 52, 295.6f), normalize(vec(0, -0.042612f, -1))); // cam pos, dir
    Vec cx = vec(width * .5135f / height, 0, 0), cy = normalize(cross(cx, cam.d)) * .5135f;
    int x = get_global_id(0);
    int y = get_global_id(1);
    float sx = 0, sy = 0, dx = 0, dy = 0;
    Vec d
        = cx*( ( (sx+.5f + dx)/2 + x)/width - .5f)
        + cy*( ( (sy+.5f + dy)/2 + y)/height - .5f)
        + cam.d;
    //r = r + radiance(Ray(cam.o+d*140,d.norm()),0,Xi)*(1./samps);
    Ray r = ray(cam.o + d * 140.f, normalize(d));
    float t;
    int id;
    char4 pixel;
    if (intersect_scene(scene, sceneSize, &r, &t, &id)) {
        switch (id) {
            case 0:
                pixel = color(255, 0, 0);
                break;

            case 1:
                pixel = color(0, 255, 0);
                break;

            case 2:
                pixel = color(0, 0, 255);
                break;

            case 3:
                pixel = color(128, 0, 0);
                break;

            case 4:
                pixel = color(0, 128, 0);
                break;

            case 5:
                pixel = color(0, 0, 128);
                break;

            case 6:
                pixel = color(64, 0, 0);
                break;

            case 7:
                pixel = color(0, 64, 0);
                break;

            case 8:
                pixel = color(0, 0, 64);
                break;

            default:
                pixel = color(255, 255, 0);
                break;
        }
    } else {
        pixel = color(0, 0, 0);
    }

    output[(height - y - 1) * stride / 4 + x] = pixel;
}