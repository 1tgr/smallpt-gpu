#include "render.h"

#ifndef __OPENCL_VERSION__

#include <math.h>
#define __global
#define __kernel
#define M_PI_F        3.14159265358979323846264338327950288f   /* pi */

#endif

static num xdot(Vec a, Vec b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec xcross(Vec a, Vec b) {
    return vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static Vec xnormalize(Vec v) {
    return v * (1 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z));
}

typedef struct Ray {
    Vec o, d;
} Ray;

static Ray ray(Vec o, Vec d) {
    Ray ray;
    ray.o = o;
    ray.d = d;
    return ray;
}

static num intersect_sphere(__global const Sphere *sphere, Ray r) { // returns distance, 0 if nohit
    Vec op = sphere->p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    num t,
        eps = 1e-4f,
        b = xdot(op, r.d),
        det = b * b - xdot(op, op) + sphere->rad * sphere->rad;
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

static bool intersect_scene(__global const Sphere *scene, const int sceneSize, Ray r, num *t, int *id){
    num d, inf = *t = 1e20f;

    for(int i =sceneSize; i--;) {
        if ((d = intersect_sphere(&scene[i], r)) && d < *t) {
            *t = d;
            *id = i;
        }
    }

    return *t < inf;
}

static void rand_seed(long long *seed, int n) {
    *seed = (n ^ 0x5DEECE66DLL) & ((1LL << 48) - 1);
}

static int rand_next(long long *seed, int bits) {
    *seed = (*seed * 0x5DEECE66DLL + 0xBLL) & ((1LL << 48) - 1);
    return (int) (*seed >> (48 - bits));
}

static num randf(long long *seed) {
    return rand_next(seed, 24) / ((num)(1 << 24));
}

static Vec radiance(__global const Sphere* scene, const int sceneSize, Ray r, long long *seed) {
    Vec color = vec(0, 0 ,0);
    Vec scale = vec(1, 1, 1);
    int depth = 0;
    while (true) {
        num t;                               // distance to intersection
        int id;                               // id of intersected object
        if (!intersect_scene(scene, sceneSize, r, &t, &id)) {
            // if miss, return black
            return color;
        }

        __global const Sphere *obj = scene + id;        // the hit object
        Vec x = r.o + r.d * t,
            n = xnormalize(x - obj->p),
            nl = xdot(n, r.d) < 0 ? n : n * -1,
            f = obj->c;
        num p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
        if (++depth > 5) {
            //R.R.
            if (randf(seed) >= p) {
                return color + obj->e;
            }

            f = f * (1 / p);
        }

        color += scale * obj->e;
        scale *= f;

        if (obj->refl == DIFF) {
            // Ideal DIFFUSE reflection
            num r1 = 2 * M_PI_F * randf(seed),
                r2 = randf(seed),
                r2s = sqrt(r2);
            Vec w = nl,
                u = xnormalize(xcross(fabs(w.x) > .1f ? vec(0, 1, 0) : vec(1, 0, 0), w)),
                v = xcross(w, u),
                d = xnormalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));
            r = ray(x, d);
            continue;
        }

        if (obj->refl == SPEC) {
            // Ideal SPECULAR reflection
            r = ray(x, r.d - n * 2 * xdot(n, r.d));
            continue;
        }

        // Ideal dielectric REFRACTION
        Ray reflRay = ray(x, r.d - n * 2 * xdot(n, r.d));
        bool into = xdot(n, nl) > 0;                // Ray from outside going in?
        num nc = 1, nt = 1.5f, nnt = into ? nc / nt : nt / nc, ddn = xdot(r.d, nl), cos2t;
        if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) {
            // Total internal reflection
            r = reflRay;
            continue;
        }

        Vec tdir = xnormalize(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))));
        num a = nt - nc,
            b = nt + nc,
            R0 = a * a / (b * b),
            c = 1 - (into ? -ddn : xdot(tdir, n)),
            Re = R0 + (1 - R0) * c * c * c * c * c,
            Tr = 1 - Re,
            P = .25f + .5f * Re,
            RP = Re / P,
            TP = Tr / (1 - P);

        if (randf(seed) < P) {
            scale *= RP;
            r = reflRay;
        } else {
            scale *= TP;
            r = ray(x, tdir);
        }
    }
}

__kernel void renderKernel(
    const int width,
    const int height,
    const int samples,
    __global const Sphere *scene,
    const int sceneSize,
    __global Vec *output

#ifdef __OPENCL_VERSION__
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int samp = get_global_id(2);
#else
    ,
    int x,
    int y,
    int samp
) {
#endif

    long long seed;
    rand_seed(&seed, x * y * samp);
    Ray cam = ray(vec(50, 52, 295.6f), xnormalize(vec(0, -0.042612f, -1))); // cam pos, dir
    Vec cx = vec(width * .5135f / height, 0, 0), cy = xnormalize(xcross(cx, cam.d)) * .5135f;
    num r1 = 2 * randf(&seed), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
    num r2 = 2 * randf(&seed), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
    Vec d
        = cx * (((dx + .5f) / 2 + x) / width - .5f)
        + cy * (((dy + .5f) / 2 + y) / height - .5f)
        + cam.d;
    Ray r = ray(cam.o + d * 140.f, xnormalize(d));
    Vec color = radiance(scene, sceneSize, r, &seed) / samples;
    output[(height - y - 1) * width + x] += color;
}