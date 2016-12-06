#pragma once

typedef float num;

#ifdef __OPENCL_VERSION__
typedef float3 num3;
#else
typedef cl_float3 num3;
#endif

typedef num3 Vec;

typedef enum { DIFF, SPEC, REFR } Refl_t;  // material types, used in radiance()

typedef struct Sphere {
    num rad;       // radius
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

#ifdef __cplusplus
    Sphere(num rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
#endif
} Sphere;

#ifdef __cplusplus

static Vec vec(num x = 0.0, num y = 0.0, num z = 0.0) {
    Vec vec = { .x = x, .y = y, .z = z };
    return vec;
}

static Vec operator+(Vec a, Vec b) {
    return vec(a.x + b.x, a.y + b.y, a.z + b.z);
}

static Vec& operator+=(Vec& a, Vec b) {
    a = a + b;
    return a;
}

static Vec operator-(Vec a, Vec b) {
    return vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

static Vec& operator-=(Vec& a, Vec b) {
    a = a - b;
    return a;
}

static Vec operator*(Vec a, num b) {
    return vec(a.x * b, a.y * b, a.z * b);
}

static Vec& operator*=(Vec& a, num b) {
    a = a * b;
    return a;
}

static Vec operator*(Vec a, Vec b) {
    return vec(a.x * b.x, a.y * b.y, a.z * b.z);
}

static Vec& operator*=(Vec& a, Vec b) {
    a = a * b;
    return a;
}

static Vec operator/(Vec a, num b) {
    return vec(a.x / b, a.y / b, a.z / b);
}

static Vec& operator/=(Vec& a, num b) {
    a = a / b;
    return a;
}

#else

static Vec vec(num x, num y, num z) {
    Vec vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    return vec;
}

#endif