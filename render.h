#ifdef __OPENCL_VERSION__
typedef float3 Vec;
#else
typedef cl_float3 Vec;
#endif

typedef struct Ray {
    Vec o, d;

#ifdef __cplusplus
    Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
#endif
} Ray;

typedef enum { DIFF, SPEC, REFR } Refl_t;  // material types, used in radiance()

typedef struct Sphere {
    float rad;       // radius
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

#ifdef __cplusplus
    Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
#endif
} Sphere;
