// A golden-angle spiral. Change 2.399963f to reshape the orbits.
__device__ float wave(float angle);

extern "C" __global__ void particles(float4 *points, const float *params) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = (unsigned int)params[1];
    if (i >= count) return;

    float time = params[0];
    float seed = (float)i;
    float radius = sqrtf((seed + 0.5f) / (float)count);
    float angle = seed * 2.399963f + time * (0.15f + 0.4f / (radius + 0.2f));
    float ripple = wave(radius * 32.0f - time * 2.0f);
    radius = radius * (0.92f + ripple * 0.07f);
    points[i].x = wave(angle + 1.5707963f) * radius * 0.88f;
    points[i].y = wave(angle) * radius * 0.88f;
    points[i].z = radius;
    points[i].w = 0.4f + ripple * 0.3f;
}

__device__ float wave(float angle) {
    float x = angle - floorf(angle / 6.2831853f + 0.5f) * 6.2831853f;
    float y = 1.2732395f * x - 0.4052847f * x * fabsf(x);
    return y + 0.225f * (y * fabsf(y) - y);
}
