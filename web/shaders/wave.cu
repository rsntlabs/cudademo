// A rippling particle fabric. Try a different wave frequency.
__device__ float wave(float angle);

extern "C" __global__ void particles(float4 *points, const float *params) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = (unsigned int)params[1];
    if (i >= count) return;

    float time = params[0];
    unsigned int side = (unsigned int)sqrtf((float)count);
    float x = (float)(i % side) / (float)side * 2.0f - 1.0f;
    float y = (float)(i / side) / (float)side * 2.0f - 1.0f;
    float distance = sqrtf(x * x + y * y);
    float ripple = wave(distance * 14.0f - time * 2.0f);
    points[i].x = x * 0.84f;
    points[i].y = y * 0.72f + ripple * 0.13f;
    points[i].z = distance * 0.65f;
    points[i].w = 0.3f + (ripple + 1.0f) * 0.25f;
}

__device__ float wave(float angle) {
    float x = angle - floorf(angle / 6.2831853f + 0.5f) * 6.2831853f;
    float y = 1.2732395f * x - 0.4052847f * x * fabsf(x);
    return y + 0.225f * (y * fabsf(y) - y);
}
