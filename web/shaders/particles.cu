// Each CUDA thread places one particle in a spiral galaxy.
// Try changing the twist (0.22f), the arms (5.0f), or the wave.
// Bindings: 0 = float4 particle buffer, 1 = frame parameters.
__device__ float wave(float angle);

extern "C" __global__ void particles(float4 *points, const float *params) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = (unsigned int)params[1];
    if (i >= count) return;

    float time = params[0];
    float seed = (float)i;
    unsigned int hash = i * 747796405u + 2891336453u;
    hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
    float random = (float)(((hash >> 22u) ^ hash) & 65535u) / 65535.0f;
    float radius = sqrtf((seed + 0.5f) / (float)count);
    float arm = (float)(i % 5u) * 1.256637f;
    float angle = arm + radius * 5.0f + time * 0.22f;
    float scatter = (random - 0.5f) * (0.12f + radius * 0.5f);
    angle = angle + scatter;

    float ripple = wave(radius * 12.0f - time * 1.4f) * 0.035f;
    float x = wave(angle + 1.5707963f) * (radius + ripple);
    float y = wave(angle) * (radius + ripple);
    points[i].x = x * 0.86f;
    points[i].y = y * 0.86f;
    points[i].z = radius;
    points[i].w = 0.5f + random * 0.5f;
}

// A small sine approximation keeps this visual effect inexpensive.
__device__ float wave(float angle) {
    float x = angle - floorf(angle / 6.2831853f + 0.5f) * 6.2831853f;
    float y = 1.2732395f * x - 0.4052847f * x * fabsf(x);
    return y + 0.225f * (y * fabsf(y) - y);
}
