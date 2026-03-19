#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

namespace {

    int32_t cuda_code(const cudaError_t status) noexcept {
        return status == cudaSuccess ? 0 : 5001;
    }

    dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
        return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    }

    __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    __device__ int clampi(const int value, const int lo, const int hi) {
        return value < lo ? lo : (value > hi ? hi : value);
    }

    __device__ float fetch_clamped(const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz) {
        return field[index_3d(clampi(x, 0, sx - 1), clampi(y, 0, sy - 1), clampi(z, 0, sz - 1), sx, sy)];
    }

    __global__ void collocated_velocity_magnitude_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = index_3d(x, y, z, nx, ny);
        const float vx   = velocity_x[index];
        const float vy   = velocity_y[index];
        const float vz   = velocity_z[index];
        destination[index] = sqrtf(vx * vx + vy * vy + vz * vz);
    }

    __global__ void staggered_velocity_magnitude_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const float vx = 0.5f * (fetch_clamped(velocity_x, x, y, z, nx + 1, ny, nz) + fetch_clamped(velocity_x, x + 1, y, z, nx + 1, ny, nz));
        const float vy = 0.5f * (fetch_clamped(velocity_y, x, y, z, nx, ny + 1, nz) + fetch_clamped(velocity_y, x, y + 1, z, nx, ny + 1, nz));
        const float vz = 0.5f * (fetch_clamped(velocity_z, x, y, z, nx, ny, nz + 1) + fetch_clamped(velocity_z, x, y, z + 1, nx, ny, nz + 1));
        destination[index_3d(x, y, z, nx, ny)] = sqrtf(vx * vx + vy * vy + vz * vz);
    }

} // namespace

int32_t app_compute_collocated_velocity_magnitude_async(void* destination, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
    if (destination == nullptr) return 2001;
    if (velocity_x == nullptr) return 2002;
    if (velocity_y == nullptr) return 2003;
    if (velocity_z == nullptr) return 2004;

    nvtx3::scoped_range range{"smoke_app.snapshot_velocity_magnitude.collocated"};
    const dim3 block{static_cast<unsigned>(std::max(block_x, 1)), static_cast<unsigned>(std::max(block_y, 1)), static_cast<unsigned>(std::max(block_z, 1))};
    collocated_velocity_magnitude_kernel<<<make_grid(nx, ny, nz, block), block, 0, reinterpret_cast<cudaStream_t>(cuda_stream)>>>(reinterpret_cast<float*>(destination), reinterpret_cast<const float*>(velocity_x), reinterpret_cast<const float*>(velocity_y), reinterpret_cast<const float*>(velocity_z), nx, ny, nz);
    return cuda_code(cudaGetLastError());
}

int32_t app_compute_staggered_velocity_magnitude_async(void* destination, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
    if (destination == nullptr) return 2001;
    if (velocity_x == nullptr) return 2002;
    if (velocity_y == nullptr) return 2003;
    if (velocity_z == nullptr) return 2004;

    nvtx3::scoped_range range{"smoke_app.snapshot_velocity_magnitude.staggered"};
    const dim3 block{static_cast<unsigned>(std::max(block_x, 1)), static_cast<unsigned>(std::max(block_y, 1)), static_cast<unsigned>(std::max(block_z, 1))};
    staggered_velocity_magnitude_kernel<<<make_grid(nx, ny, nz, block), block, 0, reinterpret_cast<cudaStream_t>(cuda_stream)>>>(reinterpret_cast<float*>(destination), reinterpret_cast<const float*>(velocity_x), reinterpret_cast<const float*>(velocity_y), reinterpret_cast<const float*>(velocity_z), nx, ny, nz);
    return cuda_code(cudaGetLastError());
}
