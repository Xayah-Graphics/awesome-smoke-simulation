#include <cuda_runtime.h>
#include <imgui.h>

#include <nvtx3/nvtx3.hpp>
#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include "stable-fluids.h"
#include "visual-simulation-of-smoke.h"

#include <vulkan/vulkan_raii.hpp>

import app;
import std;
import vk.memory;

int32_t app_compute_staggered_velocity_magnitude_async(void* destination, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream);
int32_t app_add_stable_source_async(void* density, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float density_amount, float velocity_source_x, float velocity_source_y, float velocity_source_z, int32_t block_x, int32_t block_y,
    int32_t block_z, void* cuda_stream);
int32_t app_add_visual_source_async(void* density, void* temperature, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float density_amount, float temperature_amount, float velocity_source_x, float velocity_source_y,
    float velocity_source_z, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream);

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    enum class BackendKind : uint32_t {
        StableFluids001 = 0,
        VisualSmoke002  = 1,
    };

    enum class ExecutionBackend : uint32_t {
        Cpu      = 0,
        Parallel = 1,
        Cuda     = 2,
    };

    enum class FieldId : uint32_t {
        Density           = 0,
        Temperature       = 1,
        VelocityMagnitude = 2,
    };

    struct PlaybackState {
        bool paused             = false;
        bool step_once          = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval   = 2;
    };

    struct FieldChoice {
        FieldId id;
        std::string_view label;
        app::FieldSemantic semantic;
    };

    constexpr std::array stable_fields{
        FieldChoice{FieldId::Density, "Density", app::FieldSemantic::Density},
        FieldChoice{FieldId::VelocityMagnitude, "Velocity Magnitude", app::FieldSemantic::VelocityMagnitude},
    };

    constexpr std::array visual_fields{
        FieldChoice{FieldId::Density, "Density", app::FieldSemantic::Density},
        FieldChoice{FieldId::Temperature, "Temperature", app::FieldSemantic::Temperature},
        FieldChoice{FieldId::VelocityMagnitude, "Velocity Magnitude", app::FieldSemantic::VelocityMagnitude},
    };

    struct ViewerRuntime {
        uint64_t field_bytes          = 0;
        uint64_t snapshot_generation  = 0;
        uint64_t submit_serial        = 0;
        uint32_t steps_since_snapshot = 0;
        int active_snapshot_slot      = -1;
    };

    struct SnapshotSlot {
        vk::raii::Buffer buffer{nullptr};
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Semaphore timeline_semaphore{nullptr};
        vk::raii::DescriptorSet descriptor_set{nullptr};
        cudaExternalMemory_t external_memory       = nullptr;
        cudaExternalSemaphore_t external_semaphore = nullptr;
        void* cuda_ptr                             = nullptr;
        uint64_t ready_generation                  = 0;
        uint64_t last_used_submit_serial           = 0;
    };

    struct StableStepConfig {
        int32_t nx                  = 64;
        int32_t ny                  = 96;
        int32_t nz                  = 64;
        float dt                    = 1.0f / 90.0f;
        float cell_size             = 1.0f;
        float viscosity             = 0.00015f;
        float diffusion             = 0.00005f;
        int32_t diffuse_iterations  = 24;
        int32_t pressure_iterations = 80;
        int32_t block_x             = 8;
        int32_t block_y             = 8;
        int32_t block_z             = 4;
    };

    struct StableSettings {
        StableStepConfig desc{};
        int selected_field = 0;
        bool emit_source   = true;
        float source_u      = 0.5f;
        float source_v      = 0.18f;
        float source_w      = 0.5f;
        float source_radius  = 5.0f;
        float density_amount = 0.85f;
        float velocity_x     = 0.0f;
        float velocity_y     = 1.2f;
        float velocity_z     = 0.0f;
    };

    struct StableRuntime {
        cudaStream_t sim_stream              = nullptr;
        float* density                       = nullptr;
        float* velocity_x                    = nullptr;
        float* velocity_y                    = nullptr;
        float* velocity_z                    = nullptr;
        float* temporary_density             = nullptr;
        float* temporary_velocity_x          = nullptr;
        float* temporary_velocity_y          = nullptr;
        float* temporary_velocity_z          = nullptr;
        float* temporary_previous_density    = nullptr;
        float* temporary_previous_velocity_x = nullptr;
        float* temporary_previous_velocity_y = nullptr;
        float* temporary_previous_velocity_z = nullptr;
        float* temporary_pressure            = nullptr;
        float* temporary_divergence          = nullptr;
        std::vector<float> host_density{};
        std::vector<float> host_velocity_x{};
        std::vector<float> host_velocity_y{};
        std::vector<float> host_velocity_z{};
        std::vector<float> host_temporary_density{};
        std::vector<float> host_temporary_velocity_x{};
        std::vector<float> host_temporary_velocity_y{};
        std::vector<float> host_temporary_velocity_z{};
        std::vector<float> host_temporary_previous_density{};
        std::vector<float> host_temporary_previous_velocity_x{};
        std::vector<float> host_temporary_previous_velocity_y{};
        std::vector<float> host_temporary_previous_velocity_z{};
        std::vector<float> host_temporary_pressure{};
        std::vector<float> host_temporary_divergence{};
    };

    struct VisualStepConfig {
        int32_t nx                   = 64;
        int32_t ny                   = 96;
        int32_t nz                   = 64;
        float dt                     = 1.0f / 90.0f;
        float cell_size              = 1.0f;
        float ambient_temperature    = 0.0f;
        float density_buoyancy       = 0.045f;
        float temperature_buoyancy   = 0.12f;
        float vorticity_epsilon      = 2.0f;
        int32_t pressure_iterations  = 80;
        int32_t block_x              = 8;
        int32_t block_y              = 8;
        int32_t block_z              = 4;
        uint32_t use_monotonic_cubic = 1u;
    };

    struct VisualSettings {
        VisualStepConfig desc{};
        int selected_field       = 0;
        bool emit_source         = true;
        float source_u           = 0.5f;
        float source_v           = 0.18f;
        float source_w           = 0.5f;
        float source_radius      = 5.0f;
        float density_amount     = 0.85f;
        float temperature_amount = 1.35f;
        float velocity_x         = 0.0f;
        float velocity_y         = 1.2f;
        float velocity_z         = 0.0f;
    };

    struct VisualRuntime {
        cudaStream_t sim_stream               = nullptr;
        float* density                        = nullptr;
        float* temperature                    = nullptr;
        float* velocity_x                     = nullptr;
        float* velocity_y                     = nullptr;
        float* velocity_z                     = nullptr;
        float* temporary_previous_density     = nullptr;
        float* temporary_previous_temperature = nullptr;
        float* temporary_previous_velocity_x  = nullptr;
        float* temporary_previous_velocity_y  = nullptr;
        float* temporary_previous_velocity_z  = nullptr;
        float* temporary_pressure             = nullptr;
        float* temporary_divergence           = nullptr;
        float* temporary_omega_x              = nullptr;
        float* temporary_omega_y              = nullptr;
        float* temporary_omega_z              = nullptr;
        float* temporary_omega_magnitude      = nullptr;
        float* temporary_force_x              = nullptr;
        float* temporary_force_y              = nullptr;
        float* temporary_force_z              = nullptr;
        std::vector<float> host_density{};
        std::vector<float> host_temperature{};
        std::vector<float> host_velocity_x{};
        std::vector<float> host_velocity_y{};
        std::vector<float> host_velocity_z{};
        std::vector<float> host_temporary_previous_density{};
        std::vector<float> host_temporary_previous_temperature{};
        std::vector<float> host_temporary_previous_velocity_x{};
        std::vector<float> host_temporary_previous_velocity_y{};
        std::vector<float> host_temporary_previous_velocity_z{};
        std::vector<float> host_temporary_pressure{};
        std::vector<float> host_temporary_divergence{};
        std::vector<float> host_temporary_omega_x{};
        std::vector<float> host_temporary_omega_y{};
        std::vector<float> host_temporary_omega_z{};
        std::vector<float> host_temporary_omega_magnitude{};
        std::vector<float> host_temporary_force_x{};
        std::vector<float> host_temporary_force_y{};
        std::vector<float> host_temporary_force_z{};
    };

    auto cuda_ok(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) {
            return;
        }
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }

    auto stable_ok(const int32_t code, const char* what) {
        if (code == 0) {
            return;
        }
        throw std::runtime_error(std::string(what) + " failed");
    }

    auto smoke_ok(const int32_t code, const char* what) {
        if (code == 0) {
            return;
        }
        throw std::runtime_error(std::string(what) + " failed");
    }

} // namespace

int main() {
    try {
        app::FieldRendererApp renderer;
        PlaybackState playback{};
        BackendKind backend_kind = BackendKind::StableFluids001;
        ExecutionBackend execution_backend = ExecutionBackend::Cuda;
        StableSettings stable_settings{};
        StableRuntime stable_runtime{};
        VisualSettings visual_settings{};
        VisualRuntime visual_runtime{};
        ViewerRuntime viewer_runtime{};
        std::vector<SnapshotSlot> snapshot_slots{};

        auto scalar_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_x_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_y_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_z_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz + 1) * sizeof(float); };
        auto scalar_elements = [&](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<size_t>(scalar_bytes(nx, ny, nz) / sizeof(float)); };
        auto velocity_x_elements = [&](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<size_t>(velocity_x_bytes(nx, ny, nz) / sizeof(float)); };
        auto velocity_y_elements = [&](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<size_t>(velocity_y_bytes(nx, ny, nz) / sizeof(float)); };
        auto velocity_z_elements = [&](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<size_t>(velocity_z_bytes(nx, ny, nz) / sizeof(float)); };

        auto add_stable_source_cpu = [&](float* density, float* velocity_x, float* velocity_y, float* velocity_z, const int32_t nx, const int32_t ny, const int32_t nz, const float center_x, const float center_y, const float center_z, const float radius, const float density_amount, const float velocity_source_x,
                                         const float velocity_source_y, const float velocity_source_z) {
            const float radius2 = radius * radius;
            auto stable_index = [](const int x, const int y, const int z, const int sx, const int sy) { return static_cast<size_t>(z) * static_cast<size_t>(sx) * static_cast<size_t>(sy) + static_cast<size_t>(y) * static_cast<size_t>(sx) + static_cast<size_t>(x); };
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x) {
                        const float dx = (static_cast<float>(x) + 0.5f) - center_x;
                        const float dy = (static_cast<float>(y) + 0.5f) - center_y;
                        const float dz = (static_cast<float>(z) + 0.5f) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        density[stable_index(x, y, z, nx, ny)] += density_amount * ((1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f);
                    }
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x <= nx; ++x) {
                        const float dx = static_cast<float>(x) - center_x;
                        const float dy = (static_cast<float>(y) + 0.5f) - center_y;
                        const float dz = (static_cast<float>(z) + 0.5f) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        velocity_x[stable_index(x, y, z, nx + 1, ny)] += velocity_source_x * ((1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f);
                    }
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y <= ny; ++y)
                    for (int x = 0; x < nx; ++x) {
                        const float dx = (static_cast<float>(x) + 0.5f) - center_x;
                        const float dy = static_cast<float>(y) - center_y;
                        const float dz = (static_cast<float>(z) + 0.5f) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        velocity_y[stable_index(x, y, z, nx, ny + 1)] += velocity_source_y * ((1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f);
                    }
            for (int z = 0; z <= nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x) {
                        const float dx = (static_cast<float>(x) + 0.5f) - center_x;
                        const float dy = (static_cast<float>(y) + 0.5f) - center_y;
                        const float dz = static_cast<float>(z) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        velocity_z[stable_index(x, y, z, nx, ny)] += velocity_source_z * ((1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f);
                    }
        };

        auto add_visual_source_cpu = [&](float* density, float* temperature, float* velocity_x, float* velocity_y, float* velocity_z, const int32_t nx, const int32_t ny, const int32_t nz, const float center_x, const float center_y, const float center_z, const float radius, const float density_amount, const float temperature_amount,
                                         const float velocity_source_x, const float velocity_source_y, const float velocity_source_z) {
            const float radius2 = radius * radius;
            auto visual_index = [](const int x, const int y, const int z, const int sx, const int sy) { return static_cast<size_t>(z) * static_cast<size_t>(sx) * static_cast<size_t>(sy) + static_cast<size_t>(y) * static_cast<size_t>(sx) + static_cast<size_t>(x); };
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x) {
                        const float dx = (static_cast<float>(x) + 0.5f) - center_x;
                        const float dy = (static_cast<float>(y) + 0.5f) - center_y;
                        const float dz = (static_cast<float>(z) + 0.5f) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        const float weight = (1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f;
                        const auto index = visual_index(x, y, z, nx, ny);
                        density[index] += density_amount * weight;
                        temperature[index] += temperature_amount * weight;
                    }
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x <= nx; ++x) {
                        const float dx = static_cast<float>(x) - center_x;
                        const float dy = (static_cast<float>(y) + 0.5f) - center_y;
                        const float dz = (static_cast<float>(z) + 0.5f) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        velocity_x[visual_index(x, y, z, nx + 1, ny)] += velocity_source_x * ((1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f);
                    }
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y <= ny; ++y)
                    for (int x = 0; x < nx; ++x) {
                        const float dx = (static_cast<float>(x) + 0.5f) - center_x;
                        const float dy = static_cast<float>(y) - center_y;
                        const float dz = (static_cast<float>(z) + 0.5f) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        velocity_y[visual_index(x, y, z, nx, ny + 1)] += velocity_source_y * ((1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f);
                    }
            for (int z = 0; z <= nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x) {
                        const float dx = (static_cast<float>(x) + 0.5f) - center_x;
                        const float dy = (static_cast<float>(y) + 0.5f) - center_y;
                        const float dz = static_cast<float>(z) - center_z;
                        const float dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 > radius2) continue;
                        velocity_z[visual_index(x, y, z, nx, ny)] += velocity_source_z * ((1.0f - dist2 / radius2) > 0.0f ? (1.0f - dist2 / radius2) : 0.0f);
                    }
        };

        auto current_fields = [&]() -> std::span<const FieldChoice> { return backend_kind == BackendKind::StableFluids001 ? std::span<const FieldChoice>(stable_fields) : std::span<const FieldChoice>(visual_fields); };

        auto current_field_index = [&]() -> int& { return backend_kind == BackendKind::StableFluids001 ? stable_settings.selected_field : visual_settings.selected_field; };

        auto current_field = [&]() -> const FieldChoice& {
            auto fields    = current_fields();
            auto& selected = current_field_index();
            selected       = std::clamp(selected, 0, static_cast<int>(fields.size()) - 1);
            return fields[static_cast<size_t>(selected)];
        };

        auto current_stream = [&]() -> cudaStream_t {
            if (execution_backend != ExecutionBackend::Cuda) return nullptr;
            return backend_kind == BackendKind::StableFluids001 ? stable_runtime.sim_stream : visual_runtime.sim_stream;
        };

        auto current_grid = [&]() {
            struct GridInfo {
                uint32_t nx;
                uint32_t ny;
                uint32_t nz;
                float cell_size;
            };

            return backend_kind == BackendKind::StableFluids001
                ? GridInfo{
                      static_cast<uint32_t>(stable_settings.desc.nx),
                      static_cast<uint32_t>(stable_settings.desc.ny),
                      static_cast<uint32_t>(stable_settings.desc.nz),
                      stable_settings.desc.cell_size,
                  }
                : GridInfo{
                      static_cast<uint32_t>(visual_settings.desc.nx),
                      static_cast<uint32_t>(visual_settings.desc.ny),
                      static_cast<uint32_t>(visual_settings.desc.nz),
                      visual_settings.desc.cell_size,
                  };
        };

        auto apply_field_defaults = [&](const FieldChoice& field) {
            auto& render = renderer.render_settings();
            if (field.id == FieldId::Density) {
                render.mode          = app::RenderMode::Smoke;
                render.density_scale = 1.0f;
                render.absorption    = 1.4f;
            } else if (field.id == FieldId::Temperature) {
                render.mode           = app::RenderMode::Scalar;
                render.scalar_min     = 0.0f;
                render.scalar_max     = 2.0f;
                render.scalar_opacity = 2.0f;
                render.scalar_low_r   = 0.08f;
                render.scalar_low_g   = 0.16f;
                render.scalar_low_b   = 0.45f;
                render.scalar_high_r  = 0.98f;
                render.scalar_high_g  = 0.42f;
                render.scalar_high_b  = 0.12f;
            } else {
                render.mode           = app::RenderMode::Scalar;
                render.scalar_min     = 0.0f;
                render.scalar_max     = 3.0f;
                render.scalar_opacity = 1.6f;
                render.scalar_low_r   = 0.04f;
                render.scalar_low_g   = 0.18f;
                render.scalar_low_b   = 0.36f;
                render.scalar_high_r  = 0.74f;
                render.scalar_high_g  = 0.94f;
                render.scalar_high_b  = 0.96f;
            }
        };

        auto active_snapshot = [&]() -> std::optional<app::ScalarFieldView> {
            if (viewer_runtime.active_snapshot_slot < 0) {
                return std::nullopt;
            }

            const auto& field = current_field();
            const auto grid   = current_grid();
            const auto& slot  = snapshot_slots.at(static_cast<size_t>(viewer_runtime.active_snapshot_slot));
            return app::ScalarFieldView{
                .descriptor_set     = *slot.descriptor_set,
                .timeline_semaphore = *slot.timeline_semaphore,
                .ready_generation   = slot.ready_generation,
                .nx                 = grid.nx,
                .ny                 = grid.ny,
                .nz                 = grid.nz,
                .cell_size          = grid.cell_size,
                .semantic           = field.semantic,
                .label              = field.label,
            };
        };

        auto destroy_snapshot_slots = [&]() {
            for (auto& slot : snapshot_slots) {
                if (slot.cuda_ptr != nullptr) {
                    cudaFree(slot.cuda_ptr);
                    slot.cuda_ptr = nullptr;
                }
                if (slot.external_semaphore != nullptr) {
                    cudaDestroyExternalSemaphore(slot.external_semaphore);
                    slot.external_semaphore = nullptr;
                }
                if (slot.external_memory != nullptr) {
                    cudaDestroyExternalMemory(slot.external_memory);
                    slot.external_memory = nullptr;
                }
                slot.ready_generation        = 0;
                slot.last_used_submit_serial = 0;
            }
            snapshot_slots.clear();
            viewer_runtime.field_bytes          = 0;
            viewer_runtime.snapshot_generation  = 0;
            viewer_runtime.submit_serial        = 0;
            viewer_runtime.steps_since_snapshot = 0;
            viewer_runtime.active_snapshot_slot = -1;
        };

        auto destroy_stable_backend = [&]() {
            if (stable_runtime.sim_stream != nullptr) {
                cudaStreamSynchronize(stable_runtime.sim_stream);
            }
            if (stable_runtime.density != nullptr) {
                cudaFree(stable_runtime.density);
                stable_runtime.density = nullptr;
            }
            if (stable_runtime.velocity_x != nullptr) {
                cudaFree(stable_runtime.velocity_x);
                stable_runtime.velocity_x = nullptr;
            }
            if (stable_runtime.velocity_y != nullptr) {
                cudaFree(stable_runtime.velocity_y);
                stable_runtime.velocity_y = nullptr;
            }
            if (stable_runtime.velocity_z != nullptr) {
                cudaFree(stable_runtime.velocity_z);
                stable_runtime.velocity_z = nullptr;
            }
            if (stable_runtime.temporary_density != nullptr) {
                cudaFree(stable_runtime.temporary_density);
                stable_runtime.temporary_density = nullptr;
            }
            if (stable_runtime.temporary_velocity_x != nullptr) {
                cudaFree(stable_runtime.temporary_velocity_x);
                stable_runtime.temporary_velocity_x = nullptr;
            }
            if (stable_runtime.temporary_velocity_y != nullptr) {
                cudaFree(stable_runtime.temporary_velocity_y);
                stable_runtime.temporary_velocity_y = nullptr;
            }
            if (stable_runtime.temporary_velocity_z != nullptr) {
                cudaFree(stable_runtime.temporary_velocity_z);
                stable_runtime.temporary_velocity_z = nullptr;
            }
            if (stable_runtime.temporary_previous_density != nullptr) {
                cudaFree(stable_runtime.temporary_previous_density);
                stable_runtime.temporary_previous_density = nullptr;
            }
            if (stable_runtime.temporary_previous_velocity_x != nullptr) {
                cudaFree(stable_runtime.temporary_previous_velocity_x);
                stable_runtime.temporary_previous_velocity_x = nullptr;
            }
            if (stable_runtime.temporary_previous_velocity_y != nullptr) {
                cudaFree(stable_runtime.temporary_previous_velocity_y);
                stable_runtime.temporary_previous_velocity_y = nullptr;
            }
            if (stable_runtime.temporary_previous_velocity_z != nullptr) {
                cudaFree(stable_runtime.temporary_previous_velocity_z);
                stable_runtime.temporary_previous_velocity_z = nullptr;
            }
            if (stable_runtime.temporary_pressure != nullptr) {
                cudaFree(stable_runtime.temporary_pressure);
                stable_runtime.temporary_pressure = nullptr;
            }
            if (stable_runtime.temporary_divergence != nullptr) {
                cudaFree(stable_runtime.temporary_divergence);
                stable_runtime.temporary_divergence = nullptr;
            }
            if (stable_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(stable_runtime.sim_stream);
                stable_runtime.sim_stream = nullptr;
            }
            stable_runtime.host_density.clear();
            stable_runtime.host_velocity_x.clear();
            stable_runtime.host_velocity_y.clear();
            stable_runtime.host_velocity_z.clear();
            stable_runtime.host_temporary_density.clear();
            stable_runtime.host_temporary_velocity_x.clear();
            stable_runtime.host_temporary_velocity_y.clear();
            stable_runtime.host_temporary_velocity_z.clear();
            stable_runtime.host_temporary_previous_density.clear();
            stable_runtime.host_temporary_previous_velocity_x.clear();
            stable_runtime.host_temporary_previous_velocity_y.clear();
            stable_runtime.host_temporary_previous_velocity_z.clear();
            stable_runtime.host_temporary_pressure.clear();
            stable_runtime.host_temporary_divergence.clear();
            stable_runtime.density = nullptr;
            stable_runtime.velocity_x = nullptr;
            stable_runtime.velocity_y = nullptr;
            stable_runtime.velocity_z = nullptr;
            stable_runtime.temporary_density = nullptr;
            stable_runtime.temporary_velocity_x = nullptr;
            stable_runtime.temporary_velocity_y = nullptr;
            stable_runtime.temporary_velocity_z = nullptr;
            stable_runtime.temporary_previous_density = nullptr;
            stable_runtime.temporary_previous_velocity_x = nullptr;
            stable_runtime.temporary_previous_velocity_y = nullptr;
            stable_runtime.temporary_previous_velocity_z = nullptr;
            stable_runtime.temporary_pressure = nullptr;
            stable_runtime.temporary_divergence = nullptr;
        };

        auto destroy_visual_backend = [&]() {
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamSynchronize(visual_runtime.sim_stream);
            }
            if (visual_runtime.density != nullptr) {
                cudaFree(visual_runtime.density);
                visual_runtime.density = nullptr;
            }
            if (visual_runtime.temperature != nullptr) {
                cudaFree(visual_runtime.temperature);
                visual_runtime.temperature = nullptr;
            }
            if (visual_runtime.velocity_x != nullptr) {
                cudaFree(visual_runtime.velocity_x);
                visual_runtime.velocity_x = nullptr;
            }
            if (visual_runtime.velocity_y != nullptr) {
                cudaFree(visual_runtime.velocity_y);
                visual_runtime.velocity_y = nullptr;
            }
            if (visual_runtime.velocity_z != nullptr) {
                cudaFree(visual_runtime.velocity_z);
                visual_runtime.velocity_z = nullptr;
            }
            if (visual_runtime.temporary_previous_density != nullptr) {
                cudaFree(visual_runtime.temporary_previous_density);
                visual_runtime.temporary_previous_density = nullptr;
            }
            if (visual_runtime.temporary_previous_temperature != nullptr) {
                cudaFree(visual_runtime.temporary_previous_temperature);
                visual_runtime.temporary_previous_temperature = nullptr;
            }
            if (visual_runtime.temporary_previous_velocity_x != nullptr) {
                cudaFree(visual_runtime.temporary_previous_velocity_x);
                visual_runtime.temporary_previous_velocity_x = nullptr;
            }
            if (visual_runtime.temporary_previous_velocity_y != nullptr) {
                cudaFree(visual_runtime.temporary_previous_velocity_y);
                visual_runtime.temporary_previous_velocity_y = nullptr;
            }
            if (visual_runtime.temporary_previous_velocity_z != nullptr) {
                cudaFree(visual_runtime.temporary_previous_velocity_z);
                visual_runtime.temporary_previous_velocity_z = nullptr;
            }
            if (visual_runtime.temporary_pressure != nullptr) {
                cudaFree(visual_runtime.temporary_pressure);
                visual_runtime.temporary_pressure = nullptr;
            }
            if (visual_runtime.temporary_divergence != nullptr) {
                cudaFree(visual_runtime.temporary_divergence);
                visual_runtime.temporary_divergence = nullptr;
            }
            if (visual_runtime.temporary_omega_x != nullptr) {
                cudaFree(visual_runtime.temporary_omega_x);
                visual_runtime.temporary_omega_x = nullptr;
            }
            if (visual_runtime.temporary_omega_y != nullptr) {
                cudaFree(visual_runtime.temporary_omega_y);
                visual_runtime.temporary_omega_y = nullptr;
            }
            if (visual_runtime.temporary_omega_z != nullptr) {
                cudaFree(visual_runtime.temporary_omega_z);
                visual_runtime.temporary_omega_z = nullptr;
            }
            if (visual_runtime.temporary_omega_magnitude != nullptr) {
                cudaFree(visual_runtime.temporary_omega_magnitude);
                visual_runtime.temporary_omega_magnitude = nullptr;
            }
            if (visual_runtime.temporary_force_x != nullptr) {
                cudaFree(visual_runtime.temporary_force_x);
                visual_runtime.temporary_force_x = nullptr;
            }
            if (visual_runtime.temporary_force_y != nullptr) {
                cudaFree(visual_runtime.temporary_force_y);
                visual_runtime.temporary_force_y = nullptr;
            }
            if (visual_runtime.temporary_force_z != nullptr) {
                cudaFree(visual_runtime.temporary_force_z);
                visual_runtime.temporary_force_z = nullptr;
            }
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(visual_runtime.sim_stream);
                visual_runtime.sim_stream = nullptr;
            }
            visual_runtime.host_density.clear();
            visual_runtime.host_temperature.clear();
            visual_runtime.host_velocity_x.clear();
            visual_runtime.host_velocity_y.clear();
            visual_runtime.host_velocity_z.clear();
            visual_runtime.host_temporary_previous_density.clear();
            visual_runtime.host_temporary_previous_temperature.clear();
            visual_runtime.host_temporary_previous_velocity_x.clear();
            visual_runtime.host_temporary_previous_velocity_y.clear();
            visual_runtime.host_temporary_previous_velocity_z.clear();
            visual_runtime.host_temporary_pressure.clear();
            visual_runtime.host_temporary_divergence.clear();
            visual_runtime.host_temporary_omega_x.clear();
            visual_runtime.host_temporary_omega_y.clear();
            visual_runtime.host_temporary_omega_z.clear();
            visual_runtime.host_temporary_omega_magnitude.clear();
            visual_runtime.host_temporary_force_x.clear();
            visual_runtime.host_temporary_force_y.clear();
            visual_runtime.host_temporary_force_z.clear();
            visual_runtime.density = nullptr;
            visual_runtime.temperature = nullptr;
            visual_runtime.velocity_x = nullptr;
            visual_runtime.velocity_y = nullptr;
            visual_runtime.velocity_z = nullptr;
            visual_runtime.temporary_previous_density = nullptr;
            visual_runtime.temporary_previous_temperature = nullptr;
            visual_runtime.temporary_previous_velocity_x = nullptr;
            visual_runtime.temporary_previous_velocity_y = nullptr;
            visual_runtime.temporary_previous_velocity_z = nullptr;
            visual_runtime.temporary_pressure = nullptr;
            visual_runtime.temporary_divergence = nullptr;
            visual_runtime.temporary_omega_x = nullptr;
            visual_runtime.temporary_omega_y = nullptr;
            visual_runtime.temporary_omega_z = nullptr;
            visual_runtime.temporary_omega_magnitude = nullptr;
            visual_runtime.temporary_force_x = nullptr;
            visual_runtime.temporary_force_y = nullptr;
            visual_runtime.temporary_force_z = nullptr;
        };

        auto destroy_everything = [&]() {
            nvtx3::scoped_range range{"smoke_app.destroy_everything"};
            renderer.vk_context().device.waitIdle();
            destroy_snapshot_slots();
            destroy_stable_backend();
            destroy_visual_backend();
        };

        auto create_snapshot_slots = [&]() {
            std::vector<vk::raii::DescriptorSet> descriptor_sets = renderer.allocate_field_descriptor_sets(snapshot_slot_count);
            snapshot_slots.clear();
            snapshot_slots.reserve(snapshot_slot_count);

            for (uint32_t slot_index = 0; slot_index < snapshot_slot_count; ++slot_index) {
                SnapshotSlot slot{};
                slot.descriptor_set = std::move(descriptor_sets[slot_index]);

                vk::SemaphoreTypeCreateInfo timeline_semaphore_ci{
                    .semaphoreType = vk::SemaphoreType::eTimeline,
                    .initialValue  = 0,
                };
                vk::ExportSemaphoreCreateInfo export_semaphore_ci{
                    .pNext       = &timeline_semaphore_ci,
                    .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32,
                };
                vk::SemaphoreCreateInfo semaphore_ci{
                    .pNext = &export_semaphore_ci,
                };
                slot.timeline_semaphore = vk::raii::Semaphore{renderer.vk_context().device, semaphore_ci};

                vk::ExternalMemoryBufferCreateInfo external_buffer_ci{
                    .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
                };
                vk::BufferCreateInfo buffer_ci{
                    .pNext       = &external_buffer_ci,
                    .size        = viewer_runtime.field_bytes,
                    .usage       = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                    .sharingMode = vk::SharingMode::eExclusive,
                };
                slot.buffer = vk::raii::Buffer{renderer.vk_context().device, buffer_ci};

                const vk::MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
                vk::ExportMemoryAllocateInfo export_memory_ci{
                    .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
                };
                vk::MemoryAllocateInfo alloc_ci{
                    .pNext           = &export_memory_ci,
                    .allocationSize  = requirements.size,
                    .memoryTypeIndex = vk::memory::find_memory_type(renderer.vk_context().physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
                };
                slot.memory = vk::raii::DeviceMemory{renderer.vk_context().device, alloc_ci};
                slot.buffer.bindMemory(*slot.memory, 0);

#if defined(_WIN32)
                vk::MemoryGetWin32HandleInfoKHR handle_info{
                    .memory     = *slot.memory,
                    .handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
                };
                HANDLE memory_handle = renderer.vk_context().device.getMemoryWin32HandleKHR(handle_info);
                if (memory_handle == nullptr) {
                    throw std::runtime_error("getMemoryWin32HandleKHR returned a null handle");
                }

                cudaExternalMemoryHandleDesc external_desc{};
                external_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
                external_desc.handle.win32.handle = memory_handle;
                external_desc.size                = requirements.size;
                cuda_ok(cudaImportExternalMemory(&slot.external_memory, &external_desc), "cudaImportExternalMemory");
                CloseHandle(memory_handle);

                cudaExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.offset = 0;
                buffer_desc.size   = viewer_runtime.field_bytes;
                cuda_ok(cudaExternalMemoryGetMappedBuffer(&slot.cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");

                vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                    .semaphore  = *slot.timeline_semaphore,
                    .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32,
                };
                HANDLE semaphore_handle = renderer.vk_context().device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
                if (semaphore_handle == nullptr) {
                    throw std::runtime_error("getSemaphoreWin32HandleKHR returned a null handle");
                }

                cudaExternalSemaphoreHandleDesc external_semaphore_desc{};
                external_semaphore_desc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
                external_semaphore_desc.handle.win32.handle = semaphore_handle;
                cuda_ok(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
                CloseHandle(semaphore_handle);
#else
                throw std::runtime_error("smoke-visualizer currently requires Windows external memory interop");
#endif

                vk::DescriptorBufferInfo field_info{
                    .buffer = *slot.buffer,
                    .offset = 0,
                    .range  = viewer_runtime.field_bytes,
                };
                vk::WriteDescriptorSet field_write{
                    .dstSet          = *slot.descriptor_set,
                    .dstBinding      = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo     = &field_info,
                };
                renderer.vk_context().device.updateDescriptorSets(field_write, {});
                snapshot_slots.push_back(std::move(slot));
            }
        };

        auto find_available_snapshot_slot = [&]() -> int {
            for (uint32_t slot_index = 0; slot_index < snapshot_slots.size(); ++slot_index) {
                auto& slot = snapshot_slots[slot_index];
                if (static_cast<int>(slot_index) == viewer_runtime.active_snapshot_slot) {
                    continue;
                }
                if (slot.ready_generation != 0 && viewer_runtime.submit_serial < slot.last_used_submit_serial + renderer.frames_in_flight() + 1) {
                    continue;
                }
                return static_cast<int>(slot_index);
            }
            return -1;
        };

        auto snapshot_current_field_to_slot = [&](const int slot_index, const char* tag) {
            nvtx3::scoped_range range{tag};
            auto& slot        = snapshot_slots.at(static_cast<size_t>(slot_index));
            const auto& field = current_field();
            const auto grid   = current_grid();

            if (backend_kind == BackendKind::StableFluids001) {
                if (execution_backend == ExecutionBackend::Cuda) {
                    if (field.id == FieldId::Density) {
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, stable_runtime.density, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, stable_runtime.sim_stream), "cudaMemcpyAsync stable density snapshot");
                    } else {
                        stable_ok(app_compute_staggered_velocity_magnitude_async(slot.cuda_ptr, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz), stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z,
                                      stable_runtime.sim_stream),
                            "app_compute_staggered_velocity_magnitude_async");
                    }
                } else {
                    if (field.id == FieldId::Density) {
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, stable_runtime.density, viewer_runtime.field_bytes, cudaMemcpyHostToDevice, nullptr), "cudaMemcpyAsync stable density cpu snapshot");
                    } else {
                        auto stable_index = [](const int x, const int y, const int z, const int sx, const int sy) { return static_cast<size_t>(z) * static_cast<size_t>(sx) * static_cast<size_t>(sy) + static_cast<size_t>(y) * static_cast<size_t>(sx) + static_cast<size_t>(x); };
                        for (uint32_t z = 0; z < grid.nz; ++z)
                            for (uint32_t y = 0; y < grid.ny; ++y)
                                for (uint32_t x = 0; x < grid.nx; ++x) {
                                    const float vx = 0.5f * (stable_runtime.velocity_x[stable_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx) + 1, static_cast<int>(grid.ny))] + stable_runtime.velocity_x[stable_index(static_cast<int>(x) + 1, static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx) + 1, static_cast<int>(grid.ny))]);
                                    const float vy = 0.5f * (stable_runtime.velocity_y[stable_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny) + 1)] + stable_runtime.velocity_y[stable_index(static_cast<int>(x), static_cast<int>(y) + 1, static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny) + 1)]);
                                    const float vz = 0.5f * (stable_runtime.velocity_z[stable_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny))] + stable_runtime.velocity_z[stable_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) + 1, static_cast<int>(grid.nx), static_cast<int>(grid.ny))]);
                                    stable_runtime.temporary_density[stable_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny))] = std::sqrt(vx * vx + vy * vy + vz * vz);
                                }
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, stable_runtime.temporary_density, viewer_runtime.field_bytes, cudaMemcpyHostToDevice, nullptr), "cudaMemcpyAsync stable velocity magnitude cpu snapshot");
                    }
                }
            } else {
                if (execution_backend == ExecutionBackend::Cuda) {
                    if (field.id == FieldId::Density) {
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.density, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual density snapshot");
                    } else if (field.id == FieldId::Temperature) {
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.temperature, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual temperature snapshot");
                    } else {
                        smoke_ok(app_compute_staggered_velocity_magnitude_async(slot.cuda_ptr, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz), visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z,
                                     visual_runtime.sim_stream),
                            "app_compute_staggered_velocity_magnitude_async");
                    }
                } else {
                    auto visual_index = [](const int x, const int y, const int z, const int sx, const int sy) { return static_cast<size_t>(z) * static_cast<size_t>(sx) * static_cast<size_t>(sy) + static_cast<size_t>(y) * static_cast<size_t>(sx) + static_cast<size_t>(x); };
                    if (field.id == FieldId::Density) {
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.density, viewer_runtime.field_bytes, cudaMemcpyHostToDevice, nullptr), "cudaMemcpyAsync visual density cpu snapshot");
                    } else if (field.id == FieldId::Temperature) {
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.temperature, viewer_runtime.field_bytes, cudaMemcpyHostToDevice, nullptr), "cudaMemcpyAsync visual temperature cpu snapshot");
                    } else {
                        for (uint32_t z = 0; z < grid.nz; ++z)
                            for (uint32_t y = 0; y < grid.ny; ++y)
                                for (uint32_t x = 0; x < grid.nx; ++x) {
                                    const float vx = 0.5f * (visual_runtime.velocity_x[visual_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx) + 1, static_cast<int>(grid.ny))] + visual_runtime.velocity_x[visual_index(static_cast<int>(x) + 1, static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx) + 1, static_cast<int>(grid.ny))]);
                                    const float vy = 0.5f * (visual_runtime.velocity_y[visual_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny) + 1)] + visual_runtime.velocity_y[visual_index(static_cast<int>(x), static_cast<int>(y) + 1, static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny) + 1)]);
                                    const float vz = 0.5f * (visual_runtime.velocity_z[visual_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny))] + visual_runtime.velocity_z[visual_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) + 1, static_cast<int>(grid.nx), static_cast<int>(grid.ny))]);
                                    visual_runtime.temporary_pressure[visual_index(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(grid.nx), static_cast<int>(grid.ny))] = std::sqrt(vx * vx + vy * vy + vz * vz);
                                }
                        cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.temporary_pressure, viewer_runtime.field_bytes, cudaMemcpyHostToDevice, nullptr), "cudaMemcpyAsync visual velocity magnitude cpu snapshot");
                    }
                }
            }

            const uint64_t next_generation = viewer_runtime.snapshot_generation + 1;
            cudaExternalSemaphoreSignalParams signal_params{};
            signal_params.params.fence.value = next_generation;
            cuda_ok(cudaSignalExternalSemaphoresAsync(&slot.external_semaphore, &signal_params, 1, current_stream()), "cudaSignalExternalSemaphoresAsync snapshot");
            cuda_ok(cudaStreamSynchronize(current_stream()), "cudaStreamSynchronize snapshot");
            slot.ready_generation               = next_generation;
            viewer_runtime.snapshot_generation  = next_generation;
            viewer_runtime.active_snapshot_slot = slot_index;
            viewer_runtime.steps_since_snapshot = 0;
        };

        auto reset_backend = [&]() {
            nvtx3::scoped_range range{"smoke_app.reset_backend"};

            const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
            if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) {
                throw std::runtime_error("smoke-visualizer requires Vulkan timeline semaphore support");
            }

            int cuda_device_index = 0;
            cuda_ok(cudaGetDevice(&cuda_device_index), "cudaGetDevice");
            int cuda_timeline_semaphore_interop_supported = 0;
            cuda_ok(cudaDeviceGetAttribute(&cuda_timeline_semaphore_interop_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index), "cudaDeviceGetAttribute cudaDevAttrTimelineSemaphoreInteropSupported");
            if (cuda_timeline_semaphore_interop_supported == 0) {
                throw std::runtime_error("smoke-visualizer requires CUDA timeline semaphore interop support");
            }

            destroy_everything();

            if (backend_kind == BackendKind::StableFluids001) {
                viewer_runtime.field_bytes     = scalar_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                const auto stable_velocity_x_bytes = velocity_x_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                const auto stable_velocity_y_bytes = velocity_y_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                const auto stable_velocity_z_bytes = velocity_z_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                if (execution_backend == ExecutionBackend::Cuda) {
                    cuda_ok(cudaStreamCreateWithFlags(&stable_runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags stable_stream");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.density), viewer_runtime.field_bytes), "cudaMalloc stable density");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_x), stable_velocity_x_bytes), "cudaMalloc stable velocity_x");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_y), stable_velocity_y_bytes), "cudaMalloc stable velocity_y");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_z), stable_velocity_z_bytes), "cudaMalloc stable velocity_z");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_density), viewer_runtime.field_bytes), "cudaMalloc stable temporary_density");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_velocity_x), stable_velocity_x_bytes), "cudaMalloc stable temporary_velocity_x");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_velocity_y), stable_velocity_y_bytes), "cudaMalloc stable temporary_velocity_y");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_velocity_z), stable_velocity_z_bytes), "cudaMalloc stable temporary_velocity_z");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_density), viewer_runtime.field_bytes), "cudaMalloc stable temporary_previous_density");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_velocity_x), stable_velocity_x_bytes), "cudaMalloc stable temporary_previous_velocity_x");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_velocity_y), stable_velocity_y_bytes), "cudaMalloc stable temporary_previous_velocity_y");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_velocity_z), stable_velocity_z_bytes), "cudaMalloc stable temporary_previous_velocity_z");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_pressure), viewer_runtime.field_bytes), "cudaMalloc stable temporary_pressure");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_divergence), viewer_runtime.field_bytes), "cudaMalloc stable temporary_divergence");
                } else {
                    stable_runtime.host_density.resize(scalar_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_velocity_x.resize(velocity_x_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_velocity_y.resize(velocity_y_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_velocity_z.resize(velocity_z_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_density.resize(scalar_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_velocity_x.resize(velocity_x_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_velocity_y.resize(velocity_y_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_velocity_z.resize(velocity_z_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_previous_density.resize(scalar_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_previous_velocity_x.resize(velocity_x_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_previous_velocity_y.resize(velocity_y_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_previous_velocity_z.resize(velocity_z_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_pressure.resize(scalar_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.host_temporary_divergence.resize(scalar_elements(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz));
                    stable_runtime.density = stable_runtime.host_density.data();
                    stable_runtime.velocity_x = stable_runtime.host_velocity_x.data();
                    stable_runtime.velocity_y = stable_runtime.host_velocity_y.data();
                    stable_runtime.velocity_z = stable_runtime.host_velocity_z.data();
                    stable_runtime.temporary_density = stable_runtime.host_temporary_density.data();
                    stable_runtime.temporary_velocity_x = stable_runtime.host_temporary_velocity_x.data();
                    stable_runtime.temporary_velocity_y = stable_runtime.host_temporary_velocity_y.data();
                    stable_runtime.temporary_velocity_z = stable_runtime.host_temporary_velocity_z.data();
                    stable_runtime.temporary_previous_density = stable_runtime.host_temporary_previous_density.data();
                    stable_runtime.temporary_previous_velocity_x = stable_runtime.host_temporary_previous_velocity_x.data();
                    stable_runtime.temporary_previous_velocity_y = stable_runtime.host_temporary_previous_velocity_y.data();
                    stable_runtime.temporary_previous_velocity_z = stable_runtime.host_temporary_previous_velocity_z.data();
                    stable_runtime.temporary_pressure = stable_runtime.host_temporary_pressure.data();
                    stable_runtime.temporary_divergence = stable_runtime.host_temporary_divergence.data();
                }
            } else {
                viewer_runtime.field_bytes  = scalar_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                const auto visual_velocity_x_bytes = velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                const auto visual_velocity_y_bytes = velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                const auto visual_velocity_z_bytes = velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                if (execution_backend == ExecutionBackend::Cuda) {
                    cuda_ok(cudaStreamCreateWithFlags(&visual_runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags visual_stream");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.density), viewer_runtime.field_bytes), "cudaMalloc visual density");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temperature), viewer_runtime.field_bytes), "cudaMalloc visual temperature");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.velocity_x), visual_velocity_x_bytes), "cudaMalloc visual velocity_x");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.velocity_y), visual_velocity_y_bytes), "cudaMalloc visual velocity_y");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.velocity_z), visual_velocity_z_bytes), "cudaMalloc visual velocity_z");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_previous_density), viewer_runtime.field_bytes), "cudaMalloc visual temporary_previous_density");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_previous_temperature), viewer_runtime.field_bytes), "cudaMalloc visual temporary_previous_temperature");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_previous_velocity_x), visual_velocity_x_bytes), "cudaMalloc visual temporary_previous_velocity_x");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_previous_velocity_y), visual_velocity_y_bytes), "cudaMalloc visual temporary_previous_velocity_y");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_previous_velocity_z), visual_velocity_z_bytes), "cudaMalloc visual temporary_previous_velocity_z");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_pressure), viewer_runtime.field_bytes), "cudaMalloc visual temporary_pressure");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_divergence), viewer_runtime.field_bytes), "cudaMalloc visual temporary_divergence");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_omega_x), viewer_runtime.field_bytes), "cudaMalloc visual temporary_omega_x");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_omega_y), viewer_runtime.field_bytes), "cudaMalloc visual temporary_omega_y");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_omega_z), viewer_runtime.field_bytes), "cudaMalloc visual temporary_omega_z");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_omega_magnitude), viewer_runtime.field_bytes), "cudaMalloc visual temporary_omega_magnitude");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_force_x), viewer_runtime.field_bytes), "cudaMalloc visual temporary_force_x");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_force_y), viewer_runtime.field_bytes), "cudaMalloc visual temporary_force_y");
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temporary_force_z), viewer_runtime.field_bytes), "cudaMalloc visual temporary_force_z");
                } else {
                    visual_runtime.host_density.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temperature.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_velocity_x.resize(velocity_x_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_velocity_y.resize(velocity_y_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_velocity_z.resize(velocity_z_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_previous_density.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_previous_temperature.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_previous_velocity_x.resize(velocity_x_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_previous_velocity_y.resize(velocity_y_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_previous_velocity_z.resize(velocity_z_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_pressure.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_divergence.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_omega_x.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_omega_y.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_omega_z.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_omega_magnitude.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_force_x.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_force_y.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.host_temporary_force_z.resize(scalar_elements(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz));
                    visual_runtime.density = visual_runtime.host_density.data();
                    visual_runtime.temperature = visual_runtime.host_temperature.data();
                    visual_runtime.velocity_x = visual_runtime.host_velocity_x.data();
                    visual_runtime.velocity_y = visual_runtime.host_velocity_y.data();
                    visual_runtime.velocity_z = visual_runtime.host_velocity_z.data();
                    visual_runtime.temporary_previous_density = visual_runtime.host_temporary_previous_density.data();
                    visual_runtime.temporary_previous_temperature = visual_runtime.host_temporary_previous_temperature.data();
                    visual_runtime.temporary_previous_velocity_x = visual_runtime.host_temporary_previous_velocity_x.data();
                    visual_runtime.temporary_previous_velocity_y = visual_runtime.host_temporary_previous_velocity_y.data();
                    visual_runtime.temporary_previous_velocity_z = visual_runtime.host_temporary_previous_velocity_z.data();
                    visual_runtime.temporary_pressure = visual_runtime.host_temporary_pressure.data();
                    visual_runtime.temporary_divergence = visual_runtime.host_temporary_divergence.data();
                    visual_runtime.temporary_omega_x = visual_runtime.host_temporary_omega_x.data();
                    visual_runtime.temporary_omega_y = visual_runtime.host_temporary_omega_y.data();
                    visual_runtime.temporary_omega_z = visual_runtime.host_temporary_omega_z.data();
                    visual_runtime.temporary_omega_magnitude = visual_runtime.host_temporary_omega_magnitude.data();
                    visual_runtime.temporary_force_x = visual_runtime.host_temporary_force_x.data();
                    visual_runtime.temporary_force_y = visual_runtime.host_temporary_force_y.data();
                    visual_runtime.temporary_force_z = visual_runtime.host_temporary_force_z.data();
                }
            }

            create_snapshot_slots();

            if (backend_kind == BackendKind::StableFluids001) {
                if (execution_backend == ExecutionBackend::Cuda) {
                    cuda_ok(cudaMemsetAsync(stable_runtime.density, 0, viewer_runtime.field_bytes, stable_runtime.sim_stream), "cudaMemsetAsync stable density");
                    cuda_ok(cudaMemsetAsync(stable_runtime.velocity_x, 0, velocity_x_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream), "cudaMemsetAsync stable velocity_x");
                    cuda_ok(cudaMemsetAsync(stable_runtime.velocity_y, 0, velocity_y_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream), "cudaMemsetAsync stable velocity_y");
                    cuda_ok(cudaMemsetAsync(stable_runtime.velocity_z, 0, velocity_z_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream), "cudaMemsetAsync stable velocity_z");
                } else {
                    std::fill(stable_runtime.host_density.begin(), stable_runtime.host_density.end(), 0.0f);
                    std::fill(stable_runtime.host_velocity_x.begin(), stable_runtime.host_velocity_x.end(), 0.0f);
                    std::fill(stable_runtime.host_velocity_y.begin(), stable_runtime.host_velocity_y.end(), 0.0f);
                    std::fill(stable_runtime.host_velocity_z.begin(), stable_runtime.host_velocity_z.end(), 0.0f);
                }
                if (stable_settings.emit_source) {
                    const float center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx);
                    const float center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny);
                    const float center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz);
                    if (execution_backend == ExecutionBackend::Cuda) {
                        stable_ok(app_add_stable_source_async(stable_runtime.density, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount, stable_settings.velocity_x,
                                      stable_settings.velocity_y, stable_settings.velocity_z, stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z,
                                      stable_runtime.sim_stream),
                            "app_add_stable_source_async");
                    } else {
                        add_stable_source_cpu(stable_runtime.density, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount, stable_settings.velocity_x,
                            stable_settings.velocity_y, stable_settings.velocity_z);
                    }
                    StableFluidsStepDesc step_desc{};
                    step_desc.struct_size                    = sizeof(StableFluidsStepDesc);
                    step_desc.api_version                    = 1;
                    step_desc.nx                             = stable_settings.desc.nx;
                    step_desc.ny                             = stable_settings.desc.ny;
                    step_desc.nz                             = stable_settings.desc.nz;
                    step_desc.cell_size                      = stable_settings.desc.cell_size;
                    step_desc.dt                             = stable_settings.desc.dt;
                    step_desc.viscosity                      = stable_settings.desc.viscosity;
                    step_desc.diffusion                      = stable_settings.desc.diffusion;
                    step_desc.diffuse_iterations             = stable_settings.desc.diffuse_iterations;
                    step_desc.pressure_iterations            = stable_settings.desc.pressure_iterations;
                    step_desc.density                        = stable_runtime.density;
                    step_desc.velocity_x                     = stable_runtime.velocity_x;
                    step_desc.velocity_y                     = stable_runtime.velocity_y;
                    step_desc.velocity_z                     = stable_runtime.velocity_z;
                    step_desc.temporary_density              = stable_runtime.temporary_density;
                    step_desc.temporary_velocity_x           = stable_runtime.temporary_velocity_x;
                    step_desc.temporary_velocity_y           = stable_runtime.temporary_velocity_y;
                    step_desc.temporary_velocity_z           = stable_runtime.temporary_velocity_z;
                    step_desc.temporary_previous_density     = stable_runtime.temporary_previous_density;
                    step_desc.temporary_previous_velocity_x  = stable_runtime.temporary_previous_velocity_x;
                    step_desc.temporary_previous_velocity_y  = stable_runtime.temporary_previous_velocity_y;
                    step_desc.temporary_previous_velocity_z  = stable_runtime.temporary_previous_velocity_z;
                    step_desc.temporary_pressure             = stable_runtime.temporary_pressure;
                    step_desc.temporary_divergence           = stable_runtime.temporary_divergence;
                    step_desc.block_x                        = stable_settings.desc.block_x;
                    step_desc.block_y                        = stable_settings.desc.block_y;
                    step_desc.block_z                        = stable_settings.desc.block_z;
                    step_desc.stream                         = execution_backend == ExecutionBackend::Cuda ? stable_runtime.sim_stream : nullptr;
                    stable_ok(stable_fluids_validate_desc(&step_desc), "stable_fluids_validate_desc");
                    if (execution_backend == ExecutionBackend::Cuda) stable_ok(stable_fluids_step_cuda(&step_desc), "stable_fluids_step_cuda");
                    else if (execution_backend == ExecutionBackend::Parallel) stable_ok(stable_fluids_step_parallel(&step_desc), "stable_fluids_step_parallel");
                    else stable_ok(stable_fluids_step_cpu(&step_desc), "stable_fluids_step_cpu");
                }
            } else {
                if (execution_backend == ExecutionBackend::Cuda) {
                    cuda_ok(cudaMemsetAsync(visual_runtime.density, 0, viewer_runtime.field_bytes, visual_runtime.sim_stream), "cudaMemsetAsync visual density");
                    cuda_ok(cudaMemsetAsync(visual_runtime.temperature, 0, viewer_runtime.field_bytes, visual_runtime.sim_stream), "cudaMemsetAsync visual temperature");
                    cuda_ok(cudaMemsetAsync(visual_runtime.velocity_x, 0, velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.sim_stream), "cudaMemsetAsync visual velocity_x");
                    cuda_ok(cudaMemsetAsync(visual_runtime.velocity_y, 0, velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.sim_stream), "cudaMemsetAsync visual velocity_y");
                    cuda_ok(cudaMemsetAsync(visual_runtime.velocity_z, 0, velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.sim_stream), "cudaMemsetAsync visual velocity_z");
                } else {
                    std::fill(visual_runtime.host_density.begin(), visual_runtime.host_density.end(), 0.0f);
                    std::fill(visual_runtime.host_temperature.begin(), visual_runtime.host_temperature.end(), 0.0f);
                    std::fill(visual_runtime.host_velocity_x.begin(), visual_runtime.host_velocity_x.end(), 0.0f);
                    std::fill(visual_runtime.host_velocity_y.begin(), visual_runtime.host_velocity_y.end(), 0.0f);
                    std::fill(visual_runtime.host_velocity_z.begin(), visual_runtime.host_velocity_z.end(), 0.0f);
                }
                if (visual_settings.emit_source) {
                    const float center_x = visual_settings.source_u * static_cast<float>(visual_settings.desc.nx);
                    const float center_y = visual_settings.source_v * static_cast<float>(visual_settings.desc.ny);
                    const float center_z = visual_settings.source_w * static_cast<float>(visual_settings.desc.nz);
                    if (execution_backend == ExecutionBackend::Cuda) {
                        smoke_ok(app_add_visual_source_async(visual_runtime.density, visual_runtime.temperature, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, center_x, center_y, center_z, visual_settings.source_radius, visual_settings.density_amount,
                                     visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream),
                            "app_add_visual_source_async");
                    } else {
                        add_visual_source_cpu(visual_runtime.density, visual_runtime.temperature, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, center_x, center_y, center_z, visual_settings.source_radius, visual_settings.density_amount,
                            visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z);
                    }
                    VisualSimulationOfSmokeStepDesc step_desc{};
                    step_desc.struct_size                    = sizeof(VisualSimulationOfSmokeStepDesc);
                    step_desc.api_version                    = 1;
                    step_desc.nx                             = visual_settings.desc.nx;
                    step_desc.ny                             = visual_settings.desc.ny;
                    step_desc.nz                             = visual_settings.desc.nz;
                    step_desc.cell_size                      = visual_settings.desc.cell_size;
                    step_desc.dt                             = visual_settings.desc.dt;
                    step_desc.ambient_temperature            = visual_settings.desc.ambient_temperature;
                    step_desc.density_buoyancy               = visual_settings.desc.density_buoyancy;
                    step_desc.temperature_buoyancy           = visual_settings.desc.temperature_buoyancy;
                    step_desc.vorticity_epsilon              = visual_settings.desc.vorticity_epsilon;
                    step_desc.pressure_iterations            = visual_settings.desc.pressure_iterations;
                    step_desc.use_monotonic_cubic            = visual_settings.desc.use_monotonic_cubic;
                    step_desc.density                        = visual_runtime.density;
                    step_desc.temperature                    = visual_runtime.temperature;
                    step_desc.velocity_x                     = visual_runtime.velocity_x;
                    step_desc.velocity_y                     = visual_runtime.velocity_y;
                    step_desc.velocity_z                     = visual_runtime.velocity_z;
                    step_desc.temporary_previous_density     = visual_runtime.temporary_previous_density;
                    step_desc.temporary_previous_temperature = visual_runtime.temporary_previous_temperature;
                    step_desc.temporary_previous_velocity_x  = visual_runtime.temporary_previous_velocity_x;
                    step_desc.temporary_previous_velocity_y  = visual_runtime.temporary_previous_velocity_y;
                    step_desc.temporary_previous_velocity_z  = visual_runtime.temporary_previous_velocity_z;
                    step_desc.temporary_pressure             = visual_runtime.temporary_pressure;
                    step_desc.temporary_divergence           = visual_runtime.temporary_divergence;
                    step_desc.temporary_omega_x              = visual_runtime.temporary_omega_x;
                    step_desc.temporary_omega_y              = visual_runtime.temporary_omega_y;
                    step_desc.temporary_omega_z              = visual_runtime.temporary_omega_z;
                    step_desc.temporary_omega_magnitude      = visual_runtime.temporary_omega_magnitude;
                    step_desc.temporary_force_x              = visual_runtime.temporary_force_x;
                    step_desc.temporary_force_y              = visual_runtime.temporary_force_y;
                    step_desc.temporary_force_z              = visual_runtime.temporary_force_z;
                    step_desc.block_x                        = visual_settings.desc.block_x;
                    step_desc.block_y                        = visual_settings.desc.block_y;
                    step_desc.block_z                        = visual_settings.desc.block_z;
                    step_desc.stream                         = execution_backend == ExecutionBackend::Cuda ? visual_runtime.sim_stream : nullptr;
                    smoke_ok(visual_simulation_of_smoke_validate_desc(&step_desc), "visual_simulation_of_smoke_validate_desc");
                    if (execution_backend == ExecutionBackend::Cuda) smoke_ok(visual_simulation_of_smoke_step_cuda(&step_desc), "visual_simulation_of_smoke_step_cuda");
                    else if (execution_backend == ExecutionBackend::Parallel) smoke_ok(visual_simulation_of_smoke_step_parallel(&step_desc), "visual_simulation_of_smoke_step_parallel");
                    else smoke_ok(visual_simulation_of_smoke_step_cpu(&step_desc), "visual_simulation_of_smoke_step_cpu");
                }
            }

            snapshot_current_field_to_slot(0, "smoke_app.reset_backend.initial_snapshot");
            apply_field_defaults(current_field());
            if (const auto field = active_snapshot()) {
                renderer.frame_volume(*field);
            }
        };

        bool force_snapshot_requested = false;
        reset_backend();

        while (!renderer.should_close()) {
            nvtx3::scoped_range frame_range{"smoke_app.frame"};
            renderer.begin_frame();

            bool reset_requested = false;
            bool field_changed   = false;

            ImGui::Begin("Simulation");
            int backend_index            = static_cast<int>(backend_kind);
            const char* backend_labels[] = {
                "001-stable-fluids",
                "002-visual-simulation-of-smoke",
            };
            if (ImGui::Combo("Backend", &backend_index, backend_labels, 2)) {
                backend_kind    = static_cast<BackendKind>(backend_index);
                reset_requested = true;
                field_changed   = true;
            }
            int execution_backend_index = static_cast<int>(execution_backend);
            const char* execution_backend_labels[] = {
                "CPU",
                "Parallel",
                "CUDA",
            };
            if (ImGui::Combo("Execution", &execution_backend_index, execution_backend_labels, 3)) reset_requested = true;
            execution_backend = static_cast<ExecutionBackend>(execution_backend_index);

            const auto fields    = current_fields();
            auto& selected_field = current_field_index();
            selected_field       = std::clamp(selected_field, 0, static_cast<int>(fields.size()) - 1);
            if (ImGui::BeginCombo("Field", fields[static_cast<size_t>(selected_field)].label.data())) {
                for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
                    const bool is_selected = selected_field == i;
                    if (ImGui::Selectable(fields[static_cast<size_t>(i)].label.data(), is_selected)) {
                        selected_field = i;
                        field_changed  = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("Pause Simulation", &playback.paused);
            if (ImGui::Button("Single Step")) {
                playback.step_once = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Backend")) {
                reset_requested = true;
            }
            ImGui::SliderInt("Sim Steps / Frame", &playback.sim_steps_per_frame, 1, 8);
            ImGui::SliderInt("Snapshot Interval", &playback.snapshot_interval, 1, 8);
            ImGui::Separator();

            if (backend_kind == BackendKind::StableFluids001) {
                ImGui::TextUnformatted("Backend Parameters: 001-stable-fluids");
                if (ImGui::SliderInt("Grid X", &stable_settings.desc.nx, 16, 192)) reset_requested = true;
                if (ImGui::SliderInt("Grid Y", &stable_settings.desc.ny, 16, 192)) reset_requested = true;
                if (ImGui::SliderInt("Grid Z", &stable_settings.desc.nz, 16, 192)) reset_requested = true;
                if (ImGui::SliderFloat("Dt", &stable_settings.desc.dt, 1.0f / 240.0f, 1.0f / 24.0f, "%.5f")) reset_requested = true;
                if (ImGui::SliderFloat("Cell Size", &stable_settings.desc.cell_size, 0.25f, 2.0f, "%.2f")) reset_requested = true;
                if (ImGui::SliderFloat("Viscosity", &stable_settings.desc.viscosity, 0.0f, 0.002f, "%.5f")) reset_requested = true;
                if (ImGui::SliderFloat("Diffusion", &stable_settings.desc.diffusion, 0.0f, 0.002f, "%.5f")) reset_requested = true;
                if (ImGui::SliderInt("Diffuse Iterations", &stable_settings.desc.diffuse_iterations, 1, 64)) reset_requested = true;
                if (ImGui::SliderInt("Pressure Iterations", &stable_settings.desc.pressure_iterations, 4, 192)) reset_requested = true;
                ImGui::Separator();
                ImGui::Checkbox("Emit Source", &stable_settings.emit_source);
                ImGui::SliderFloat("Source U", &stable_settings.source_u, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source V", &stable_settings.source_v, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source W", &stable_settings.source_w, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source Radius", &stable_settings.source_radius, 1.0f, 16.0f, "%.1f");
                ImGui::SliderFloat("Density Amount", &stable_settings.density_amount, 0.0f, 12.0f, "%.2f");
                ImGui::SliderFloat("Velocity X", &stable_settings.velocity_x, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Velocity Y", &stable_settings.velocity_y, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Velocity Z", &stable_settings.velocity_z, -4.0f, 4.0f, "%.2f");
            } else {
                ImGui::TextUnformatted("Backend Parameters: 002-visual-simulation-of-smoke");
                if (ImGui::SliderInt("Grid X", &visual_settings.desc.nx, 16, 192)) reset_requested = true;
                if (ImGui::SliderInt("Grid Y", &visual_settings.desc.ny, 16, 192)) reset_requested = true;
                if (ImGui::SliderInt("Grid Z", &visual_settings.desc.nz, 16, 192)) reset_requested = true;
                if (ImGui::SliderFloat("Dt", &visual_settings.desc.dt, 1.0f / 240.0f, 1.0f / 24.0f, "%.5f")) reset_requested = true;
                if (ImGui::SliderFloat("Cell Size", &visual_settings.desc.cell_size, 0.25f, 2.0f, "%.2f")) reset_requested = true;
                if (ImGui::SliderFloat("Ambient Temp", &visual_settings.desc.ambient_temperature, -2.0f, 2.0f, "%.2f")) reset_requested = true;
                if (ImGui::SliderFloat("Density Buoyancy", &visual_settings.desc.density_buoyancy, 0.0f, 0.2f, "%.4f")) reset_requested = true;
                if (ImGui::SliderFloat("Temperature Buoyancy", &visual_settings.desc.temperature_buoyancy, 0.0f, 0.4f, "%.4f")) reset_requested = true;
                if (ImGui::SliderFloat("Vorticity Epsilon", &visual_settings.desc.vorticity_epsilon, 0.0f, 6.0f, "%.2f")) reset_requested = true;
                if (ImGui::SliderInt("Pressure Iterations", &visual_settings.desc.pressure_iterations, 8, 160)) reset_requested = true;
                bool use_monotonic_cubic = visual_settings.desc.use_monotonic_cubic != 0u;
                if (ImGui::Checkbox("Monotonic Cubic", &use_monotonic_cubic)) {
                    visual_settings.desc.use_monotonic_cubic = use_monotonic_cubic ? 1u : 0u;
                    reset_requested                          = true;
                }
                ImGui::Separator();
                ImGui::Checkbox("Emit Source", &visual_settings.emit_source);
                ImGui::SliderFloat("Source U", &visual_settings.source_u, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source V", &visual_settings.source_v, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source W", &visual_settings.source_w, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source Radius", &visual_settings.source_radius, 1.0f, 16.0f, "%.1f");
                ImGui::SliderFloat("Density Amount", &visual_settings.density_amount, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Temperature Amount", &visual_settings.temperature_amount, 0.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Velocity X", &visual_settings.velocity_x, -3.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Velocity Y", &visual_settings.velocity_y, -3.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Velocity Z", &visual_settings.velocity_z, -3.0f, 3.0f, "%.2f");
            }
            ImGui::End();

            if (field_changed) {
                apply_field_defaults(current_field());
                force_snapshot_requested = true;
            }

            if (reset_requested) {
                reset_backend();
                force_snapshot_requested = false;
            } else {
                const bool run_simulation = !playback.paused || playback.step_once;
                if (run_simulation) {
                    nvtx3::scoped_range sim_range{"smoke_app.simulation"};
                    for (int sim_step = 0; sim_step < playback.sim_steps_per_frame; ++sim_step) {
                        if (backend_kind == BackendKind::StableFluids001) {
                            if (stable_settings.emit_source) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_source"};
                                const float center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx);
                                const float center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny);
                                const float center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz);
                                if (execution_backend == ExecutionBackend::Cuda) {
                                    stable_ok(app_add_stable_source_async(stable_runtime.density, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount,
                                                  stable_settings.velocity_x, stable_settings.velocity_y, stable_settings.velocity_z, stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z, stable_runtime.sim_stream),
                                        "app_add_stable_source_async");
                                } else {
                                    add_stable_source_cpu(stable_runtime.density, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount,
                                        stable_settings.velocity_x, stable_settings.velocity_y, stable_settings.velocity_z);
                                }
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                StableFluidsStepDesc step_desc{};
                                step_desc.struct_size                    = sizeof(StableFluidsStepDesc);
                                step_desc.api_version                    = 1;
                                step_desc.nx                             = stable_settings.desc.nx;
                                step_desc.ny                             = stable_settings.desc.ny;
                                step_desc.nz                             = stable_settings.desc.nz;
                                step_desc.cell_size                      = stable_settings.desc.cell_size;
                                step_desc.dt                             = stable_settings.desc.dt;
                                step_desc.viscosity                      = stable_settings.desc.viscosity;
                                step_desc.diffusion                      = stable_settings.desc.diffusion;
                                step_desc.diffuse_iterations             = stable_settings.desc.diffuse_iterations;
                                step_desc.pressure_iterations            = stable_settings.desc.pressure_iterations;
                                step_desc.density                        = stable_runtime.density;
                                step_desc.velocity_x                     = stable_runtime.velocity_x;
                                step_desc.velocity_y                     = stable_runtime.velocity_y;
                                step_desc.velocity_z                     = stable_runtime.velocity_z;
                                step_desc.temporary_density              = stable_runtime.temporary_density;
                                step_desc.temporary_velocity_x           = stable_runtime.temporary_velocity_x;
                                step_desc.temporary_velocity_y           = stable_runtime.temporary_velocity_y;
                                step_desc.temporary_velocity_z           = stable_runtime.temporary_velocity_z;
                                step_desc.temporary_previous_density     = stable_runtime.temporary_previous_density;
                                step_desc.temporary_previous_velocity_x  = stable_runtime.temporary_previous_velocity_x;
                                step_desc.temporary_previous_velocity_y  = stable_runtime.temporary_previous_velocity_y;
                                step_desc.temporary_previous_velocity_z  = stable_runtime.temporary_previous_velocity_z;
                                step_desc.temporary_pressure             = stable_runtime.temporary_pressure;
                                step_desc.temporary_divergence           = stable_runtime.temporary_divergence;
                                step_desc.block_x                        = stable_settings.desc.block_x;
                                step_desc.block_y                        = stable_settings.desc.block_y;
                                step_desc.block_z                        = stable_settings.desc.block_z;
                                step_desc.stream                         = execution_backend == ExecutionBackend::Cuda ? stable_runtime.sim_stream : nullptr;
                                stable_ok(stable_fluids_validate_desc(&step_desc), "stable_fluids_validate_desc");
                                if (execution_backend == ExecutionBackend::Cuda) stable_ok(stable_fluids_step_cuda(&step_desc), "stable_fluids_step_cuda");
                                else if (execution_backend == ExecutionBackend::Parallel) stable_ok(stable_fluids_step_parallel(&step_desc), "stable_fluids_step_parallel");
                                else stable_ok(stable_fluids_step_cpu(&step_desc), "stable_fluids_step_cpu");
                            }
                        } else {
                            if (visual_settings.emit_source) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_source"};
                                const float center_x = visual_settings.source_u * static_cast<float>(visual_settings.desc.nx);
                                const float center_y = visual_settings.source_v * static_cast<float>(visual_settings.desc.ny);
                                const float center_z = visual_settings.source_w * static_cast<float>(visual_settings.desc.nz);
                                if (execution_backend == ExecutionBackend::Cuda) {
                                    smoke_ok(app_add_visual_source_async(visual_runtime.density, visual_runtime.temperature, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, center_x, center_y, center_z, visual_settings.source_radius,
                                                 visual_settings.density_amount, visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream),
                                        "app_add_visual_source_async");
                                } else {
                                    add_visual_source_cpu(visual_runtime.density, visual_runtime.temperature, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, center_x, center_y, center_z, visual_settings.source_radius,
                                        visual_settings.density_amount, visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z);
                                }
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                VisualSimulationOfSmokeStepDesc step_desc{};
                                step_desc.struct_size                    = sizeof(VisualSimulationOfSmokeStepDesc);
                                step_desc.api_version                    = 1;
                                step_desc.nx                             = visual_settings.desc.nx;
                                step_desc.ny                             = visual_settings.desc.ny;
                                step_desc.nz                             = visual_settings.desc.nz;
                                step_desc.cell_size                      = visual_settings.desc.cell_size;
                                step_desc.dt                             = visual_settings.desc.dt;
                                step_desc.ambient_temperature            = visual_settings.desc.ambient_temperature;
                                step_desc.density_buoyancy               = visual_settings.desc.density_buoyancy;
                                step_desc.temperature_buoyancy           = visual_settings.desc.temperature_buoyancy;
                                step_desc.vorticity_epsilon              = visual_settings.desc.vorticity_epsilon;
                                step_desc.pressure_iterations            = visual_settings.desc.pressure_iterations;
                                step_desc.use_monotonic_cubic            = visual_settings.desc.use_monotonic_cubic;
                                step_desc.density                        = visual_runtime.density;
                                step_desc.temperature                    = visual_runtime.temperature;
                                step_desc.velocity_x                     = visual_runtime.velocity_x;
                                step_desc.velocity_y                     = visual_runtime.velocity_y;
                                step_desc.velocity_z                     = visual_runtime.velocity_z;
                                step_desc.temporary_previous_density     = visual_runtime.temporary_previous_density;
                                step_desc.temporary_previous_temperature = visual_runtime.temporary_previous_temperature;
                                step_desc.temporary_previous_velocity_x  = visual_runtime.temporary_previous_velocity_x;
                                step_desc.temporary_previous_velocity_y  = visual_runtime.temporary_previous_velocity_y;
                                step_desc.temporary_previous_velocity_z  = visual_runtime.temporary_previous_velocity_z;
                                step_desc.temporary_pressure             = visual_runtime.temporary_pressure;
                                step_desc.temporary_divergence           = visual_runtime.temporary_divergence;
                                step_desc.temporary_omega_x              = visual_runtime.temporary_omega_x;
                                step_desc.temporary_omega_y              = visual_runtime.temporary_omega_y;
                                step_desc.temporary_omega_z              = visual_runtime.temporary_omega_z;
                                step_desc.temporary_omega_magnitude      = visual_runtime.temporary_omega_magnitude;
                                step_desc.temporary_force_x              = visual_runtime.temporary_force_x;
                                step_desc.temporary_force_y              = visual_runtime.temporary_force_y;
                                step_desc.temporary_force_z              = visual_runtime.temporary_force_z;
                                step_desc.block_x                        = visual_settings.desc.block_x;
                                step_desc.block_y                        = visual_settings.desc.block_y;
                                step_desc.block_z                        = visual_settings.desc.block_z;
                                step_desc.stream                         = execution_backend == ExecutionBackend::Cuda ? visual_runtime.sim_stream : nullptr;
                                smoke_ok(visual_simulation_of_smoke_validate_desc(&step_desc), "visual_simulation_of_smoke_validate_desc");
                                if (execution_backend == ExecutionBackend::Cuda) smoke_ok(visual_simulation_of_smoke_step_cuda(&step_desc), "visual_simulation_of_smoke_step_cuda");
                                else if (execution_backend == ExecutionBackend::Parallel) smoke_ok(visual_simulation_of_smoke_step_parallel(&step_desc), "visual_simulation_of_smoke_step_parallel");
                                else smoke_ok(visual_simulation_of_smoke_step_cpu(&step_desc), "visual_simulation_of_smoke_step_cpu");
                            }
                        }

                        if (viewer_runtime.steps_since_snapshot < static_cast<uint32_t>(playback.snapshot_interval)) {
                            ++viewer_runtime.steps_since_snapshot;
                        }
                        if (viewer_runtime.steps_since_snapshot >= static_cast<uint32_t>(playback.snapshot_interval)) {
                            const int slot_index = find_available_snapshot_slot();
                            if (slot_index >= 0) {
                                snapshot_current_field_to_slot(slot_index, "smoke_app.simulation.snapshot");
                                force_snapshot_requested = false;
                            }
                        }
                    }
                }
            }

            playback.step_once = false;

            if (force_snapshot_requested) {
                const int slot_index = find_available_snapshot_slot();
                if (slot_index >= 0) {
                    snapshot_current_field_to_slot(slot_index, "smoke_app.field_change_snapshot");
                    force_snapshot_requested = false;
                }
            }

            const auto field = active_snapshot();
            renderer.draw_renderer_ui(field);

            bool submitted = false;
            {
                nvtx3::scoped_range render_range{"smoke_app.render"};
                submitted = renderer.render_frame(field);
            }
            if (submitted) {
                const uint64_t next_submit_serial = viewer_runtime.submit_serial + 1;
                if (viewer_runtime.active_snapshot_slot >= 0) {
                    snapshot_slots[static_cast<size_t>(viewer_runtime.active_snapshot_slot)].last_used_submit_serial = next_submit_serial;
                }
                viewer_runtime.submit_serial = next_submit_serial;
            }
        }

        destroy_everything();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
