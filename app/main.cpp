#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "stable-fluids.h"
#include "visual-simulation-of-smoke.h"
#include <cuda_runtime.h>
#include <imgui.h>

#include <nvtx3/nvtx3.hpp>
#include <vulkan/vulkan_raii.hpp>

import app;
import std;
import vk.memory;

int32_t app_compute_staggered_velocity_magnitude_async(void* destination, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream);
int32_t app_add_stable_source_async(
    void* density, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float density_amount, float velocity_source_x, float velocity_source_y, float velocity_source_z, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream);
int32_t app_add_visual_source_async(void* density, void* temperature, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float density_amount, float temperature_amount, float velocity_source_x, float velocity_source_y, float velocity_source_z,
    int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream);

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    enum class BackendKind : uint32_t {
        StableFluids001 = 0,
        VisualSmoke002  = 1,
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

    constexpr std::array stable_boundary_labels{
        "Fixed",
        "Periodic",
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

    struct SolverStats {
        double last_step_call_ms    = 0.0;
        double average_step_call_ms = 0.0;
        double last_snapshot_ms     = 0.0;
        double average_snapshot_ms  = 0.0;
        uint64_t step_count         = 0;
        uint64_t snapshot_count     = 0;
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
        uint32_t nx                                = 0;
        uint32_t ny                                = 0;
        uint32_t nz                                = 0;
        float cell_size                            = 1.0f;
        app::FieldSemantic semantic                = app::FieldSemantic::GenericScalar;
        std::string_view label{};
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
        uint32_t boundary_x_min     = STABLE_FLUIDS_BOUNDARY_FIXED;
        uint32_t boundary_x_max     = STABLE_FLUIDS_BOUNDARY_FIXED;
        uint32_t boundary_y_min     = STABLE_FLUIDS_BOUNDARY_FIXED;
        uint32_t boundary_y_max     = STABLE_FLUIDS_BOUNDARY_FIXED;
        uint32_t boundary_z_min     = STABLE_FLUIDS_BOUNDARY_FIXED;
        uint32_t boundary_z_max     = STABLE_FLUIDS_BOUNDARY_FIXED;
        int32_t block_x             = 8;
        int32_t block_y             = 8;
        int32_t block_z             = 4;
    };

    struct StableSettings {
        StableStepConfig desc{};
        int selected_field   = 0;
        bool emit_source     = true;
        float source_u       = 0.5f;
        float source_v       = 0.18f;
        float source_w       = 0.5f;
        float source_radius  = 5.0f;
        float density_amount = 0.85f;
        float velocity_x     = 0.0f;
        float velocity_y     = 1.2f;
        float velocity_z     = 0.0f;
    };

    struct StableRuntime {
        bool device_allocated                = false;
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
        bool device_allocated                 = false;
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
        StableSettings stable_settings{};
        StableRuntime stable_runtime{};
        SolverStats stable_solver_stats{};
        VisualSettings visual_settings{};
        VisualRuntime visual_runtime{};
        SolverStats visual_solver_stats{};
        ViewerRuntime viewer_runtime{};
        std::vector<SnapshotSlot> snapshot_slots{};

        auto scalar_bytes     = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_x_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_y_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_z_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz + 1) * sizeof(float); };
        auto current_fields   = [&]() -> std::span<const FieldChoice> { return backend_kind == BackendKind::StableFluids001 ? std::span<const FieldChoice>(stable_fields) : std::span<const FieldChoice>(visual_fields); };

        auto current_field_index = [&]() -> int& { return backend_kind == BackendKind::StableFluids001 ? stable_settings.selected_field : visual_settings.selected_field; };

        auto current_field = [&]() -> const FieldChoice& {
            auto fields    = current_fields();
            auto& selected = current_field_index();
            selected       = std::clamp(selected, 0, static_cast<int>(fields.size()) - 1);
            return fields[static_cast<size_t>(selected)];
        };

        auto current_stream = [&]() -> cudaStream_t { return backend_kind == BackendKind::StableFluids001 ? stable_runtime.sim_stream : visual_runtime.sim_stream; };

        auto current_solver_stats = [&]() -> SolverStats& { return backend_kind == BackendKind::StableFluids001 ? stable_solver_stats : visual_solver_stats; };
        auto accumulate_sample    = [](const double sample_ms, double& last_ms, double& average_ms, uint64_t& count) {
            last_ms = sample_ms;
            ++count;
            average_ms += (sample_ms - average_ms) / static_cast<double>(count);
        };

        auto run_stable_step = [&]() {
            StableFluidsAdvectVelocityDesc advect_velocity_desc{};
            advect_velocity_desc.struct_size = sizeof(StableFluidsAdvectVelocityDesc);
            advect_velocity_desc.api_version = STABLE_FLUIDS_API_VERSION;
            advect_velocity_desc.nx = stable_settings.desc.nx;
            advect_velocity_desc.ny = stable_settings.desc.ny;
            advect_velocity_desc.nz = stable_settings.desc.nz;
            advect_velocity_desc.cell_size = stable_settings.desc.cell_size;
            advect_velocity_desc.dt = stable_settings.desc.dt;
            advect_velocity_desc.boundary_x_min = stable_settings.desc.boundary_x_min;
            advect_velocity_desc.boundary_x_max = stable_settings.desc.boundary_x_max;
            advect_velocity_desc.boundary_y_min = stable_settings.desc.boundary_y_min;
            advect_velocity_desc.boundary_y_max = stable_settings.desc.boundary_y_max;
            advect_velocity_desc.boundary_z_min = stable_settings.desc.boundary_z_min;
            advect_velocity_desc.boundary_z_max = stable_settings.desc.boundary_z_max;
            advect_velocity_desc.velocity_x = stable_runtime.velocity_x;
            advect_velocity_desc.velocity_y = stable_runtime.velocity_y;
            advect_velocity_desc.velocity_z = stable_runtime.velocity_z;
            advect_velocity_desc.temporary_velocity_x = stable_runtime.temporary_velocity_x;
            advect_velocity_desc.temporary_velocity_y = stable_runtime.temporary_velocity_y;
            advect_velocity_desc.temporary_velocity_z = stable_runtime.temporary_velocity_z;
            advect_velocity_desc.temporary_previous_velocity_x = stable_runtime.temporary_previous_velocity_x;
            advect_velocity_desc.temporary_previous_velocity_y = stable_runtime.temporary_previous_velocity_y;
            advect_velocity_desc.temporary_previous_velocity_z = stable_runtime.temporary_previous_velocity_z;
            advect_velocity_desc.block_x = stable_settings.desc.block_x;
            advect_velocity_desc.block_y = stable_settings.desc.block_y;
            advect_velocity_desc.block_z = stable_settings.desc.block_z;
            advect_velocity_desc.stream = stable_runtime.sim_stream;

            StableFluidsDiffuseVelocityDesc diffuse_velocity_desc{};
            diffuse_velocity_desc.struct_size = sizeof(StableFluidsDiffuseVelocityDesc);
            diffuse_velocity_desc.api_version = STABLE_FLUIDS_API_VERSION;
            diffuse_velocity_desc.nx = stable_settings.desc.nx;
            diffuse_velocity_desc.ny = stable_settings.desc.ny;
            diffuse_velocity_desc.nz = stable_settings.desc.nz;
            diffuse_velocity_desc.cell_size = stable_settings.desc.cell_size;
            diffuse_velocity_desc.dt = stable_settings.desc.dt;
            diffuse_velocity_desc.viscosity = stable_settings.desc.viscosity;
            diffuse_velocity_desc.diffuse_iterations = stable_settings.desc.diffuse_iterations;
            diffuse_velocity_desc.boundary_x_min = stable_settings.desc.boundary_x_min;
            diffuse_velocity_desc.boundary_x_max = stable_settings.desc.boundary_x_max;
            diffuse_velocity_desc.boundary_y_min = stable_settings.desc.boundary_y_min;
            diffuse_velocity_desc.boundary_y_max = stable_settings.desc.boundary_y_max;
            diffuse_velocity_desc.boundary_z_min = stable_settings.desc.boundary_z_min;
            diffuse_velocity_desc.boundary_z_max = stable_settings.desc.boundary_z_max;
            diffuse_velocity_desc.velocity_x = stable_runtime.velocity_x;
            diffuse_velocity_desc.velocity_y = stable_runtime.velocity_y;
            diffuse_velocity_desc.velocity_z = stable_runtime.velocity_z;
            diffuse_velocity_desc.temporary_velocity_x = stable_runtime.temporary_velocity_x;
            diffuse_velocity_desc.temporary_velocity_y = stable_runtime.temporary_velocity_y;
            diffuse_velocity_desc.temporary_velocity_z = stable_runtime.temporary_velocity_z;
            diffuse_velocity_desc.temporary_density = stable_runtime.temporary_density;
            diffuse_velocity_desc.temporary_previous_density = stable_runtime.temporary_previous_density;
            diffuse_velocity_desc.block_x = stable_settings.desc.block_x;
            diffuse_velocity_desc.block_y = stable_settings.desc.block_y;
            diffuse_velocity_desc.block_z = stable_settings.desc.block_z;
            diffuse_velocity_desc.stream = stable_runtime.sim_stream;

            StableFluidsProjectDesc project_desc{};
            project_desc.struct_size = sizeof(StableFluidsProjectDesc);
            project_desc.api_version = STABLE_FLUIDS_API_VERSION;
            project_desc.nx = stable_settings.desc.nx;
            project_desc.ny = stable_settings.desc.ny;
            project_desc.nz = stable_settings.desc.nz;
            project_desc.cell_size = stable_settings.desc.cell_size;
            project_desc.pressure_iterations = stable_settings.desc.pressure_iterations;
            project_desc.boundary_x_min = stable_settings.desc.boundary_x_min;
            project_desc.boundary_x_max = stable_settings.desc.boundary_x_max;
            project_desc.boundary_y_min = stable_settings.desc.boundary_y_min;
            project_desc.boundary_y_max = stable_settings.desc.boundary_y_max;
            project_desc.boundary_z_min = stable_settings.desc.boundary_z_min;
            project_desc.boundary_z_max = stable_settings.desc.boundary_z_max;
            project_desc.velocity_x = stable_runtime.velocity_x;
            project_desc.velocity_y = stable_runtime.velocity_y;
            project_desc.velocity_z = stable_runtime.velocity_z;
            project_desc.temporary_pressure = stable_runtime.temporary_pressure;
            project_desc.temporary_divergence = stable_runtime.temporary_divergence;
            project_desc.temporary_density = stable_runtime.temporary_density;
            project_desc.temporary_previous_density = stable_runtime.temporary_previous_density;
            project_desc.block_x = stable_settings.desc.block_x;
            project_desc.block_y = stable_settings.desc.block_y;
            project_desc.block_z = stable_settings.desc.block_z;
            project_desc.stream = stable_runtime.sim_stream;

            StableFluidsAdvectScalarDesc advect_scalar_desc{};
            advect_scalar_desc.struct_size = sizeof(StableFluidsAdvectScalarDesc);
            advect_scalar_desc.api_version = STABLE_FLUIDS_API_VERSION;
            advect_scalar_desc.nx = stable_settings.desc.nx;
            advect_scalar_desc.ny = stable_settings.desc.ny;
            advect_scalar_desc.nz = stable_settings.desc.nz;
            advect_scalar_desc.cell_size = stable_settings.desc.cell_size;
            advect_scalar_desc.dt = stable_settings.desc.dt;
            advect_scalar_desc.boundary_x_min = stable_settings.desc.boundary_x_min;
            advect_scalar_desc.boundary_x_max = stable_settings.desc.boundary_x_max;
            advect_scalar_desc.boundary_y_min = stable_settings.desc.boundary_y_min;
            advect_scalar_desc.boundary_y_max = stable_settings.desc.boundary_y_max;
            advect_scalar_desc.boundary_z_min = stable_settings.desc.boundary_z_min;
            advect_scalar_desc.boundary_z_max = stable_settings.desc.boundary_z_max;
            advect_scalar_desc.scalar = stable_runtime.density;
            advect_scalar_desc.temporary_scalar = stable_runtime.temporary_density;
            advect_scalar_desc.temporary_previous_scalar = stable_runtime.temporary_previous_density;
            advect_scalar_desc.velocity_x = stable_runtime.velocity_x;
            advect_scalar_desc.velocity_y = stable_runtime.velocity_y;
            advect_scalar_desc.velocity_z = stable_runtime.velocity_z;
            advect_scalar_desc.clamp_non_negative = 1u;
            advect_scalar_desc.block_x = stable_settings.desc.block_x;
            advect_scalar_desc.block_y = stable_settings.desc.block_y;
            advect_scalar_desc.block_z = stable_settings.desc.block_z;
            advect_scalar_desc.stream = stable_runtime.sim_stream;

            StableFluidsDiffuseScalarDesc diffuse_scalar_desc{};
            diffuse_scalar_desc.struct_size = sizeof(StableFluidsDiffuseScalarDesc);
            diffuse_scalar_desc.api_version = STABLE_FLUIDS_API_VERSION;
            diffuse_scalar_desc.nx = stable_settings.desc.nx;
            diffuse_scalar_desc.ny = stable_settings.desc.ny;
            diffuse_scalar_desc.nz = stable_settings.desc.nz;
            diffuse_scalar_desc.cell_size = stable_settings.desc.cell_size;
            diffuse_scalar_desc.dt = stable_settings.desc.dt;
            diffuse_scalar_desc.diffusion = stable_settings.desc.diffusion;
            diffuse_scalar_desc.diffuse_iterations = stable_settings.desc.diffuse_iterations;
            diffuse_scalar_desc.boundary_x_min = stable_settings.desc.boundary_x_min;
            diffuse_scalar_desc.boundary_x_max = stable_settings.desc.boundary_x_max;
            diffuse_scalar_desc.boundary_y_min = stable_settings.desc.boundary_y_min;
            diffuse_scalar_desc.boundary_y_max = stable_settings.desc.boundary_y_max;
            diffuse_scalar_desc.boundary_z_min = stable_settings.desc.boundary_z_min;
            diffuse_scalar_desc.boundary_z_max = stable_settings.desc.boundary_z_max;
            diffuse_scalar_desc.scalar = stable_runtime.density;
            diffuse_scalar_desc.temporary_scalar = stable_runtime.temporary_density;
            diffuse_scalar_desc.temporary_solution_storage = stable_runtime.temporary_pressure;
            diffuse_scalar_desc.temporary_rhs_storage = stable_runtime.temporary_divergence;
            diffuse_scalar_desc.clamp_non_negative = 1u;
            diffuse_scalar_desc.block_x = stable_settings.desc.block_x;
            diffuse_scalar_desc.block_y = stable_settings.desc.block_y;
            diffuse_scalar_desc.block_z = stable_settings.desc.block_z;
            diffuse_scalar_desc.stream = stable_runtime.sim_stream;

            stable_ok(stable_fluids_advect_velocity_cuda(&advect_velocity_desc), "stable_fluids_advect_velocity_cuda");
            stable_ok(stable_fluids_diffuse_velocity_cuda(&diffuse_velocity_desc), "stable_fluids_diffuse_velocity_cuda");
            stable_ok(stable_fluids_project_cuda(&project_desc), "stable_fluids_project_cuda");
            stable_ok(stable_fluids_advect_scalar_cuda(&advect_scalar_desc), "stable_fluids_advect_scalar_cuda");
            stable_ok(stable_fluids_diffuse_scalar_cuda(&diffuse_scalar_desc), "stable_fluids_diffuse_scalar_cuda");
        };

        auto run_visual_step = [&]() {
            VisualSimulationOfSmokeForcesDesc forces_desc{};
            forces_desc.struct_size = sizeof(VisualSimulationOfSmokeForcesDesc);
            forces_desc.api_version = VISUAL_SIMULATION_OF_SMOKE_API_VERSION;
            forces_desc.nx = visual_settings.desc.nx;
            forces_desc.ny = visual_settings.desc.ny;
            forces_desc.nz = visual_settings.desc.nz;
            forces_desc.cell_size = visual_settings.desc.cell_size;
            forces_desc.dt = visual_settings.desc.dt;
            forces_desc.ambient_temperature = visual_settings.desc.ambient_temperature;
            forces_desc.density_buoyancy = visual_settings.desc.density_buoyancy;
            forces_desc.temperature_buoyancy = visual_settings.desc.temperature_buoyancy;
            forces_desc.vorticity_epsilon = visual_settings.desc.vorticity_epsilon;
            forces_desc.density = visual_runtime.density;
            forces_desc.temperature = visual_runtime.temperature;
            forces_desc.velocity_x = visual_runtime.velocity_x;
            forces_desc.velocity_y = visual_runtime.velocity_y;
            forces_desc.velocity_z = visual_runtime.velocity_z;
            forces_desc.temporary_omega_x = visual_runtime.temporary_omega_x;
            forces_desc.temporary_omega_y = visual_runtime.temporary_omega_y;
            forces_desc.temporary_omega_z = visual_runtime.temporary_omega_z;
            forces_desc.temporary_omega_magnitude = visual_runtime.temporary_omega_magnitude;
            forces_desc.temporary_force_x = visual_runtime.temporary_force_x;
            forces_desc.temporary_force_y = visual_runtime.temporary_force_y;
            forces_desc.temporary_force_z = visual_runtime.temporary_force_z;
            forces_desc.block_x = visual_settings.desc.block_x;
            forces_desc.block_y = visual_settings.desc.block_y;
            forces_desc.block_z = visual_settings.desc.block_z;
            forces_desc.stream = visual_runtime.sim_stream;

            VisualSimulationOfSmokeAdvectVelocityDesc advect_velocity_desc{};
            advect_velocity_desc.struct_size = sizeof(VisualSimulationOfSmokeAdvectVelocityDesc);
            advect_velocity_desc.api_version = VISUAL_SIMULATION_OF_SMOKE_API_VERSION;
            advect_velocity_desc.nx = visual_settings.desc.nx;
            advect_velocity_desc.ny = visual_settings.desc.ny;
            advect_velocity_desc.nz = visual_settings.desc.nz;
            advect_velocity_desc.cell_size = visual_settings.desc.cell_size;
            advect_velocity_desc.dt = visual_settings.desc.dt;
            advect_velocity_desc.use_monotonic_cubic = visual_settings.desc.use_monotonic_cubic;
            advect_velocity_desc.velocity_x = visual_runtime.velocity_x;
            advect_velocity_desc.velocity_y = visual_runtime.velocity_y;
            advect_velocity_desc.velocity_z = visual_runtime.velocity_z;
            advect_velocity_desc.temporary_previous_velocity_x = visual_runtime.temporary_previous_velocity_x;
            advect_velocity_desc.temporary_previous_velocity_y = visual_runtime.temporary_previous_velocity_y;
            advect_velocity_desc.temporary_previous_velocity_z = visual_runtime.temporary_previous_velocity_z;
            advect_velocity_desc.block_x = visual_settings.desc.block_x;
            advect_velocity_desc.block_y = visual_settings.desc.block_y;
            advect_velocity_desc.block_z = visual_settings.desc.block_z;
            advect_velocity_desc.stream = visual_runtime.sim_stream;

            VisualSimulationOfSmokeProjectDesc project_desc{};
            project_desc.struct_size = sizeof(VisualSimulationOfSmokeProjectDesc);
            project_desc.api_version = VISUAL_SIMULATION_OF_SMOKE_API_VERSION;
            project_desc.nx = visual_settings.desc.nx;
            project_desc.ny = visual_settings.desc.ny;
            project_desc.nz = visual_settings.desc.nz;
            project_desc.cell_size = visual_settings.desc.cell_size;
            project_desc.dt = visual_settings.desc.dt;
            project_desc.pressure_iterations = visual_settings.desc.pressure_iterations;
            project_desc.temporary_previous_velocity_x = visual_runtime.temporary_previous_velocity_x;
            project_desc.temporary_previous_velocity_y = visual_runtime.temporary_previous_velocity_y;
            project_desc.temporary_previous_velocity_z = visual_runtime.temporary_previous_velocity_z;
            project_desc.temporary_pressure = visual_runtime.temporary_pressure;
            project_desc.temporary_divergence = visual_runtime.temporary_divergence;
            project_desc.temporary_omega_x = visual_runtime.temporary_omega_x;
            project_desc.temporary_omega_y = visual_runtime.temporary_omega_y;
            project_desc.block_x = visual_settings.desc.block_x;
            project_desc.block_y = visual_settings.desc.block_y;
            project_desc.block_z = visual_settings.desc.block_z;
            project_desc.stream = visual_runtime.sim_stream;

            VisualSimulationOfSmokeScalarFlowBinding scalar_bindings[2] = {
                VisualSimulationOfSmokeScalarFlowBinding{.scalar = visual_runtime.density, .temporary_previous_scalar = visual_runtime.temporary_previous_density, .clamp_non_negative = 1u},
                VisualSimulationOfSmokeScalarFlowBinding{.scalar = visual_runtime.temperature, .temporary_previous_scalar = visual_runtime.temporary_previous_temperature, .clamp_non_negative = 0u},
            };
            VisualSimulationOfSmokeAdvectScalarFlowDesc scalar_flow_desc{};
            scalar_flow_desc.struct_size = sizeof(VisualSimulationOfSmokeAdvectScalarFlowDesc);
            scalar_flow_desc.api_version = VISUAL_SIMULATION_OF_SMOKE_API_VERSION;
            scalar_flow_desc.nx = visual_settings.desc.nx;
            scalar_flow_desc.ny = visual_settings.desc.ny;
            scalar_flow_desc.nz = visual_settings.desc.nz;
            scalar_flow_desc.cell_size = visual_settings.desc.cell_size;
            scalar_flow_desc.dt = visual_settings.desc.dt;
            scalar_flow_desc.use_monotonic_cubic = visual_settings.desc.use_monotonic_cubic;
            scalar_flow_desc.scalar_bindings = scalar_bindings;
            scalar_flow_desc.scalar_count = 2;
            scalar_flow_desc.velocity_x = visual_runtime.velocity_x;
            scalar_flow_desc.velocity_y = visual_runtime.velocity_y;
            scalar_flow_desc.velocity_z = visual_runtime.velocity_z;
            scalar_flow_desc.block_x = visual_settings.desc.block_x;
            scalar_flow_desc.block_y = visual_settings.desc.block_y;
            scalar_flow_desc.block_z = visual_settings.desc.block_z;
            scalar_flow_desc.stream = visual_runtime.sim_stream;

            smoke_ok(visual_simulation_of_smoke_forces_cuda(&forces_desc), "visual_simulation_of_smoke_forces_cuda");
            smoke_ok(visual_simulation_of_smoke_advect_velocity_cuda(&advect_velocity_desc), "visual_simulation_of_smoke_advect_velocity_cuda");
            smoke_ok(visual_simulation_of_smoke_project_cuda(&project_desc), "visual_simulation_of_smoke_project_cuda");
            cuda_ok(cudaMemcpyAsync(visual_runtime.velocity_x, visual_runtime.temporary_previous_velocity_x, velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual velocity_x");
            cuda_ok(cudaMemcpyAsync(visual_runtime.velocity_y, visual_runtime.temporary_previous_velocity_y, velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual velocity_y");
            cuda_ok(cudaMemcpyAsync(visual_runtime.velocity_z, visual_runtime.temporary_previous_velocity_z, velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual velocity_z");
            smoke_ok(visual_simulation_of_smoke_advect_scalar_flow_cuda(&scalar_flow_desc), "visual_simulation_of_smoke_advect_scalar_flow_cuda");
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

            const auto& slot = snapshot_slots.at(static_cast<size_t>(viewer_runtime.active_snapshot_slot));
            return app::ScalarFieldView{
                .descriptor_set     = *slot.descriptor_set,
                .timeline_semaphore = slot.external_semaphore != nullptr ? *slot.timeline_semaphore : vk::Semaphore{},
                .ready_generation   = slot.ready_generation,
                .nx                 = slot.nx,
                .ny                 = slot.ny,
                .nz                 = slot.nz,
                .cell_size          = slot.cell_size,
                .semantic           = slot.semantic,
                .label              = slot.label,
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
                slot.nx                      = 0;
                slot.ny                      = 0;
                slot.nz                      = 0;
                slot.cell_size               = 1.0f;
                slot.semantic                = app::FieldSemantic::GenericScalar;
                slot.label                   = {};
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
            auto release = [&](float*& ptr) {
                if (stable_runtime.device_allocated && ptr != nullptr) cudaFree(ptr);
                ptr = nullptr;
            };
            release(stable_runtime.density);
            release(stable_runtime.velocity_x);
            release(stable_runtime.velocity_y);
            release(stable_runtime.velocity_z);
            release(stable_runtime.temporary_density);
            release(stable_runtime.temporary_velocity_x);
            release(stable_runtime.temporary_velocity_y);
            release(stable_runtime.temporary_velocity_z);
            release(stable_runtime.temporary_previous_density);
            release(stable_runtime.temporary_previous_velocity_x);
            release(stable_runtime.temporary_previous_velocity_y);
            release(stable_runtime.temporary_previous_velocity_z);
            release(stable_runtime.temporary_pressure);
            release(stable_runtime.temporary_divergence);
            if (stable_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(stable_runtime.sim_stream);
                stable_runtime.sim_stream = nullptr;
            }
            stable_runtime.device_allocated              = false;
            stable_runtime.density                       = nullptr;
            stable_runtime.velocity_x                    = nullptr;
            stable_runtime.velocity_y                    = nullptr;
            stable_runtime.velocity_z                    = nullptr;
            stable_runtime.temporary_density             = nullptr;
            stable_runtime.temporary_velocity_x          = nullptr;
            stable_runtime.temporary_velocity_y          = nullptr;
            stable_runtime.temporary_velocity_z          = nullptr;
            stable_runtime.temporary_previous_density    = nullptr;
            stable_runtime.temporary_previous_velocity_x = nullptr;
            stable_runtime.temporary_previous_velocity_y = nullptr;
            stable_runtime.temporary_previous_velocity_z = nullptr;
            stable_runtime.temporary_pressure            = nullptr;
            stable_runtime.temporary_divergence          = nullptr;
        };

        auto destroy_visual_backend = [&]() {
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamSynchronize(visual_runtime.sim_stream);
            }
            auto release = [&](float*& ptr) {
                if (visual_runtime.device_allocated && ptr != nullptr) cudaFree(ptr);
                ptr = nullptr;
            };
            release(visual_runtime.density);
            release(visual_runtime.temperature);
            release(visual_runtime.velocity_x);
            release(visual_runtime.velocity_y);
            release(visual_runtime.velocity_z);
            release(visual_runtime.temporary_previous_density);
            release(visual_runtime.temporary_previous_temperature);
            release(visual_runtime.temporary_previous_velocity_x);
            release(visual_runtime.temporary_previous_velocity_y);
            release(visual_runtime.temporary_previous_velocity_z);
            release(visual_runtime.temporary_pressure);
            release(visual_runtime.temporary_divergence);
            release(visual_runtime.temporary_omega_x);
            release(visual_runtime.temporary_omega_y);
            release(visual_runtime.temporary_omega_z);
            release(visual_runtime.temporary_omega_magnitude);
            release(visual_runtime.temporary_force_x);
            release(visual_runtime.temporary_force_y);
            release(visual_runtime.temporary_force_z);
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(visual_runtime.sim_stream);
                visual_runtime.sim_stream = nullptr;
            }
            visual_runtime.device_allocated               = false;
            visual_runtime.density                        = nullptr;
            visual_runtime.temperature                    = nullptr;
            visual_runtime.velocity_x                     = nullptr;
            visual_runtime.velocity_y                     = nullptr;
            visual_runtime.velocity_z                     = nullptr;
            visual_runtime.temporary_previous_density     = nullptr;
            visual_runtime.temporary_previous_temperature = nullptr;
            visual_runtime.temporary_previous_velocity_x  = nullptr;
            visual_runtime.temporary_previous_velocity_y  = nullptr;
            visual_runtime.temporary_previous_velocity_z  = nullptr;
            visual_runtime.temporary_pressure             = nullptr;
            visual_runtime.temporary_divergence           = nullptr;
            visual_runtime.temporary_omega_x              = nullptr;
            visual_runtime.temporary_omega_y              = nullptr;
            visual_runtime.temporary_omega_z              = nullptr;
            visual_runtime.temporary_omega_magnitude      = nullptr;
            visual_runtime.temporary_force_x              = nullptr;
            visual_runtime.temporary_force_y              = nullptr;
            visual_runtime.temporary_force_z              = nullptr;
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
#if defined(_WIN32)
                constexpr auto memory_handle_type    = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
                constexpr auto semaphore_handle_type = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
#else
                constexpr auto memory_handle_type    = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
                constexpr auto semaphore_handle_type = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
#endif
                vk::SemaphoreTypeCreateInfo timeline_semaphore_ci{
                    .semaphoreType = vk::SemaphoreType::eTimeline,
                    .initialValue  = 0,
                };
                vk::ExportSemaphoreCreateInfo export_semaphore_ci{
                    .pNext       = &timeline_semaphore_ci,
                    .handleTypes = semaphore_handle_type,
                };
                vk::SemaphoreCreateInfo semaphore_ci{
                    .pNext = &export_semaphore_ci,
                };
                slot.timeline_semaphore = vk::raii::Semaphore{renderer.vk_context().device, semaphore_ci};

                vk::ExternalMemoryBufferCreateInfo external_buffer_ci{
                    .handleTypes = memory_handle_type,
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
                    .handleTypes = memory_handle_type,
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
                    .handleType = memory_handle_type,
                };
                HANDLE memory_handle = renderer.vk_context().device.getMemoryWin32HandleKHR(handle_info);
                if (memory_handle == nullptr) throw std::runtime_error("getMemoryWin32HandleKHR returned a null handle");

                cudaExternalMemoryHandleDesc external_desc{};
                external_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
                external_desc.handle.win32.handle = memory_handle;
                external_desc.size                = requirements.size;
                cuda_ok(cudaImportExternalMemory(&slot.external_memory, &external_desc), "cudaImportExternalMemory");
                CloseHandle(memory_handle);

                vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                    .semaphore  = *slot.timeline_semaphore,
                    .handleType = semaphore_handle_type,
                };
                HANDLE semaphore_handle = renderer.vk_context().device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
                if (semaphore_handle == nullptr) throw std::runtime_error("getSemaphoreWin32HandleKHR returned a null handle");

                cudaExternalSemaphoreHandleDesc external_semaphore_desc{};
                external_semaphore_desc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
                external_semaphore_desc.handle.win32.handle = semaphore_handle;
                cuda_ok(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
                CloseHandle(semaphore_handle);
#else
                vk::MemoryGetFdInfoKHR handle_info{
                    .memory     = *slot.memory,
                    .handleType = memory_handle_type,
                };
                const int memory_fd = renderer.vk_context().device.getMemoryFdKHR(handle_info);
                if (memory_fd < 0) throw std::runtime_error("getMemoryFdKHR returned an invalid fd");

                cudaExternalMemoryHandleDesc external_desc{};
                external_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
                external_desc.handle.fd = memory_fd;
                external_desc.size      = requirements.size;
                cuda_ok(cudaImportExternalMemory(&slot.external_memory, &external_desc), "cudaImportExternalMemory");

                vk::SemaphoreGetFdInfoKHR semaphore_handle_info{
                    .semaphore  = *slot.timeline_semaphore,
                    .handleType = semaphore_handle_type,
                };
                const int semaphore_fd = renderer.vk_context().device.getSemaphoreFdKHR(semaphore_handle_info);
                if (semaphore_fd < 0) throw std::runtime_error("getSemaphoreFdKHR returned an invalid fd");

                cudaExternalSemaphoreHandleDesc external_semaphore_desc{};
                external_semaphore_desc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
                external_semaphore_desc.handle.fd = semaphore_fd;
                cuda_ok(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
#endif

                cudaExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.offset = 0;
                buffer_desc.size   = viewer_runtime.field_bytes;
                cuda_ok(cudaExternalMemoryGetMappedBuffer(&slot.cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");

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
            const auto begin  = std::chrono::steady_clock::now();
            auto& slot        = snapshot_slots.at(static_cast<size_t>(slot_index));
            const auto& field = current_field();
            const auto grid   = current_grid();

            if (backend_kind == BackendKind::StableFluids001) {
                if (field.id == FieldId::Density) {
                    cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, stable_runtime.density, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, stable_runtime.sim_stream), "cudaMemcpyAsync stable density snapshot");
                } else {
                    stable_ok(app_compute_staggered_velocity_magnitude_async(
                                  slot.cuda_ptr, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz), stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z, stable_runtime.sim_stream),
                        "app_compute_staggered_velocity_magnitude_async");
                }
            } else {
                if (field.id == FieldId::Density) {
                    cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.density, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual density snapshot");
                } else if (field.id == FieldId::Temperature) {
                    cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.temperature, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual temperature snapshot");
                } else {
                    smoke_ok(app_compute_staggered_velocity_magnitude_async(
                                 slot.cuda_ptr, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz), visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream),
                        "app_compute_staggered_velocity_magnitude_async");
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
            slot.nx                             = grid.nx;
            slot.ny                             = grid.ny;
            slot.nz                             = grid.nz;
            slot.cell_size                      = grid.cell_size;
            slot.semantic                       = field.semantic;
            slot.label                          = field.label;
            const auto elapsed_ms               = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            auto& solver_stats                  = current_solver_stats();
            accumulate_sample(elapsed_ms, solver_stats.last_snapshot_ms, solver_stats.average_snapshot_ms, solver_stats.snapshot_count);
        };

        auto reset_backend = [&]() {
            nvtx3::scoped_range range{"smoke_app.reset_backend"};

            const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
            if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) throw std::runtime_error("smoke-visualizer requires Vulkan timeline semaphore support");

            int cuda_device_index = 0;
            cuda_ok(cudaGetDevice(&cuda_device_index), "cudaGetDevice");
            int cuda_timeline_semaphore_interop_supported = 0;
            cuda_ok(cudaDeviceGetAttribute(&cuda_timeline_semaphore_interop_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index), "cudaDeviceGetAttribute cudaDevAttrTimelineSemaphoreInteropSupported");
            if (cuda_timeline_semaphore_interop_supported == 0) throw std::runtime_error("smoke-visualizer requires CUDA timeline semaphore interop support");

            destroy_everything();
            current_solver_stats() = {};

            if (backend_kind == BackendKind::StableFluids001) {
                viewer_runtime.field_bytes         = scalar_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                const auto stable_velocity_x_bytes = velocity_x_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                const auto stable_velocity_y_bytes = velocity_y_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                const auto stable_velocity_z_bytes = velocity_z_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                stable_runtime.device_allocated    = true;
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
                viewer_runtime.field_bytes         = scalar_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                const auto visual_velocity_x_bytes = velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                const auto visual_velocity_y_bytes = velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                const auto visual_velocity_z_bytes = velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                visual_runtime.device_allocated    = true;
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
            }

            create_snapshot_slots();

            if (backend_kind == BackendKind::StableFluids001) {
                cuda_ok(cudaMemsetAsync(stable_runtime.density, 0, viewer_runtime.field_bytes, stable_runtime.sim_stream), "cudaMemsetAsync stable density");
                cuda_ok(cudaMemsetAsync(stable_runtime.velocity_x, 0, velocity_x_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream), "cudaMemsetAsync stable velocity_x");
                cuda_ok(cudaMemsetAsync(stable_runtime.velocity_y, 0, velocity_y_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream), "cudaMemsetAsync stable velocity_y");
                cuda_ok(cudaMemsetAsync(stable_runtime.velocity_z, 0, velocity_z_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream), "cudaMemsetAsync stable velocity_z");
                if (stable_settings.emit_source) {
                    const float center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx);
                    const float center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny);
                    const float center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz);
                    stable_ok(app_add_stable_source_async(stable_runtime.density, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount, stable_settings.velocity_x,
                                  stable_settings.velocity_y, stable_settings.velocity_z, stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z, stable_runtime.sim_stream),
                        "app_add_stable_source_async");
                    run_stable_step();
                }
            } else {
                cuda_ok(cudaMemsetAsync(visual_runtime.density, 0, viewer_runtime.field_bytes, visual_runtime.sim_stream), "cudaMemsetAsync visual density");
                cuda_ok(cudaMemsetAsync(visual_runtime.temperature, 0, viewer_runtime.field_bytes, visual_runtime.sim_stream), "cudaMemsetAsync visual temperature");
                cuda_ok(cudaMemsetAsync(visual_runtime.velocity_x, 0, velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.sim_stream), "cudaMemsetAsync visual velocity_x");
                cuda_ok(cudaMemsetAsync(visual_runtime.velocity_y, 0, velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.sim_stream), "cudaMemsetAsync visual velocity_y");
                cuda_ok(cudaMemsetAsync(visual_runtime.velocity_z, 0, velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.sim_stream), "cudaMemsetAsync visual velocity_z");
                if (visual_settings.emit_source) {
                    const float center_x = visual_settings.source_u * static_cast<float>(visual_settings.desc.nx);
                    const float center_y = visual_settings.source_v * static_cast<float>(visual_settings.desc.ny);
                    const float center_z = visual_settings.source_w * static_cast<float>(visual_settings.desc.nz);
                    smoke_ok(app_add_visual_source_async(visual_runtime.density, visual_runtime.temperature, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, center_x, center_y, center_z, visual_settings.source_radius, visual_settings.density_amount,
                                 visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream),
                        "app_add_visual_source_async");
                    run_visual_step();
                }
            }

            snapshot_current_field_to_slot(0, "smoke_app.reset_backend.initial_snapshot");
            current_solver_stats() = {};
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
            {
                const auto count_levels = [](int32_t nx, int32_t ny, int32_t nz) {
                    int levels = 1;
                    while (nx > 1 || ny > 1 || nz > 1) {
                        nx = std::max(1, (nx + 1) / 2);
                        ny = std::max(1, (ny + 1) / 2);
                        nz = std::max(1, (nz + 1) / 2);
                        ++levels;
                    }
                    return std::array<int32_t, 4>{levels, nx, ny, nz};
                };
                const auto& solver_stats = current_solver_stats();
                ImGui::TextUnformatted("Solver");
                if (backend_kind == BackendKind::StableFluids001) {
                    const auto hierarchy            = count_levels(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                    const int pressure_v_cycles     = std::max(1, stable_settings.desc.pressure_iterations / 40);
                    const int pressure_coarse_steps = std::max(8, stable_settings.desc.pressure_iterations / 10);
                    ImGui::Text("Projection: Multigrid V-cycle");
                    ImGui::Text("Pressure budget: %d iterations -> %d V-cycles, 1 smooth, %d coarse", stable_settings.desc.pressure_iterations, pressure_v_cycles, pressure_coarse_steps);
                    if (stable_settings.desc.diffusion > 0.0f) {
                        const int diffusion_v_cycles     = std::max(1, stable_settings.desc.diffuse_iterations / 12);
                        const int diffusion_coarse_steps = std::max(6, stable_settings.desc.diffuse_iterations / 4);
                        ImGui::Text("Density diffusion: Multigrid V-cycle");
                        ImGui::Text("Diffuse budget: %d iterations -> %d V-cycles, 1 smooth, %d coarse", stable_settings.desc.diffuse_iterations, diffusion_v_cycles, diffusion_coarse_steps);
                    } else
                        ImGui::TextUnformatted("Density diffusion: Disabled");
                    if (stable_settings.desc.viscosity > 0.0f) {
                        const int viscosity_v_cycles     = std::max(1, stable_settings.desc.diffuse_iterations / 12);
                        const int viscosity_coarse_steps = std::max(6, stable_settings.desc.diffuse_iterations / 4);
                        ImGui::Text("Velocity diffusion: Multigrid V-cycle");
                        ImGui::Text("Viscosity budget: %d iterations -> %d V-cycles, 1 smooth, %d coarse", stable_settings.desc.diffuse_iterations, viscosity_v_cycles, viscosity_coarse_steps);
                    } else
                        ImGui::TextUnformatted("Velocity diffusion: Disabled");
                    ImGui::Text("Hierarchy: %d levels, coarsest %d x %d x %d", hierarchy[0], hierarchy[1], hierarchy[2], hierarchy[3]);
                } else {
                    const auto hierarchy            = count_levels(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                    const int pressure_v_cycles     = std::max(1, visual_settings.desc.pressure_iterations / 40);
                    const int pressure_coarse_steps = std::max(8, visual_settings.desc.pressure_iterations / 10);
                    ImGui::Text("Projection: Multigrid V-cycle");
                    ImGui::Text("Pressure budget: %d iterations -> %d V-cycles, 1 smooth, %d coarse", visual_settings.desc.pressure_iterations, pressure_v_cycles, pressure_coarse_steps);
                    ImGui::Text("Advection: %s", visual_settings.desc.use_monotonic_cubic != 0u ? "Monotonic cubic" : "Linear");
                    ImGui::Text("Hierarchy: %d levels, coarsest %d x %d x %d", hierarchy[0], hierarchy[1], hierarchy[2], hierarchy[3]);
                }
                ImGui::Text("Step calls: %llu", static_cast<unsigned long long>(solver_stats.step_count));
                ImGui::Text("Snapshot commits: %llu", static_cast<unsigned long long>(solver_stats.snapshot_count));
                ImGui::Text("Last step call: %.3f ms", solver_stats.last_step_call_ms);
                ImGui::Text("Avg step call: %.3f ms", solver_stats.average_step_call_ms);
                ImGui::Text("Last snapshot: %.3f ms", solver_stats.last_snapshot_ms);
                ImGui::Text("Avg snapshot: %.3f ms", solver_stats.average_snapshot_ms);
                ImGui::TextUnformatted("CUDA timings are app-side call costs, not kernel profiler numbers.");
                ImGui::Separator();
            }

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
                auto draw_boundary_combo = [&](const char* label, uint32_t& value) {
                    int boundary = std::clamp(static_cast<int>(value), 0, static_cast<int>(stable_boundary_labels.size()) - 1);
                    if (ImGui::BeginCombo(label, stable_boundary_labels[static_cast<size_t>(boundary)])) {
                        for (int i = 0; i < static_cast<int>(stable_boundary_labels.size()); ++i) {
                            const bool is_selected = boundary == i;
                            if (ImGui::Selectable(stable_boundary_labels[static_cast<size_t>(i)], is_selected)) {
                                boundary        = i;
                                value           = static_cast<uint32_t>(i);
                                reset_requested = true;
                            }
                            if (is_selected) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                };
                ImGui::Separator();
                ImGui::TextUnformatted("Boundary Conditions");
                draw_boundary_combo("Boundary X-", stable_settings.desc.boundary_x_min);
                draw_boundary_combo("Boundary X+", stable_settings.desc.boundary_x_max);
                draw_boundary_combo("Boundary Y-", stable_settings.desc.boundary_y_min);
                draw_boundary_combo("Boundary Y+", stable_settings.desc.boundary_y_max);
                draw_boundary_combo("Boundary Z-", stable_settings.desc.boundary_z_min);
                draw_boundary_combo("Boundary Z+", stable_settings.desc.boundary_z_max);
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
                                stable_ok(app_add_stable_source_async(stable_runtime.density, stable_runtime.velocity_x, stable_runtime.velocity_y, stable_runtime.velocity_z, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount,
                                              stable_settings.velocity_x, stable_settings.velocity_y, stable_settings.velocity_z, stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z, stable_runtime.sim_stream),
                                    "app_add_stable_source_async");
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                const auto begin = std::chrono::steady_clock::now();
                                run_stable_step();
                                const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                                accumulate_sample(elapsed_ms, stable_solver_stats.last_step_call_ms, stable_solver_stats.average_step_call_ms, stable_solver_stats.step_count);
                            }
                        } else {
                            if (visual_settings.emit_source) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_source"};
                                const float center_x = visual_settings.source_u * static_cast<float>(visual_settings.desc.nx);
                                const float center_y = visual_settings.source_v * static_cast<float>(visual_settings.desc.ny);
                                const float center_z = visual_settings.source_w * static_cast<float>(visual_settings.desc.nz);
                                smoke_ok(app_add_visual_source_async(visual_runtime.density, visual_runtime.temperature, visual_runtime.velocity_x, visual_runtime.velocity_y, visual_runtime.velocity_z, visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, center_x, center_y, center_z, visual_settings.source_radius,
                                             visual_settings.density_amount, visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream),
                                    "app_add_visual_source_async");
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                const auto begin = std::chrono::steady_clock::now();
                                run_visual_step();
                                const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                                accumulate_sample(elapsed_ms, visual_solver_stats.last_step_call_ms, visual_solver_stats.average_step_call_ms, visual_solver_stats.step_count);
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
