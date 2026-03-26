#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "stable-fluids.h"
#include <cuda_runtime.h>
#include <imgui.h>

#include <nvtx3/nvtx3.hpp>
#include <vulkan/vulkan_raii.hpp>

import app;
import std;
import vk.memory;

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    enum class FieldId : uint32_t {
        SmokeColor        = 0,
        Density           = 1,
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
        FieldChoice{FieldId::SmokeColor, "Smoke Color", app::FieldSemantic::DyeColor},
        FieldChoice{FieldId::Density, "Density", app::FieldSemantic::Density},
        FieldChoice{FieldId::VelocityMagnitude, "Velocity Magnitude", app::FieldSemantic::VelocityMagnitude},
    };

    constexpr std::array boundary_labels{
        "No-slip",
        "Free-slip",
        "Inflow",
        "Outflow",
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
        int32_t nx                  = 128;
        int32_t ny                  = 128;
        int32_t nz                  = 128;
        float dt                    = 1.0f / 90.0f;
        float cell_size             = 1.0f;
        float viscosity             = 0.00015f;
        float diffusion             = 0.00005f;
        int32_t diffuse_iterations  = 24;
        int32_t pressure_iterations = 80;
        uint32_t boundary_x_min     = STABLE_FLUIDS_BOUNDARY_OUTFLOW;
        uint32_t boundary_x_max     = STABLE_FLUIDS_BOUNDARY_OUTFLOW;
        uint32_t boundary_y_min     = STABLE_FLUIDS_BOUNDARY_OUTFLOW;
        uint32_t boundary_y_max     = STABLE_FLUIDS_BOUNDARY_OUTFLOW;
        uint32_t boundary_z_min     = STABLE_FLUIDS_BOUNDARY_OUTFLOW;
        uint32_t boundary_z_max     = STABLE_FLUIDS_BOUNDARY_OUTFLOW;
        float inflow_velocity_x_min = 0.0f;
        float inflow_velocity_x_max = 0.0f;
        float inflow_velocity_y_min = 1.2f;
        float inflow_velocity_y_max = 0.0f;
        float inflow_velocity_z_min = 0.0f;
        float inflow_velocity_z_max = 0.0f;
        float inflow_scalar_x_min   = 0.8f;
        float inflow_scalar_x_max   = 0.8f;
        float inflow_scalar_y_min   = 0.8f;
        float inflow_scalar_y_max   = 0.0f;
        float inflow_scalar_z_min   = 0.8f;
        float inflow_scalar_z_max   = 0.8f;
        float ambient_temperature   = 0.0f;
        float density_buoyancy      = 0.35f;
        float temperature_buoyancy  = 0.0f;
        float uniform_force_x       = 0.0f;
        float uniform_force_y       = 0.0f;
        float uniform_force_z       = 0.0f;
        int32_t block_x             = 8;
        int32_t block_y             = 8;
        int32_t block_z             = 4;
    };

    struct StableSettings {
        StableStepConfig desc{};
        int selected_field   = 0;
        bool emit_source     = true;
        float source_radius  = 3.5f;
        float density_amount = 0.55f;
        float dye_amount     = 0.65f;
        float source_a_r     = 1.00f;
        float source_a_g     = 0.20f;
        float source_a_b     = 0.72f;
        float source_b_r     = 0.12f;
        float source_b_g     = 0.38f;
        float source_b_b     = 1.00f;
        float jet_speed      = 48.0f;
        float upward_bias    = 0.20f;
        float corner_inset   = 0.14f;
        float source_height  = 0.10f;
        float source_depth   = 0.14f;
    };

    struct StableRuntime {
        bool device_allocated                = false;
        cudaStream_t sim_stream              = nullptr;
        float* density                       = nullptr;
        float* dye_r                         = nullptr;
        float* dye_g                         = nullptr;
        float* dye_b                         = nullptr;
        float* velocity_x                    = nullptr;
        float* velocity_y                    = nullptr;
        float* velocity_z                    = nullptr;
        float* temporary_density             = nullptr;
        float* temporary_dye_r               = nullptr;
        float* temporary_dye_g               = nullptr;
        float* temporary_dye_b               = nullptr;
        float* temporary_velocity_x          = nullptr;
        float* temporary_velocity_y          = nullptr;
        float* temporary_velocity_z          = nullptr;
        float* temporary_previous_density    = nullptr;
        float* temporary_previous_dye_r      = nullptr;
        float* temporary_previous_dye_g      = nullptr;
        float* temporary_previous_dye_b      = nullptr;
        float* temporary_previous_velocity_x = nullptr;
        float* temporary_previous_velocity_y = nullptr;
        float* temporary_previous_velocity_z = nullptr;
        float* temporary_pressure            = nullptr;
        float* temporary_divergence          = nullptr;
    };

} // namespace

int main() {
    try {
        app::FieldRendererApp renderer;
        PlaybackState playback{};
        StableSettings stable_settings{};
        StableRuntime stable_runtime{};
        SolverStats stable_solver_stats{};
        ViewerRuntime viewer_runtime{};
        std::vector<SnapshotSlot> snapshot_slots{};

        auto scalar_bytes     = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto rgba_bytes       = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float) * 4ull; };
        auto velocity_x_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_y_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz) * sizeof(float); };
        auto velocity_z_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) { return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz + 1) * sizeof(float); };
        auto current_field = [&]() -> const FieldChoice& {
            stable_settings.selected_field = std::clamp(stable_settings.selected_field, 0, static_cast<int>(stable_fields.size()) - 1);
            return stable_fields[static_cast<size_t>(stable_settings.selected_field)];
        };
        auto accumulate_sample    = [](const double sample_ms, double& last_ms, double& average_ms, uint64_t& count) {
            last_ms = sample_ms;
            ++count;
            average_ms += (sample_ms - average_ms) / static_cast<double>(count);
        };

        auto run_stable_step = [&]() {
            StableFluidsAdvectVelocityDesc advect_velocity_desc{
                .struct_size                   = sizeof(StableFluidsAdvectVelocityDesc),
                .api_version                   = STABLE_FLUIDS_API_VERSION,
                .nx                            = stable_settings.desc.nx,
                .ny                            = stable_settings.desc.ny,
                .nz                            = stable_settings.desc.nz,
                .cell_size                     = stable_settings.desc.cell_size,
                .dt                            = stable_settings.desc.dt,
                .boundary_x_min                = stable_settings.desc.boundary_x_min,
                .boundary_x_max                = stable_settings.desc.boundary_x_max,
                .boundary_y_min                = stable_settings.desc.boundary_y_min,
                .boundary_y_max                = stable_settings.desc.boundary_y_max,
                .boundary_z_min                = stable_settings.desc.boundary_z_min,
                .boundary_z_max                = stable_settings.desc.boundary_z_max,
                .inflow_velocity_x_min        = stable_settings.desc.inflow_velocity_x_min,
                .inflow_velocity_x_max        = stable_settings.desc.inflow_velocity_x_max,
                .inflow_velocity_y_min        = stable_settings.desc.inflow_velocity_y_min,
                .inflow_velocity_y_max        = stable_settings.desc.inflow_velocity_y_max,
                .inflow_velocity_z_min        = stable_settings.desc.inflow_velocity_z_min,
                .inflow_velocity_z_max        = stable_settings.desc.inflow_velocity_z_max,
                .velocity_x                    = stable_runtime.velocity_x,
                .velocity_y                    = stable_runtime.velocity_y,
                .velocity_z                    = stable_runtime.velocity_z,
                .temporary_velocity_x          = stable_runtime.temporary_velocity_x,
                .temporary_velocity_y          = stable_runtime.temporary_velocity_y,
                .temporary_velocity_z          = stable_runtime.temporary_velocity_z,
                .temporary_previous_velocity_x = stable_runtime.temporary_previous_velocity_x,
                .temporary_previous_velocity_y = stable_runtime.temporary_previous_velocity_y,
                .temporary_previous_velocity_z = stable_runtime.temporary_previous_velocity_z,
                .block_x                       = stable_settings.desc.block_x,
                .block_y                       = stable_settings.desc.block_y,
                .block_z                       = stable_settings.desc.block_z,
                .stream                        = stable_runtime.sim_stream,
            };

            StableFluidsDiffuseVelocityDesc diffuse_velocity_desc{
                .struct_size                = sizeof(StableFluidsDiffuseVelocityDesc),
                .api_version                = STABLE_FLUIDS_API_VERSION,
                .nx                         = stable_settings.desc.nx,
                .ny                         = stable_settings.desc.ny,
                .nz                         = stable_settings.desc.nz,
                .cell_size                  = stable_settings.desc.cell_size,
                .dt                         = stable_settings.desc.dt,
                .viscosity                  = stable_settings.desc.viscosity,
                .diffuse_iterations         = stable_settings.desc.diffuse_iterations,
                .boundary_x_min             = stable_settings.desc.boundary_x_min,
                .boundary_x_max             = stable_settings.desc.boundary_x_max,
                .boundary_y_min             = stable_settings.desc.boundary_y_min,
                .boundary_y_max             = stable_settings.desc.boundary_y_max,
                .boundary_z_min             = stable_settings.desc.boundary_z_min,
                .boundary_z_max             = stable_settings.desc.boundary_z_max,
                .inflow_velocity_x_min     = stable_settings.desc.inflow_velocity_x_min,
                .inflow_velocity_x_max     = stable_settings.desc.inflow_velocity_x_max,
                .inflow_velocity_y_min     = stable_settings.desc.inflow_velocity_y_min,
                .inflow_velocity_y_max     = stable_settings.desc.inflow_velocity_y_max,
                .inflow_velocity_z_min     = stable_settings.desc.inflow_velocity_z_min,
                .inflow_velocity_z_max     = stable_settings.desc.inflow_velocity_z_max,
                .velocity_x                 = stable_runtime.velocity_x,
                .velocity_y                 = stable_runtime.velocity_y,
                .velocity_z                 = stable_runtime.velocity_z,
                .temporary_velocity_x       = stable_runtime.temporary_velocity_x,
                .temporary_velocity_y       = stable_runtime.temporary_velocity_y,
                .temporary_velocity_z       = stable_runtime.temporary_velocity_z,
                .temporary_density          = stable_runtime.temporary_density,
                .temporary_previous_density = stable_runtime.temporary_previous_density,
                .block_x                    = stable_settings.desc.block_x,
                .block_y                    = stable_settings.desc.block_y,
                .block_z                    = stable_settings.desc.block_z,
                .stream                     = stable_runtime.sim_stream,
            };

            StableFluidsProjectDesc project_desc{
                .struct_size                = sizeof(StableFluidsProjectDesc),
                .api_version                = STABLE_FLUIDS_API_VERSION,
                .nx                         = stable_settings.desc.nx,
                .ny                         = stable_settings.desc.ny,
                .nz                         = stable_settings.desc.nz,
                .cell_size                  = stable_settings.desc.cell_size,
                .pressure_iterations        = stable_settings.desc.pressure_iterations,
                .boundary_x_min             = stable_settings.desc.boundary_x_min,
                .boundary_x_max             = stable_settings.desc.boundary_x_max,
                .boundary_y_min             = stable_settings.desc.boundary_y_min,
                .boundary_y_max             = stable_settings.desc.boundary_y_max,
                .boundary_z_min             = stable_settings.desc.boundary_z_min,
                .boundary_z_max             = stable_settings.desc.boundary_z_max,
                .inflow_velocity_x_min     = stable_settings.desc.inflow_velocity_x_min,
                .inflow_velocity_x_max     = stable_settings.desc.inflow_velocity_x_max,
                .inflow_velocity_y_min     = stable_settings.desc.inflow_velocity_y_min,
                .inflow_velocity_y_max     = stable_settings.desc.inflow_velocity_y_max,
                .inflow_velocity_z_min     = stable_settings.desc.inflow_velocity_z_min,
                .inflow_velocity_z_max     = stable_settings.desc.inflow_velocity_z_max,
                .velocity_x                 = stable_runtime.velocity_x,
                .velocity_y                 = stable_runtime.velocity_y,
                .velocity_z                 = stable_runtime.velocity_z,
                .temporary_pressure         = stable_runtime.temporary_pressure,
                .temporary_divergence       = stable_runtime.temporary_divergence,
                .temporary_density          = stable_runtime.temporary_density,
                .temporary_previous_density = stable_runtime.temporary_previous_density,
                .block_x                    = stable_settings.desc.block_x,
                .block_y                    = stable_settings.desc.block_y,
                .block_z                    = stable_settings.desc.block_z,
                .stream                     = stable_runtime.sim_stream,
            };

            const float inv_dt = 1.0f / (std::max)(stable_settings.desc.dt, 1.0e-6f);
            StableFluidsAddForceDesc add_force_desc{
                .struct_size            = sizeof(StableFluidsAddForceDesc),
                .api_version            = STABLE_FLUIDS_API_VERSION,
                .nx                     = stable_settings.desc.nx,
                .ny                     = stable_settings.desc.ny,
                .nz                     = stable_settings.desc.nz,
                .dt                     = stable_settings.desc.dt,
                .boundary_x_min         = stable_settings.desc.boundary_x_min,
                .boundary_x_max         = stable_settings.desc.boundary_x_max,
                .boundary_y_min         = stable_settings.desc.boundary_y_min,
                .boundary_y_max         = stable_settings.desc.boundary_y_max,
                .boundary_z_min         = stable_settings.desc.boundary_z_min,
                .boundary_z_max         = stable_settings.desc.boundary_z_max,
                .inflow_velocity_x_min  = stable_settings.desc.inflow_velocity_x_min,
                .inflow_velocity_x_max  = stable_settings.desc.inflow_velocity_x_max,
                .inflow_velocity_y_min  = stable_settings.desc.inflow_velocity_y_min,
                .inflow_velocity_y_max  = stable_settings.desc.inflow_velocity_y_max,
                .inflow_velocity_z_min  = stable_settings.desc.inflow_velocity_z_min,
                .inflow_velocity_z_max  = stable_settings.desc.inflow_velocity_z_max,
                .ambient_temperature    = stable_settings.desc.ambient_temperature,
                .density_buoyancy       = stable_settings.desc.density_buoyancy * inv_dt,
                .temperature_buoyancy   = 0.0f,
                .uniform_force_x        = stable_settings.desc.uniform_force_x * inv_dt,
                .uniform_force_y        = stable_settings.desc.uniform_force_y * inv_dt,
                .uniform_force_z        = stable_settings.desc.uniform_force_z * inv_dt,
                .velocity_x             = stable_runtime.velocity_x,
                .velocity_y             = stable_runtime.velocity_y,
                .velocity_z             = stable_runtime.velocity_z,
                .density                = stable_runtime.density,
                .temperature            = nullptr,
                .force_x                = nullptr,
                .force_y                = nullptr,
                .force_z                = nullptr,
                .block_x                = stable_settings.desc.block_x,
                .block_y                = stable_settings.desc.block_y,
                .block_z                = stable_settings.desc.block_z,
                .stream                 = stable_runtime.sim_stream,
            };


            auto run_scalar_flow = [&](float* scalar, float* temporary, float* temporary_previous, const uint32_t clamp_non_negative, const float inflow_x_min, const float inflow_x_max, const float inflow_y_min, const float inflow_y_max, const float inflow_z_min, const float inflow_z_max) {
                StableFluidsAdvectScalarDesc scalar_advect{
                    .struct_size               = sizeof(StableFluidsAdvectScalarDesc),
                    .api_version               = STABLE_FLUIDS_API_VERSION,
                    .nx                        = stable_settings.desc.nx,
                    .ny                        = stable_settings.desc.ny,
                    .nz                        = stable_settings.desc.nz,
                    .cell_size                 = stable_settings.desc.cell_size,
                    .dt                        = stable_settings.desc.dt,
                    .boundary_x_min            = stable_settings.desc.boundary_x_min,
                    .boundary_x_max            = stable_settings.desc.boundary_x_max,
                    .boundary_y_min            = stable_settings.desc.boundary_y_min,
                    .boundary_y_max            = stable_settings.desc.boundary_y_max,
                    .boundary_z_min            = stable_settings.desc.boundary_z_min,
                    .boundary_z_max            = stable_settings.desc.boundary_z_max,
                    .inflow_scalar_x_min       = inflow_x_min,
                    .inflow_scalar_x_max       = inflow_x_max,
                    .inflow_scalar_y_min       = inflow_y_min,
                    .inflow_scalar_y_max       = inflow_y_max,
                    .inflow_scalar_z_min       = inflow_z_min,
                    .inflow_scalar_z_max       = inflow_z_max,
                    .scalar                    = scalar,
                    .temporary_scalar          = temporary,
                    .temporary_previous_scalar = temporary_previous,
                    .velocity_x                = stable_runtime.velocity_x,
                    .velocity_y                = stable_runtime.velocity_y,
                    .velocity_z                = stable_runtime.velocity_z,
                    .clamp_non_negative        = clamp_non_negative,
                    .block_x                   = stable_settings.desc.block_x,
                    .block_y                   = stable_settings.desc.block_y,
                    .block_z                   = stable_settings.desc.block_z,
                    .stream                    = stable_runtime.sim_stream,
                };
                StableFluidsDiffuseScalarDesc scalar_diffuse{
                    .struct_size                = sizeof(StableFluidsDiffuseScalarDesc),
                    .api_version                = STABLE_FLUIDS_API_VERSION,
                    .nx                         = stable_settings.desc.nx,
                    .ny                         = stable_settings.desc.ny,
                    .nz                         = stable_settings.desc.nz,
                    .cell_size                  = stable_settings.desc.cell_size,
                    .dt                         = stable_settings.desc.dt,
                    .diffusion                  = stable_settings.desc.diffusion,
                    .diffuse_iterations         = stable_settings.desc.diffuse_iterations,
                    .boundary_x_min             = stable_settings.desc.boundary_x_min,
                    .boundary_x_max             = stable_settings.desc.boundary_x_max,
                    .boundary_y_min             = stable_settings.desc.boundary_y_min,
                    .boundary_y_max             = stable_settings.desc.boundary_y_max,
                    .boundary_z_min             = stable_settings.desc.boundary_z_min,
                    .boundary_z_max             = stable_settings.desc.boundary_z_max,
                    .inflow_scalar_x_min        = inflow_x_min,
                    .inflow_scalar_x_max        = inflow_x_max,
                    .inflow_scalar_y_min        = inflow_y_min,
                    .inflow_scalar_y_max        = inflow_y_max,
                    .inflow_scalar_z_min        = inflow_z_min,
                    .inflow_scalar_z_max        = inflow_z_max,
                    .scalar                     = scalar,
                    .temporary_scalar           = temporary,
                    .temporary_solution_storage = stable_runtime.temporary_pressure,
                    .temporary_rhs_storage      = stable_runtime.temporary_divergence,
                    .clamp_non_negative         = clamp_non_negative,
                    .block_x                    = stable_settings.desc.block_x,
                    .block_y                    = stable_settings.desc.block_y,
                    .block_z                    = stable_settings.desc.block_z,
                    .stream                     = stable_runtime.sim_stream,
                };
                if (const auto code = stable_fluids_advect_scalar_cuda(&scalar_advect); code != 0) throw std::runtime_error("stable_fluids_advect_scalar_cuda failed");
                if (const auto code = stable_fluids_diffuse_scalar_cuda(&scalar_diffuse); code != 0) throw std::runtime_error("stable_fluids_diffuse_scalar_cuda failed");
            };

            if (const auto code = stable_fluids_add_force_cuda(&add_force_desc); code != 0) throw std::runtime_error("stable_fluids_add_force_cuda failed");
            if (const auto code = stable_fluids_advect_velocity_cuda(&advect_velocity_desc); code != 0) throw std::runtime_error("stable_fluids_advect_velocity_cuda failed");
            if (const auto code = stable_fluids_diffuse_velocity_cuda(&diffuse_velocity_desc); code != 0) throw std::runtime_error("stable_fluids_diffuse_velocity_cuda failed");
            if (const auto code = stable_fluids_project_cuda(&project_desc); code != 0) throw std::runtime_error("stable_fluids_project_cuda failed");
            run_scalar_flow(stable_runtime.density, stable_runtime.temporary_density, stable_runtime.temporary_previous_density, 1u, stable_settings.desc.inflow_scalar_x_min, stable_settings.desc.inflow_scalar_x_max, stable_settings.desc.inflow_scalar_y_min, stable_settings.desc.inflow_scalar_y_max, stable_settings.desc.inflow_scalar_z_min, stable_settings.desc.inflow_scalar_z_max);
            run_scalar_flow(stable_runtime.dye_r, stable_runtime.temporary_dye_r, stable_runtime.temporary_previous_dye_r, 1u, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            run_scalar_flow(stable_runtime.dye_g, stable_runtime.temporary_dye_g, stable_runtime.temporary_previous_dye_g, 1u, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            run_scalar_flow(stable_runtime.dye_b, stable_runtime.temporary_dye_b, stable_runtime.temporary_previous_dye_b, 1u, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        };


        auto current_grid = [&]() {
            struct GridInfo {
                uint32_t nx;
                uint32_t ny;
                uint32_t nz;
                float cell_size;
            };

            return GridInfo{
                static_cast<uint32_t>(stable_settings.desc.nx),
                static_cast<uint32_t>(stable_settings.desc.ny),
                static_cast<uint32_t>(stable_settings.desc.nz),
                stable_settings.desc.cell_size,
            };
        };

        auto apply_field_defaults = [&](const FieldChoice& field) {
            auto& render = renderer.render_settings();
            if (field.id == FieldId::SmokeColor) {
                render.mode         = app::RenderMode::Smoke;
                render.march_steps  = 96;
                render.density_scale = 0.95f;
                render.absorption   = 1.20f;
            } else if (field.id == FieldId::Density) {
                render.mode           = app::RenderMode::Scalar;
                render.scalar_min     = 0.0f;
                render.scalar_max     = 0.70f;
                render.scalar_opacity = 2.1f;
                render.scalar_low_r   = 0.10f;
                render.scalar_low_g   = 0.08f;
                render.scalar_low_b   = 0.30f;
                render.scalar_high_r  = 1.00f;
                render.scalar_high_g  = 0.24f;
                render.scalar_high_b  = 0.74f;
            } else {
                render.mode           = app::RenderMode::Scalar;
                render.scalar_min     = 0.0f;
                render.scalar_max     = 3.0f;
                render.scalar_opacity = 1.35f;
                render.scalar_low_r   = 0.06f;
                render.scalar_low_g   = 0.10f;
                render.scalar_low_b   = 0.42f;
                render.scalar_high_r  = 0.18f;
                render.scalar_high_g  = 0.88f;
                render.scalar_high_b  = 1.00f;
            }
        };


        auto inject_stable_dual_sources = [&]() {
            const auto nx        = static_cast<float>(stable_settings.desc.nx);
            const auto ny        = static_cast<float>(stable_settings.desc.ny);
            const auto nz        = static_cast<float>(stable_settings.desc.nz);
            const float center_x = nx * 0.5f;
            const float center_y = ny * 0.52f;
            const float center_z = nz * 0.5f;
            const float left_x   = nx * stable_settings.corner_inset;
            const float right_x  = nx * (1.0f - stable_settings.corner_inset);
            const float source_y = ny * stable_settings.source_height;
            const float source_z = nz * stable_settings.source_depth;

            auto emit_one = [&](const float source_x, const float color_r, const float color_g, const float color_b) {
                const float dir_x      = center_x - source_x;
                const float dir_y      = center_y - source_y;
                const float dir_z      = center_z - source_z;
                const float inv_len    = 1.0f / (std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z) + 1.0e-6f);
                const float velocity_x = dir_x * inv_len * stable_settings.jet_speed;
                const float velocity_y = dir_y * inv_len * stable_settings.jet_speed + stable_settings.upward_bias;
                const float velocity_z = dir_z * inv_len * stable_settings.jet_speed;
                StableFluidsAddScalarSourceDesc scalar_source_desc{
                    .struct_size      = sizeof(StableFluidsAddScalarSourceDesc),
                    .api_version      = STABLE_FLUIDS_API_VERSION,
                    .nx               = stable_settings.desc.nx,
                    .ny               = stable_settings.desc.ny,
                    .nz               = stable_settings.desc.nz,
                    .scalar           = stable_runtime.density,
                    .center_x         = source_x,
                    .center_y         = source_y,
                    .center_z         = source_z,
                    .radius           = stable_settings.source_radius,
                    .amount           = stable_settings.density_amount,
                    .sample_offset_x  = 0.5f,
                    .sample_offset_y  = 0.5f,
                    .sample_offset_z  = 0.5f,
                    .block_x          = stable_settings.desc.block_x,
                    .block_y          = stable_settings.desc.block_y,
                    .block_z          = stable_settings.desc.block_z,
                    .stream           = stable_runtime.sim_stream,
                };
                StableFluidsAddScalarSourceDesc dye_r_source_desc{
                    .struct_size      = sizeof(StableFluidsAddScalarSourceDesc),
                    .api_version      = STABLE_FLUIDS_API_VERSION,
                    .nx               = stable_settings.desc.nx,
                    .ny               = stable_settings.desc.ny,
                    .nz               = stable_settings.desc.nz,
                    .scalar           = stable_runtime.dye_r,
                    .center_x         = source_x,
                    .center_y         = source_y,
                    .center_z         = source_z,
                    .radius           = stable_settings.source_radius,
                    .amount           = stable_settings.dye_amount * color_r,
                    .sample_offset_x  = 0.5f,
                    .sample_offset_y  = 0.5f,
                    .sample_offset_z  = 0.5f,
                    .block_x          = stable_settings.desc.block_x,
                    .block_y          = stable_settings.desc.block_y,
                    .block_z          = stable_settings.desc.block_z,
                    .stream           = stable_runtime.sim_stream,
                };
                StableFluidsAddScalarSourceDesc dye_g_source_desc{
                    .struct_size      = sizeof(StableFluidsAddScalarSourceDesc),
                    .api_version      = STABLE_FLUIDS_API_VERSION,
                    .nx               = stable_settings.desc.nx,
                    .ny               = stable_settings.desc.ny,
                    .nz               = stable_settings.desc.nz,
                    .scalar           = stable_runtime.dye_g,
                    .center_x         = source_x,
                    .center_y         = source_y,
                    .center_z         = source_z,
                    .radius           = stable_settings.source_radius,
                    .amount           = stable_settings.dye_amount * color_g,
                    .sample_offset_x  = 0.5f,
                    .sample_offset_y  = 0.5f,
                    .sample_offset_z  = 0.5f,
                    .block_x          = stable_settings.desc.block_x,
                    .block_y          = stable_settings.desc.block_y,
                    .block_z          = stable_settings.desc.block_z,
                    .stream           = stable_runtime.sim_stream,
                };
                StableFluidsAddScalarSourceDesc dye_b_source_desc{
                    .struct_size      = sizeof(StableFluidsAddScalarSourceDesc),
                    .api_version      = STABLE_FLUIDS_API_VERSION,
                    .nx               = stable_settings.desc.nx,
                    .ny               = stable_settings.desc.ny,
                    .nz               = stable_settings.desc.nz,
                    .scalar           = stable_runtime.dye_b,
                    .center_x         = source_x,
                    .center_y         = source_y,
                    .center_z         = source_z,
                    .radius           = stable_settings.source_radius,
                    .amount           = stable_settings.dye_amount * color_b,
                    .sample_offset_x  = 0.5f,
                    .sample_offset_y  = 0.5f,
                    .sample_offset_z  = 0.5f,
                    .block_x          = stable_settings.desc.block_x,
                    .block_y          = stable_settings.desc.block_y,
                    .block_z          = stable_settings.desc.block_z,
                    .stream           = stable_runtime.sim_stream,
                };
                StableFluidsAddVectorSourceDesc vector_source_desc{
                    .struct_size      = sizeof(StableFluidsAddVectorSourceDesc),
                    .api_version      = STABLE_FLUIDS_API_VERSION,
                    .nx               = stable_settings.desc.nx,
                    .ny               = stable_settings.desc.ny,
                    .nz               = stable_settings.desc.nz,
                    .vector_x         = stable_runtime.velocity_x,
                    .vector_y         = stable_runtime.velocity_y,
                    .vector_z         = stable_runtime.velocity_z,
                    .center_x         = source_x,
                    .center_y         = source_y,
                    .center_z         = source_z,
                    .radius           = stable_settings.source_radius,
                    .amount_x         = velocity_x,
                    .amount_y         = velocity_y,
                    .amount_z         = velocity_z,
                    .block_x          = stable_settings.desc.block_x,
                    .block_y          = stable_settings.desc.block_y,
                    .block_z          = stable_settings.desc.block_z,
                    .stream           = stable_runtime.sim_stream,
                };
                if (const auto code = stable_fluids_add_scalar_source_cuda(&scalar_source_desc); code != 0) throw std::runtime_error("stable_fluids_add_scalar_source_cuda failed");
                if (const auto code = stable_fluids_add_scalar_source_cuda(&dye_r_source_desc); code != 0) throw std::runtime_error("stable_fluids_add_scalar_source_cuda dye_r failed");
                if (const auto code = stable_fluids_add_scalar_source_cuda(&dye_g_source_desc); code != 0) throw std::runtime_error("stable_fluids_add_scalar_source_cuda dye_g failed");
                if (const auto code = stable_fluids_add_scalar_source_cuda(&dye_b_source_desc); code != 0) throw std::runtime_error("stable_fluids_add_scalar_source_cuda dye_b failed");
                if (const auto code = stable_fluids_add_vector_source_cuda(&vector_source_desc); code != 0) throw std::runtime_error("stable_fluids_add_vector_source_cuda failed");
            };

            emit_one(left_x, stable_settings.source_a_r, stable_settings.source_a_g, stable_settings.source_a_b);
            emit_one(right_x, stable_settings.source_b_r, stable_settings.source_b_g, stable_settings.source_b_b);
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
            release(stable_runtime.dye_r);
            release(stable_runtime.dye_g);
            release(stable_runtime.dye_b);
            release(stable_runtime.velocity_x);
            release(stable_runtime.velocity_y);
            release(stable_runtime.velocity_z);
            release(stable_runtime.temporary_density);
            release(stable_runtime.temporary_dye_r);
            release(stable_runtime.temporary_dye_g);
            release(stable_runtime.temporary_dye_b);
            release(stable_runtime.temporary_velocity_x);
            release(stable_runtime.temporary_velocity_y);
            release(stable_runtime.temporary_velocity_z);
            release(stable_runtime.temporary_previous_density);
            release(stable_runtime.temporary_previous_dye_r);
            release(stable_runtime.temporary_previous_dye_g);
            release(stable_runtime.temporary_previous_dye_b);
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
            stable_runtime.dye_r                         = nullptr;
            stable_runtime.dye_g                         = nullptr;
            stable_runtime.dye_b                         = nullptr;
            stable_runtime.velocity_x                    = nullptr;
            stable_runtime.velocity_y                    = nullptr;
            stable_runtime.velocity_z                    = nullptr;
            stable_runtime.temporary_density             = nullptr;
            stable_runtime.temporary_dye_r               = nullptr;
            stable_runtime.temporary_dye_g               = nullptr;
            stable_runtime.temporary_dye_b               = nullptr;
            stable_runtime.temporary_velocity_x          = nullptr;
            stable_runtime.temporary_velocity_y          = nullptr;
            stable_runtime.temporary_velocity_z          = nullptr;
            stable_runtime.temporary_previous_density    = nullptr;
            stable_runtime.temporary_previous_dye_r      = nullptr;
            stable_runtime.temporary_previous_dye_g      = nullptr;
            stable_runtime.temporary_previous_dye_b      = nullptr;
            stable_runtime.temporary_previous_velocity_x = nullptr;
            stable_runtime.temporary_previous_velocity_y = nullptr;
            stable_runtime.temporary_previous_velocity_z = nullptr;
            stable_runtime.temporary_pressure            = nullptr;
            stable_runtime.temporary_divergence          = nullptr;
        };

        auto destroy_everything = [&]() {
            nvtx3::scoped_range range{"smoke_app.destroy_everything"};
            renderer.vk_context().device.waitIdle();
            destroy_snapshot_slots();
            destroy_stable_backend();
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

                cudaExternalMemoryHandleDesc external_desc{
                    .type = cudaExternalMemoryHandleTypeOpaqueWin32,
                    .handle =
                        {
                            .win32 =
                                {
                                    .handle = memory_handle,
                                },
                        },
                    .size = requirements.size,
                };
                if (const auto status = cudaImportExternalMemory(&slot.external_memory, &external_desc); status != cudaSuccess) throw std::runtime_error(std::string("cudaImportExternalMemory") + ": " + cudaGetErrorString(status));
                CloseHandle(memory_handle);

                vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                    .semaphore  = *slot.timeline_semaphore,
                    .handleType = semaphore_handle_type,
                };
                HANDLE semaphore_handle = renderer.vk_context().device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
                if (semaphore_handle == nullptr) throw std::runtime_error("getSemaphoreWin32HandleKHR returned a null handle");

                cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                    .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32,
                    .handle =
                        {
                            .win32 =
                                {
                                    .handle = semaphore_handle,
                                },
                        },
                };
                if (const auto status = cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc); status != cudaSuccess) throw std::runtime_error(std::string("cudaImportExternalSemaphore") + ": " + cudaGetErrorString(status));
                CloseHandle(semaphore_handle);
#else
                vk::MemoryGetFdInfoKHR handle_info{
                    .memory     = *slot.memory,
                    .handleType = memory_handle_type,
                };
                const int memory_fd = renderer.vk_context().device.getMemoryFdKHR(handle_info);
                if (memory_fd < 0) throw std::runtime_error("getMemoryFdKHR returned an invalid fd");

                cudaExternalMemoryHandleDesc external_desc{
                    .type = cudaExternalMemoryHandleTypeOpaqueFd,
                    .handle =
                        {
                            .fd = memory_fd,
                        },
                    .size = requirements.size,
                };
                if (const auto status = cudaImportExternalMemory(&slot.external_memory, &external_desc); status != cudaSuccess) throw std::runtime_error(std::string("cudaImportExternalMemory") + ": " + cudaGetErrorString(status));

                vk::SemaphoreGetFdInfoKHR semaphore_handle_info{
                    .semaphore  = *slot.timeline_semaphore,
                    .handleType = semaphore_handle_type,
                };
                const int semaphore_fd = renderer.vk_context().device.getSemaphoreFdKHR(semaphore_handle_info);
                if (semaphore_fd < 0) throw std::runtime_error("getSemaphoreFdKHR returned an invalid fd");

                cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                    .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
                    .handle =
                        {
                            .fd = semaphore_fd,
                        },
                };
                if (const auto status = cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc); status != cudaSuccess) throw std::runtime_error(std::string("cudaImportExternalSemaphore") + ": " + cudaGetErrorString(status));
#endif

                cudaExternalMemoryBufferDesc buffer_desc{
                    .offset = 0,
                    .size   = viewer_runtime.field_bytes,
                };
                if (const auto status = cudaExternalMemoryGetMappedBuffer(&slot.cuda_ptr, slot.external_memory, &buffer_desc); status != cudaSuccess) throw std::runtime_error(std::string("cudaExternalMemoryGetMappedBuffer") + ": " + cudaGetErrorString(status));

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

            if (field.id == FieldId::SmokeColor) {
                StableFluidsPackSmokeRgbaDesc pack_desc{
                    .struct_size = sizeof(StableFluidsPackSmokeRgbaDesc),
                    .api_version = STABLE_FLUIDS_API_VERSION,
                    .nx = static_cast<int32_t>(grid.nx),
                    .ny = static_cast<int32_t>(grid.ny),
                    .nz = static_cast<int32_t>(grid.nz),
                    .destination_rgba = slot.cuda_ptr,
                    .density = stable_runtime.density,
                    .dye_r = stable_runtime.dye_r,
                    .dye_g = stable_runtime.dye_g,
                    .dye_b = stable_runtime.dye_b,
                    .block_x = stable_settings.desc.block_x,
                    .block_y = stable_settings.desc.block_y,
                    .block_z = stable_settings.desc.block_z,
                    .stream = stable_runtime.sim_stream,
                };
                if (const auto code = stable_fluids_pack_smoke_rgba_cuda(&pack_desc); code != 0) throw std::runtime_error("stable_fluids_pack_smoke_rgba_cuda failed");
            } else if (field.id == FieldId::Density) {
                if (const auto status = cudaMemcpyAsync(slot.cuda_ptr, stable_runtime.density, scalar_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), cudaMemcpyDeviceToDevice, stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync stable density snapshot") + ": " + cudaGetErrorString(status));
            } else {
                StableFluidsComputeStaggeredVelocityMagnitudeDesc magnitude_desc{
                    .struct_size = sizeof(StableFluidsComputeStaggeredVelocityMagnitudeDesc),
                    .api_version = STABLE_FLUIDS_API_VERSION,
                    .nx = static_cast<int32_t>(grid.nx),
                    .ny = static_cast<int32_t>(grid.ny),
                    .nz = static_cast<int32_t>(grid.nz),
                    .destination = slot.cuda_ptr,
                    .velocity_x = stable_runtime.velocity_x,
                    .velocity_y = stable_runtime.velocity_y,
                    .velocity_z = stable_runtime.velocity_z,
                    .block_x = stable_settings.desc.block_x,
                    .block_y = stable_settings.desc.block_y,
                    .block_z = stable_settings.desc.block_z,
                    .stream = stable_runtime.sim_stream,
                };
                if (const auto code = stable_fluids_compute_staggered_velocity_magnitude_cuda(&magnitude_desc); code != 0) throw std::runtime_error("stable_fluids_compute_staggered_velocity_magnitude_cuda failed");
            }

            const uint64_t next_generation = viewer_runtime.snapshot_generation + 1;
            cudaExternalSemaphoreSignalParams signal_params{};
            signal_params.params.fence.value = next_generation;
            if (const auto status = cudaSignalExternalSemaphoresAsync(&slot.external_semaphore, &signal_params, 1, stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaSignalExternalSemaphoresAsync snapshot") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaStreamSynchronize(stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaStreamSynchronize snapshot") + ": " + cudaGetErrorString(status));
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
            auto& solver_stats                  = stable_solver_stats;
            accumulate_sample(elapsed_ms, solver_stats.last_snapshot_ms, solver_stats.average_snapshot_ms, solver_stats.snapshot_count);
        };

        auto reset_backend = [&]() {
            nvtx3::scoped_range range{"smoke_app.reset_backend"};

            const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
            if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) throw std::runtime_error("smoke-visualizer requires Vulkan timeline semaphore support");

            int cuda_device_index = 0;
            if (const auto status = cudaGetDevice(&cuda_device_index); status != cudaSuccess) throw std::runtime_error(std::string("cudaGetDevice") + ": " + cudaGetErrorString(status));
            int cuda_timeline_semaphore_interop_supported = 0;
            if (const auto status = cudaDeviceGetAttribute(&cuda_timeline_semaphore_interop_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index); status != cudaSuccess) throw std::runtime_error(std::string("cudaDeviceGetAttribute cudaDevAttrTimelineSemaphoreInteropSupported") + ": " + cudaGetErrorString(status));
            if (cuda_timeline_semaphore_interop_supported == 0) throw std::runtime_error("smoke-visualizer requires CUDA timeline semaphore interop support");

            destroy_everything();
            stable_solver_stats = {};

            viewer_runtime.field_bytes         = rgba_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
            const auto stable_scalar_bytes     = scalar_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
            const auto stable_velocity_x_bytes = velocity_x_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
            const auto stable_velocity_y_bytes = velocity_y_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
            const auto stable_velocity_z_bytes = velocity_z_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
            stable_runtime.device_allocated    = true;
            if (const auto status = cudaStreamCreateWithFlags(&stable_runtime.sim_stream, cudaStreamNonBlocking); status != cudaSuccess) throw std::runtime_error(std::string("cudaStreamCreateWithFlags stable_stream") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.density), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable density") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.dye_r), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable dye_r") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.dye_g), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable dye_g") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.dye_b), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable dye_b") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_x), stable_velocity_x_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable velocity_x") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_y), stable_velocity_y_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable velocity_y") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_z), stable_velocity_z_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable velocity_z") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_density), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_density") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_dye_r), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_dye_r") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_dye_g), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_dye_g") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_dye_b), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_dye_b") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_velocity_x), stable_velocity_x_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_velocity_x") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_velocity_y), stable_velocity_y_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_velocity_y") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_velocity_z), stable_velocity_z_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_velocity_z") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_density), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_previous_density") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_dye_r), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_previous_dye_r") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_dye_g), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_previous_dye_g") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_dye_b), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_previous_dye_b") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_velocity_x), stable_velocity_x_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_previous_velocity_x") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_velocity_y), stable_velocity_y_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_previous_velocity_y") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_previous_velocity_z), stable_velocity_z_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_previous_velocity_z") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_pressure), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_pressure") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMalloc(reinterpret_cast<void**>(&stable_runtime.temporary_divergence), stable_scalar_bytes); status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc stable temporary_divergence") + ": " + cudaGetErrorString(status));

            create_snapshot_slots();

            if (const auto status = cudaMemsetAsync(stable_runtime.density, 0, stable_scalar_bytes, stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync stable density") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMemsetAsync(stable_runtime.dye_r, 0, stable_scalar_bytes, stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync stable dye_r") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMemsetAsync(stable_runtime.dye_g, 0, stable_scalar_bytes, stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync stable dye_g") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMemsetAsync(stable_runtime.dye_b, 0, stable_scalar_bytes, stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync stable dye_b") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMemsetAsync(stable_runtime.velocity_x, 0, velocity_x_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync stable velocity_x") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMemsetAsync(stable_runtime.velocity_y, 0, velocity_y_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync stable velocity_y") + ": " + cudaGetErrorString(status));
            if (const auto status = cudaMemsetAsync(stable_runtime.velocity_z, 0, velocity_z_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz), stable_runtime.sim_stream); status != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync stable velocity_z") + ": " + cudaGetErrorString(status));
            if (stable_settings.emit_source) {
                inject_stable_dual_sources();
                run_stable_step();
            }

            snapshot_current_field_to_slot(0, "smoke_app.reset_backend.initial_snapshot");
            stable_solver_stats = {};
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
            const auto fields    = std::span<const FieldChoice>(stable_fields);
            auto& selected_field = stable_settings.selected_field;
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
                const auto& solver_stats = stable_solver_stats;
                ImGui::TextUnformatted("Solver");
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
                ImGui::Text("Step calls: %llu", static_cast<unsigned long long>(solver_stats.step_count));
                ImGui::Text("Snapshot commits: %llu", static_cast<unsigned long long>(solver_stats.snapshot_count));
                ImGui::Text("Last step call: %.3f ms", solver_stats.last_step_call_ms);
                ImGui::Text("Avg step call: %.3f ms", solver_stats.average_step_call_ms);
                ImGui::Text("Last snapshot: %.3f ms", solver_stats.last_snapshot_ms);
                ImGui::Text("Avg snapshot: %.3f ms", solver_stats.average_snapshot_ms);
                ImGui::TextUnformatted("CUDA timings are app-side call costs, not kernel profiler numbers.");
                ImGui::Separator();
            }

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
                    int boundary = std::clamp(static_cast<int>(value), 0, static_cast<int>(boundary_labels.size()) - 1);
                    if (ImGui::BeginCombo(label, boundary_labels[static_cast<size_t>(boundary)])) {
                        for (int i = 0; i < static_cast<int>(boundary_labels.size()); ++i) {
                            const bool is_selected = boundary == i;
                            if (ImGui::Selectable(boundary_labels[static_cast<size_t>(i)], is_selected)) {
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
                ImGui::SliderFloat("Inflow Vel X-", &stable_settings.desc.inflow_velocity_x_min, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Inflow Vel X+", &stable_settings.desc.inflow_velocity_x_max, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Inflow Vel Y-", &stable_settings.desc.inflow_velocity_y_min, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Inflow Vel Y+", &stable_settings.desc.inflow_velocity_y_max, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Inflow Vel Z-", &stable_settings.desc.inflow_velocity_z_min, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Inflow Vel Z+", &stable_settings.desc.inflow_velocity_z_max, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Inflow Density X-", &stable_settings.desc.inflow_scalar_x_min, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Inflow Density X+", &stable_settings.desc.inflow_scalar_x_max, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Inflow Density Y-", &stable_settings.desc.inflow_scalar_y_min, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Inflow Density Y+", &stable_settings.desc.inflow_scalar_y_max, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Inflow Density Z-", &stable_settings.desc.inflow_scalar_z_min, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Inflow Density Z+", &stable_settings.desc.inflow_scalar_z_max, 0.0f, 3.0f, "%.2f");
                ImGui::Separator();
                ImGui::TextUnformatted("Forces");
                ImGui::SliderFloat("Ambient Temp", &stable_settings.desc.ambient_temperature, -2.0f, 2.0f, "%.2f");
                ImGui::SliderFloat("Density Buoyancy (dv/step)", &stable_settings.desc.density_buoyancy, 0.0f, 2.0f, "%.4f");
                ImGui::BeginDisabled();
                ImGui::SliderFloat("Temperature Buoyancy (dv/step)", &stable_settings.desc.temperature_buoyancy, 0.0f, 2.0f, "%.4f");
                ImGui::EndDisabled();
                ImGui::SliderFloat("Uniform Force X (dv/step)", &stable_settings.desc.uniform_force_x, -2.0f, 2.0f, "%.3f");
                ImGui::SliderFloat("Uniform Force Y (dv/step)", &stable_settings.desc.uniform_force_y, -2.0f, 2.0f, "%.3f");
                ImGui::SliderFloat("Uniform Force Z (dv/step)", &stable_settings.desc.uniform_force_z, -2.0f, 2.0f, "%.3f");
                ImGui::TextUnformatted("Note: Temperature buoyancy requires a temperature field and is currently inactive in 001 app.");
                ImGui::Separator();
                ImGui::Checkbox("Emit Source", &stable_settings.emit_source);
                ImGui::TextUnformatted("Dual corner jets: left-bottom and right-bottom -> center");
                ImGui::SliderFloat("Corner Inset", &stable_settings.corner_inset, 0.04f, 0.32f, "%.2f");
                ImGui::SliderFloat("Source Height", &stable_settings.source_height, 0.03f, 0.24f, "%.2f");
                ImGui::SliderFloat("Source Depth", &stable_settings.source_depth, 0.04f, 0.32f, "%.2f");
                ImGui::SliderFloat("Source Radius", &stable_settings.source_radius, 1.0f, 16.0f, "%.1f");
                ImGui::SliderFloat("Density Amount", &stable_settings.density_amount, 0.0f, 2.0f, "%.2f");
                ImGui::SliderFloat("Dye Amount", &stable_settings.dye_amount, 0.0f, 2.0f, "%.2f");
                ImGui::ColorEdit3("Source A Dye", &stable_settings.source_a_r);
                ImGui::ColorEdit3("Source B Dye", &stable_settings.source_b_r);
                ImGui::SliderFloat("Jet Speed", &stable_settings.jet_speed, 0.2f, 200.0f, "%.2f");
                ImGui::SliderFloat("Upward Bias", &stable_settings.upward_bias, -1.0f, 2.0f, "%.2f");

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
                        if (stable_settings.emit_source) {
                            nvtx3::scoped_range range{"smoke_app.simulation.add_source"};
                            inject_stable_dual_sources();
                        }
                        {
                            nvtx3::scoped_range range{"smoke_app.simulation.step"};
                            const auto begin = std::chrono::steady_clock::now();
                            run_stable_step();
                            const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                            accumulate_sample(elapsed_ms, stable_solver_stats.last_step_call_ms, stable_solver_stats.average_step_call_ms, stable_solver_stats.step_count);
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
