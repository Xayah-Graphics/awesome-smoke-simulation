#pragma once

#include "stable-fluids.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace smoke {

    enum class FieldId : uint32_t {
        SmokeColor        = 0,
        Density           = 1,
        VelocityMagnitude = 2,
        SolidMask         = 3,
        Pressure          = 4,
        Divergence        = 5,
    };

    struct SolverStats {
        double last_step_call_ms    = 0.0;
        double average_step_call_ms = 0.0;
        uint64_t step_count         = 0;
    };

    struct ColliderSettings {
        bool enabled          = true;
        int type              = 0;
        float center_x        = 0.50f;
        float center_y        = 0.50f;
        float center_z        = 0.50f;
        float radius          = 0.25f;
        float half_extent_x   = 0.10f;
        float half_extent_y   = 0.08f;
        float half_extent_z   = 0.10f;
        float velocity_x      = 0.0f;
        float velocity_y      = 0.0f;
        float velocity_z      = 0.0f;
        uint32_t boundary     = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_NO_SLIP);
    };

    struct Settings {
        StableFluidsSimulationConfig config{
            .nx = 128,
            .ny = 128,
            .nz = 128,
            .cell_size = 1.0f,
            .dt = 1.0f / 90.0f,
            .viscosity = 0.00015f,
            .diffuse_iterations = 24,
            .pressure_iterations = 80,
            .uniform_force_x = 0.0f,
            .uniform_force_y = 0.0f,
            .uniform_force_z = 0.0f,
            .domain_boundary = {
                .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            },
            .block_x = 8,
            .block_y = 8,
            .block_z = 4,
        };
        float density_diffusion = 0.00005f;
        float dye_diffusion     = 0.00003f;
        float density_buoyancy  = 0.35f;
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
        float source_a_x     = 0.16f;
        float source_a_y     = 0.12f;
        float source_a_z     = 0.22f;
        float source_b_x     = 0.84f;
        float source_b_y     = 0.12f;
        float source_b_z     = 0.22f;
        float focus_x        = 0.50f;
        float focus_y        = 0.50f;
        float focus_z        = 0.50f;
        float jet_speed      = 52.0f;
        float upward_bias    = 0.0f;
        ColliderSettings collider{};
    };

    class StableSimulation {
    public:
        StableSimulation();
        ~StableSimulation();

        StableSimulation(const StableSimulation&) = delete;
        StableSimulation& operator=(const StableSimulation&) = delete;

        Settings& settings();
        [[nodiscard]] const Settings& settings() const;
        [[nodiscard]] const SolverStats& stats() const;
        [[nodiscard]] cudaStream_t stream() const;

        void rebuild();
        void step(int sim_steps);
        void export_field(FieldId field, void* destination) const;
        [[nodiscard]] StableFluidsGridDesc grid_desc() const;

    private:
        void update_scene();

        Settings settings_{};
        SolverStats stats_{};
        cudaStream_t stream_        = nullptr;
        StableFluidsContext context_ = nullptr;
        StableFluidsFieldHandle density_field_ = 0;
        StableFluidsFieldHandle dye_field_     = 0;
    };

} // namespace smoke
