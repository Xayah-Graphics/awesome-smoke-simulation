#include "stable_simulation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace smoke {

    namespace {

        void check_cuda(const cudaError_t status) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string("cudaStreamCreateWithFlags") + ": " + cudaGetErrorString(status));
        }

        void check_stable(const StableFluidsResult code, const char* what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        }

    } // namespace

    StableSimulation::StableSimulation() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        rebuild();
    }

    StableSimulation::~StableSimulation() {
        if (context_ != nullptr) stable_fluids_destroy_context_cuda(context_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    Settings& StableSimulation::settings() {
        return settings_;
    }

    const Settings& StableSimulation::settings() const {
        return settings_;
    }

    const SolverStats& StableSimulation::stats() const {
        return stats_;
    }

    cudaStream_t StableSimulation::stream() const {
        return stream_;
    }

    void StableSimulation::rebuild() {
        if (context_ != nullptr) {
            check_stable(stable_fluids_destroy_context_cuda(context_), "stable_fluids_destroy_context_cuda");
            context_ = nullptr;
        }
        density_field_ = 0;
        dye_field_ = 0;

        std::array fields{
            StableFluidsFieldCreateDesc{
                .name = "density",
                .component_count = 1,
                .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
                .diffusion = settings_.density_diffusion,
                .boundary_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_BOUNDARY_STREAK),
                .default_value_0 = 0.0f,
                .default_value_1 = 0.0f,
                .default_value_2 = 0.0f,
                .default_value_3 = 0.0f,
            },
            StableFluidsFieldCreateDesc{
                .name = "dye",
                .component_count = 3,
                .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
                .diffusion = settings_.dye_diffusion,
                .boundary_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_BOUNDARY_STREAK),
                .default_value_0 = 0.0f,
                .default_value_1 = 0.0f,
                .default_value_2 = 0.0f,
                .default_value_3 = 0.0f,
            },
        };
        std::array buoyancy_terms{
            StableFluidsBuoyancyDesc{
                .field_index = 0,
                .weight = settings_.density_buoyancy,
                .ambient = 0.0f,
            },
        };
        std::array<StableFluidsFieldHandle, 2> field_handles{};

        StableFluidsContextCreateDesc create_desc{
            .config = settings_.config,
            .stream = stream_,
            .fields = fields.data(),
            .field_count = static_cast<uint32_t>(fields.size()),
            .buoyancy_terms = buoyancy_terms.data(),
            .buoyancy_term_count = static_cast<uint32_t>(buoyancy_terms.size()),
        };
        check_stable(stable_fluids_create_context_cuda(&create_desc, &context_, field_handles.data(), static_cast<uint32_t>(field_handles.size())), "stable_fluids_create_context_cuda");
        density_field_ = field_handles[0];
        dye_field_ = field_handles[1];
        update_scene();
        stats_ = {};
    }

    void StableSimulation::update_scene() {
        StableFluidsSceneDesc scene_desc{
            .colliders = nullptr,
            .collider_count = 0,
        };

        StableFluidsColliderDesc collider{
            .collider_type = static_cast<uint32_t>(settings_.collider.type == 0 ? STABLE_FLUIDS_COLLIDER_SPHERE : STABLE_FLUIDS_COLLIDER_BOX),
            .boundary_type = settings_.collider.boundary,
            .center_x = settings_.collider.center_x * static_cast<float>(settings_.config.nx) * settings_.config.cell_size,
            .center_y = settings_.collider.center_y * static_cast<float>(settings_.config.ny) * settings_.config.cell_size,
            .center_z = settings_.collider.center_z * static_cast<float>(settings_.config.nz) * settings_.config.cell_size,
            .radius = settings_.collider.radius * static_cast<float>((std::max) ({settings_.config.nx, settings_.config.ny, settings_.config.nz})) * settings_.config.cell_size,
            .half_extent_x = settings_.collider.half_extent_x * static_cast<float>(settings_.config.nx) * settings_.config.cell_size,
            .half_extent_y = settings_.collider.half_extent_y * static_cast<float>(settings_.config.ny) * settings_.config.cell_size,
            .half_extent_z = settings_.collider.half_extent_z * static_cast<float>(settings_.config.nz) * settings_.config.cell_size,
            .linear_velocity_x = settings_.collider.velocity_x,
            .linear_velocity_y = settings_.collider.velocity_y,
            .linear_velocity_z = settings_.collider.velocity_z,
        };

        if (settings_.collider.enabled) {
            scene_desc.colliders = &collider;
            scene_desc.collider_count = 1;
        }
        check_stable(stable_fluids_update_scene_cuda(context_, &scene_desc), "stable_fluids_update_scene_cuda");
    }

    void StableSimulation::step(const int sim_steps) {
        const auto nx = static_cast<float>(settings_.config.nx);
        const auto ny = static_cast<float>(settings_.config.ny);
        const auto nz = static_cast<float>(settings_.config.nz);
        const auto h = settings_.config.cell_size;
        const float focus_x = settings_.focus_x * nx;
        const float focus_y = settings_.focus_y * ny;
        const float focus_z = settings_.focus_z * nz;
        const auto make_source = [&](const float source_x, const float source_y, const float source_z, const float color_r, const float color_g, const float color_b) {
            const float dir_x = focus_x - source_x;
            const float dir_y = focus_y - source_y;
            const float dir_z = focus_z - source_z;
            const float inv_len = 1.0f / (std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z) + 1.0e-6f);
            return std::pair{
                StableFluidsVelocitySourceDesc{
                    .center_x = source_x * h,
                    .center_y = source_y * h,
                    .center_z = source_z * h,
                    .radius = settings_.source_radius * h,
                    .velocity_x = dir_x * inv_len * settings_.jet_speed,
                    .velocity_y = dir_y * inv_len * settings_.jet_speed + settings_.upward_bias,
                    .velocity_z = dir_z * inv_len * settings_.jet_speed,
                },
                std::array{
                    StableFluidsFieldSourceDesc{
                        .field = density_field_,
                        .center_x = source_x * h,
                        .center_y = source_y * h,
                        .center_z = source_z * h,
                        .radius = settings_.source_radius * h,
                        .value_0 = settings_.density_amount,
                        .value_1 = 0.0f,
                        .value_2 = 0.0f,
                        .value_3 = 0.0f,
                    },
                    StableFluidsFieldSourceDesc{
                        .field = dye_field_,
                        .center_x = source_x * h,
                        .center_y = source_y * h,
                        .center_z = source_z * h,
                        .radius = settings_.source_radius * h,
                        .value_0 = settings_.dye_amount * color_r,
                        .value_1 = settings_.dye_amount * color_g,
                        .value_2 = settings_.dye_amount * color_b,
                        .value_3 = 0.0f,
                    },
                },
            };
        };
        const std::array sources{
            make_source(settings_.source_a_x * nx, settings_.source_a_y * ny, settings_.source_a_z * nz, settings_.source_a_r, settings_.source_a_g, settings_.source_a_b),
            make_source(settings_.source_b_x * nx, settings_.source_b_y * ny, settings_.source_b_z * nz, settings_.source_b_r, settings_.source_b_g, settings_.source_b_b),
        };
        const std::array velocity_sources{
            sources[0].first,
            sources[1].first,
        };
        const std::array field_sources{
            sources[0].second[0],
            sources[0].second[1],
            sources[1].second[0],
            sources[1].second[1],
        };

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            StableFluidsStepDesc step_desc{
                .velocity_sources = settings_.emit_source ? velocity_sources.data() : nullptr,
                .velocity_source_count = settings_.emit_source ? static_cast<uint32_t>(velocity_sources.size()) : 0u,
                .field_sources = settings_.emit_source ? field_sources.data() : nullptr,
                .field_source_count = settings_.emit_source ? static_cast<uint32_t>(field_sources.size()) : 0u,
            };
            const auto begin = std::chrono::steady_clock::now();
            check_stable(stable_fluids_step_cuda(context_, &step_desc), "stable_fluids_step_cuda");
            const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            stats_.last_step_call_ms = elapsed_ms;
            ++stats_.step_count;
            stats_.average_step_call_ms += (elapsed_ms - stats_.average_step_call_ms) / static_cast<double>(stats_.step_count);
        }
    }

    void StableSimulation::export_field(const FieldId field, void* destination) const {
        if (field == FieldId::SmokeColor) {
            check_stable(stable_fluids_export_alpha_rgb_rgba_cuda(context_, density_field_, dye_field_, destination), "stable_fluids_export_alpha_rgb_rgba_cuda");
            return;
        }
        if (field == FieldId::Density) {
            check_stable(stable_fluids_export_field_components_cuda(context_, density_field_, 0, 1, destination), "stable_fluids_export_field_components_cuda");
            return;
        }
        if (field == FieldId::VelocityMagnitude) {
            check_stable(stable_fluids_export_velocity_magnitude_cuda(context_, destination), "stable_fluids_export_velocity_magnitude_cuda");
            return;
        }
        if (field == FieldId::SolidMask) {
            check_stable(stable_fluids_export_solid_mask_cuda(context_, destination), "stable_fluids_export_solid_mask_cuda");
            return;
        }
        if (field == FieldId::Pressure) {
            check_stable(stable_fluids_export_pressure_cuda(context_, destination), "stable_fluids_export_pressure_cuda");
            return;
        }
        check_stable(stable_fluids_export_divergence_cuda(context_, destination), "stable_fluids_export_divergence_cuda");
    }

    StableFluidsGridDesc StableSimulation::grid_desc() const {
        StableFluidsGridDesc desc{};
        check_stable(stable_fluids_get_grid_desc_cuda(context_, &desc), "stable_fluids_get_grid_desc_cuda");
        return desc;
    }

} // namespace smoke
