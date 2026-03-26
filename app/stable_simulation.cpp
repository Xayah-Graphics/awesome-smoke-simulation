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

        [[nodiscard]] Settings make_settings_for_preset(const ScenePreset preset) {
            Settings settings{};
            settings.scene_preset = preset;
            settings.config.domain_boundary = {
                .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP), .velocity = 0.0f, },
                .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            };
            settings.density_diffusion = 0.00005f;
            settings.dye_diffusion = 0.00003f;
            settings.density_buoyancy = 0.35f;
            settings.emit_source = true;
            settings.collider = {
                .enabled = true,
                .type = 0,
                .center_x = 0.50f,
                .center_y = 0.50f,
                .center_z = 0.50f,
                .radius = 0.25f,
                .half_extent_x = 0.10f,
                .half_extent_y = 0.08f,
                .half_extent_z = 0.10f,
                .velocity_x = 0.0f,
                .velocity_y = 0.0f,
                .velocity_z = 0.0f,
                .boundary = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP),
            };

            if (preset == ScenePreset::SmokePlume) {
                settings.config.dt = 1.0f / 120.0f;
                settings.config.pressure_iterations = 96;
                settings.density_diffusion = 0.00002f;
                settings.dye_diffusion = 0.00001f;
                settings.density_buoyancy = 0.85f;
                settings.collider.enabled = false;
                settings.emitter_a = {
                    .enabled = true,
                    .position_x = 0.49f,
                    .position_y = 0.03f,
                    .position_z = 0.51f,
                    .direction_x = 0.06f,
                    .direction_y = 1.0f,
                    .direction_z = 0.03f,
                    .speed = 9.0f,
                    .radius_cells = 2.0f,
                    .density_amount = 1.0f,
                    .dye_amount = 0.12f,
                    .color_r = 0.92f,
                    .color_g = 0.92f,
                    .color_b = 0.92f,
                };
                settings.emitter_b = {
                    .enabled = false,
                    .position_x = 0.50f,
                    .position_y = 0.08f,
                    .position_z = 0.50f,
                    .direction_x = 0.0f,
                    .direction_y = 1.0f,
                    .direction_z = 0.0f,
                    .speed = 0.0f,
                    .radius_cells = 2.5f,
                    .density_amount = 0.0f,
                    .dye_amount = 0.0f,
                    .color_r = 1.0f,
                    .color_g = 1.0f,
                    .color_b = 1.0f,
                };
                return settings;
            }

            settings.scene_preset = ScenePreset::DualJetCollider;
            settings.emitter_a = {
                .enabled = true,
                .position_x = 0.16f,
                .position_y = 0.12f,
                .position_z = 0.78f,
                .direction_x = 0.57f,
                .direction_y = 0.64f,
                .direction_z = -0.52f,
                .speed = 52.0f,
                .radius_cells = 3.5f,
                .density_amount = 0.55f,
                .dye_amount = 0.65f,
                .color_r = 1.00f,
                .color_g = 0.20f,
                .color_b = 0.72f,
            };
            settings.emitter_b = {
                .enabled = true,
                .position_x = 0.84f,
                .position_y = 0.12f,
                .position_z = 0.22f,
                .direction_x = -0.57f,
                .direction_y = 0.64f,
                .direction_z = 0.52f,
                .speed = 52.0f,
                .radius_cells = 3.5f,
                .density_amount = 0.55f,
                .dye_amount = 0.65f,
                .color_r = 0.12f,
                .color_g = 0.38f,
                .color_b = 1.00f,
            };
            return settings;
        }

    } // namespace

    StableSimulation::StableSimulation() {
        settings_ = make_settings_for_preset(ScenePreset::DualJetCollider);
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

    ScenePreset StableSimulation::scene_preset() const {
        return settings_.scene_preset;
    }

    void StableSimulation::apply_scene_preset(const ScenePreset preset) {
        const int selected_field = settings_.selected_field;
        settings_ = make_settings_for_preset(preset);
        settings_.selected_field = selected_field;
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
                .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
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
                .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
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
            .velocity_boundary_type = settings_.collider.boundary,
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
        const auto h = settings_.config.cell_size;
        std::array<StableFluidsVelocitySourceDesc, 2> velocity_sources{};
        std::array<StableFluidsFieldSourceDesc, 4> field_sources{};
        uint32_t velocity_source_count = 0;
        uint32_t field_source_count = 0;
        const auto append_emitter = [&](const SourceEmitterSettings& emitter) {
            if (!emitter.enabled) return;
            const float dir_len = std::sqrt(emitter.direction_x * emitter.direction_x + emitter.direction_y * emitter.direction_y + emitter.direction_z * emitter.direction_z);
            const float inv_len = dir_len > 1.0e-5f ? 1.0f / dir_len : 1.0f;
            const float dir_x = dir_len > 1.0e-5f ? emitter.direction_x * inv_len : 0.0f;
            const float dir_y = dir_len > 1.0e-5f ? emitter.direction_y * inv_len : 1.0f;
            const float dir_z = dir_len > 1.0e-5f ? emitter.direction_z * inv_len : 0.0f;
            velocity_sources[velocity_source_count++] = {
                .center_x = emitter.position_x * nx * h,
                .center_y = emitter.position_y * static_cast<float>(settings_.config.ny) * h,
                .center_z = emitter.position_z * static_cast<float>(settings_.config.nz) * h,
                .radius = emitter.radius_cells * h,
                .velocity_x = dir_x * emitter.speed,
                .velocity_y = dir_y * emitter.speed,
                .velocity_z = dir_z * emitter.speed,
            };
            field_sources[field_source_count++] = {
                .field = density_field_,
                .center_x = emitter.position_x * nx * h,
                .center_y = emitter.position_y * static_cast<float>(settings_.config.ny) * h,
                .center_z = emitter.position_z * static_cast<float>(settings_.config.nz) * h,
                .radius = emitter.radius_cells * h,
                .value_0 = emitter.density_amount,
                .value_1 = 0.0f,
                .value_2 = 0.0f,
                .value_3 = 0.0f,
            };
            field_sources[field_source_count++] = {
                .field = dye_field_,
                .center_x = emitter.position_x * nx * h,
                .center_y = emitter.position_y * static_cast<float>(settings_.config.ny) * h,
                .center_z = emitter.position_z * static_cast<float>(settings_.config.nz) * h,
                .radius = emitter.radius_cells * h,
                .value_0 = emitter.dye_amount * emitter.color_r,
                .value_1 = emitter.dye_amount * emitter.color_g,
                .value_2 = emitter.dye_amount * emitter.color_b,
                .value_3 = 0.0f,
            };
        };
        append_emitter(settings_.emitter_a);
        append_emitter(settings_.emitter_b);

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            StableFluidsStepDesc step_desc{
                .velocity_sources = settings_.emit_source ? velocity_sources.data() : nullptr,
                .velocity_source_count = settings_.emit_source ? velocity_source_count : 0u,
                .field_sources = settings_.emit_source ? field_sources.data() : nullptr,
                .field_source_count = settings_.emit_source ? field_source_count : 0u,
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

    void StableSimulation::export_velocity(void* destination) const {
        check_stable(stable_fluids_export_velocity_cuda(context_, destination), "stable_fluids_export_velocity_cuda");
    }

    StableFluidsGridDesc StableSimulation::grid_desc() const {
        StableFluidsGridDesc desc{};
        check_stable(stable_fluids_get_grid_desc_cuda(context_, &desc), "stable_fluids_get_grid_desc_cuda");
        return desc;
    }

} // namespace smoke
