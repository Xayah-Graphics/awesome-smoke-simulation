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

        void check_stable(const int32_t code, const char* what) {
            if (code == 0) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(code) + ")");
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

        StableFluidsContextCreateDesc create_desc{
            .struct_size = sizeof(StableFluidsContextCreateDesc),
            .api_version = STABLE_FLUIDS_API_VERSION,
            .config = settings_.config,
            .stream = stream_,
        };
        check_stable(stable_fluids_create_context_cuda(&create_desc, &context_), "stable_fluids_create_context_cuda");
        update_scene();
        stats_ = {};
    }

    void StableSimulation::update_scene() {
        StableFluidsSceneDesc scene_desc{
            .struct_size = sizeof(StableFluidsSceneDesc),
            .api_version = STABLE_FLUIDS_API_VERSION,
            .colliders = nullptr,
            .collider_count = 0,
        };

        StableFluidsColliderDesc collider{
            .struct_size = sizeof(StableFluidsColliderDesc),
            .api_version = STABLE_FLUIDS_API_VERSION,
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
        const float focus_x = settings_.focus_x * nx;
        const float focus_y = settings_.focus_y * ny;
        const float focus_z = settings_.focus_z * nz;
        const auto make_source = [&](const float source_x, const float source_y, const float source_z, const float color_r, const float color_g, const float color_b) {
            const float dir_x = focus_x - source_x;
            const float dir_y = focus_y - source_y;
            const float dir_z = focus_z - source_z;
            const float inv_len = 1.0f / (std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z) + 1.0e-6f);
            return StableFluidsSourceDesc{
                .struct_size = sizeof(StableFluidsSourceDesc),
                .api_version = STABLE_FLUIDS_API_VERSION,
                .center_x = source_x,
                .center_y = source_y,
                .center_z = source_z,
                .radius = settings_.source_radius,
                .density_amount = settings_.density_amount,
                .dye_r = settings_.dye_amount * color_r,
                .dye_g = settings_.dye_amount * color_g,
                .dye_b = settings_.dye_amount * color_b,
                .temperature_amount = 0.0f,
                .velocity_x = dir_x * inv_len * settings_.jet_speed,
                .velocity_y = dir_y * inv_len * settings_.jet_speed + settings_.upward_bias,
                .velocity_z = dir_z * inv_len * settings_.jet_speed,
            };
        };
        const std::array sources{
            make_source(settings_.source_a_x * nx, settings_.source_a_y * ny, settings_.source_a_z * nz, settings_.source_a_r, settings_.source_a_g, settings_.source_a_b),
            make_source(settings_.source_b_x * nx, settings_.source_b_y * ny, settings_.source_b_z * nz, settings_.source_b_r, settings_.source_b_g, settings_.source_b_b),
        };

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            StableFluidsStepDesc step_desc{
                .struct_size = sizeof(StableFluidsStepDesc),
                .api_version = STABLE_FLUIDS_API_VERSION,
                .sources = settings_.emit_source ? sources.data() : nullptr,
                .source_count = settings_.emit_source ? static_cast<uint32_t>(sources.size()) : 0u,
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
        auto export_field = static_cast<uint32_t>(STABLE_FLUIDS_EXPORT_DYE_RGBA);
        if (field == FieldId::Density) export_field = static_cast<uint32_t>(STABLE_FLUIDS_EXPORT_DENSITY);
        if (field == FieldId::VelocityMagnitude) export_field = static_cast<uint32_t>(STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE);
        if (field == FieldId::SolidMask) export_field = static_cast<uint32_t>(STABLE_FLUIDS_EXPORT_SOLID_MASK);
        if (field == FieldId::Pressure) export_field = static_cast<uint32_t>(STABLE_FLUIDS_EXPORT_PRESSURE);
        if (field == FieldId::Divergence) export_field = static_cast<uint32_t>(STABLE_FLUIDS_EXPORT_DIVERGENCE);

        StableFluidsExportFieldDesc export_desc{
            .struct_size = sizeof(StableFluidsExportFieldDesc),
            .api_version = STABLE_FLUIDS_API_VERSION,
            .field = export_field,
            .destination = destination,
        };
        check_stable(stable_fluids_export_field_cuda(context_, &export_desc), "stable_fluids_export_field_cuda");
    }

    StableFluidsGridDesc StableSimulation::grid_desc() const {
        StableFluidsGridDesc desc{
            .struct_size = sizeof(StableFluidsGridDesc),
            .api_version = STABLE_FLUIDS_API_VERSION,
        };
        check_stable(stable_fluids_get_grid_desc_cuda(context_, &desc), "stable_fluids_get_grid_desc_cuda");
        return desc;
    }

} // namespace smoke
