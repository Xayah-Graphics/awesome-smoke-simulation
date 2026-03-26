#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include "stable_simulation.hpp"

#include <cuda_runtime.h>
#include <imgui.h>
#include <nvtx3/nvtx3.hpp>
#include <vulkan/vulkan_raii.hpp>

import app;
import std;
import vk.memory;

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    struct PlaybackState {
        bool paused             = false;
        bool step_once          = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval   = 2;
    };

    struct FieldChoice {
        smoke::FieldId id;
        std::string_view label;
        app::FieldSemantic semantic;
    };

    constexpr std::array fields{
        FieldChoice{smoke::FieldId::SmokeColor, "Smoke Color", app::FieldSemantic::DyeColor},
        FieldChoice{smoke::FieldId::Density, "Density", app::FieldSemantic::Density},
        FieldChoice{smoke::FieldId::VelocityMagnitude, "Velocity Magnitude", app::FieldSemantic::VelocityMagnitude},
        FieldChoice{smoke::FieldId::SolidMask, "Solid Mask", app::FieldSemantic::GenericScalar},
        FieldChoice{smoke::FieldId::Pressure, "Pressure", app::FieldSemantic::GenericScalar},
        FieldChoice{smoke::FieldId::Divergence, "Divergence", app::FieldSemantic::GenericScalar},
    };

    constexpr std::array boundary_labels{
        "No-slip",
        "Free-slip",
        "Inflow",
        "Outflow",
    };

    constexpr std::array collider_type_labels{
        "Sphere",
        "Box",
    };

    struct ViewerRuntime {
        uint64_t field_bytes          = 0;
        uint64_t snapshot_generation  = 0;
        uint64_t submit_serial        = 0;
        uint32_t steps_since_snapshot = 0;
        int active_snapshot_slot      = -1;
        double last_snapshot_ms       = 0.0;
        double average_snapshot_ms    = 0.0;
        uint64_t snapshot_count       = 0;
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

} // namespace

int main() {
    try {
        app::FieldRendererApp renderer;
        PlaybackState playback{};
        smoke::StableSimulation simulation{};
        ViewerRuntime viewer_runtime{};
        std::vector<SnapshotSlot> snapshot_slots{};

        auto rgba_bytes = [](const StableFluidsGridDesc& grid) {
            return static_cast<uint64_t>(grid.nx) * static_cast<uint64_t>(grid.ny) * static_cast<uint64_t>(grid.nz) * sizeof(float) * 4ull;
        };
        auto current_field = [&]() -> const FieldChoice& {
            auto& selected = simulation.settings().selected_field;
            selected       = std::clamp(selected, 0, static_cast<int>(fields.size()) - 1);
            return fields[static_cast<size_t>(selected)];
        };
        auto apply_field_defaults = [&](const FieldChoice& field) {
            auto& render = renderer.render_settings();
            if (field.id == smoke::FieldId::SmokeColor) {
                render.mode = app::RenderMode::Smoke;
                render.march_steps = 96;
                render.density_scale = 0.95f;
                render.absorption = 1.20f;
            } else if (field.id == smoke::FieldId::Density) {
                render.mode = app::RenderMode::Scalar;
                render.scalar_min = 0.0f;
                render.scalar_max = 0.70f;
                render.scalar_opacity = 2.1f;
                render.scalar_low_r = 0.10f;
                render.scalar_low_g = 0.08f;
                render.scalar_low_b = 0.30f;
                render.scalar_high_r = 1.00f;
                render.scalar_high_g = 0.24f;
                render.scalar_high_b = 0.74f;
            } else if (field.id == smoke::FieldId::SolidMask) {
                render.mode = app::RenderMode::Scalar;
                render.scalar_min = 0.0f;
                render.scalar_max = 1.0f;
                render.scalar_opacity = 3.2f;
                render.scalar_low_r = 0.05f;
                render.scalar_low_g = 0.06f;
                render.scalar_low_b = 0.07f;
                render.scalar_high_r = 0.94f;
                render.scalar_high_g = 0.92f;
                render.scalar_high_b = 0.88f;
            } else {
                render.mode = app::RenderMode::Scalar;
                render.scalar_min = field.id == smoke::FieldId::Divergence ? -2.0f : 0.0f;
                render.scalar_max = field.id == smoke::FieldId::VelocityMagnitude ? 3.0f : 1.5f;
                render.scalar_opacity = 1.4f;
                render.scalar_low_r = 0.06f;
                render.scalar_low_g = 0.10f;
                render.scalar_low_b = 0.42f;
                render.scalar_high_r = 0.18f;
                render.scalar_high_g = 0.88f;
                render.scalar_high_b = 1.00f;
            }
        };
        auto accumulate_snapshot = [&](const double sample_ms) {
            viewer_runtime.last_snapshot_ms = sample_ms;
            ++viewer_runtime.snapshot_count;
            viewer_runtime.average_snapshot_ms += (sample_ms - viewer_runtime.average_snapshot_ms) / static_cast<double>(viewer_runtime.snapshot_count);
        };

        auto active_snapshot = [&]() -> std::optional<app::ScalarFieldView> {
            if (viewer_runtime.active_snapshot_slot < 0) return std::nullopt;
            const auto& slot = snapshot_slots.at(static_cast<size_t>(viewer_runtime.active_snapshot_slot));
            const auto& settings = simulation.settings();
            const float nx = static_cast<float>(settings.config.nx) * settings.config.cell_size;
            const float ny = static_cast<float>(settings.config.ny) * settings.config.cell_size;
            const float nz = static_cast<float>(settings.config.nz) * settings.config.cell_size;
            const float max_extent = static_cast<float>((std::max) ({settings.config.nx, settings.config.ny, settings.config.nz})) * settings.config.cell_size;
            return app::ScalarFieldView{
                .descriptor_set = *slot.descriptor_set,
                .timeline_semaphore = slot.external_semaphore != nullptr ? *slot.timeline_semaphore : vk::Semaphore{},
                .ready_generation = slot.ready_generation,
                .nx = slot.nx,
                .ny = slot.ny,
                .nz = slot.nz,
                .cell_size = slot.cell_size,
                .semantic = slot.semantic,
                .label = slot.label,
                .collider_enabled = settings.collider.enabled,
                .collider_type = static_cast<uint32_t>(settings.collider.type),
                .collider_center_x = settings.collider.center_x * nx,
                .collider_center_y = settings.collider.center_y * ny,
                .collider_center_z = settings.collider.center_z * nz,
                .collider_radius = settings.collider.radius * max_extent,
                .collider_half_x = settings.collider.half_extent_x * nx,
                .collider_half_y = settings.collider.half_extent_y * ny,
                .collider_half_z = settings.collider.half_extent_z * nz,
            };
        };

        auto destroy_snapshot_slots = [&]() {
            for (auto& slot : snapshot_slots) {
                if (slot.cuda_ptr != nullptr) cudaFree(slot.cuda_ptr);
                if (slot.external_semaphore != nullptr) cudaDestroyExternalSemaphore(slot.external_semaphore);
                if (slot.external_memory != nullptr) cudaDestroyExternalMemory(slot.external_memory);
                slot = {};
            }
            snapshot_slots.clear();
            viewer_runtime.field_bytes = 0;
            viewer_runtime.snapshot_generation = 0;
            viewer_runtime.submit_serial = 0;
            viewer_runtime.steps_since_snapshot = 0;
            viewer_runtime.active_snapshot_slot = -1;
            viewer_runtime.last_snapshot_ms = 0.0;
            viewer_runtime.average_snapshot_ms = 0.0;
            viewer_runtime.snapshot_count = 0;
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
                    .initialValue = 0,
                };
                vk::ExportSemaphoreCreateInfo export_semaphore_ci{
                    .pNext = &timeline_semaphore_ci,
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
                    .pNext = &external_buffer_ci,
                    .size = viewer_runtime.field_bytes,
                    .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                    .sharingMode = vk::SharingMode::eExclusive,
                };
                slot.buffer = vk::raii::Buffer{renderer.vk_context().device, buffer_ci};
                const vk::MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
                vk::ExportMemoryAllocateInfo export_memory_ci{
                    .handleTypes = memory_handle_type,
                };
                vk::MemoryAllocateInfo alloc_ci{
                    .pNext = &export_memory_ci,
                    .allocationSize = requirements.size,
                    .memoryTypeIndex = vk::memory::find_memory_type(renderer.vk_context().physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
                };
                slot.memory = vk::raii::DeviceMemory{renderer.vk_context().device, alloc_ci};
                slot.buffer.bindMemory(*slot.memory, 0);

#if defined(_WIN32)
                vk::MemoryGetWin32HandleInfoKHR handle_info{
                    .memory = *slot.memory,
                    .handleType = memory_handle_type,
                };
                HANDLE memory_handle = renderer.vk_context().device.getMemoryWin32HandleKHR(handle_info);
                cudaExternalMemoryHandleDesc external_desc{
                    .type = cudaExternalMemoryHandleTypeOpaqueWin32,
                    .handle = { .win32 = { .handle = memory_handle, }, },
                    .size = requirements.size,
                };
                if (cudaImportExternalMemory(&slot.external_memory, &external_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalMemory failed");
                CloseHandle(memory_handle);

                vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                    .semaphore = *slot.timeline_semaphore,
                    .handleType = semaphore_handle_type,
                };
                HANDLE semaphore_handle = renderer.vk_context().device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
                cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                    .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32,
                    .handle = { .win32 = { .handle = semaphore_handle, }, },
                };
                if (cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalSemaphore failed");
                CloseHandle(semaphore_handle);
#else
                vk::MemoryGetFdInfoKHR handle_info{
                    .memory = *slot.memory,
                    .handleType = memory_handle_type,
                };
                const int memory_fd = renderer.vk_context().device.getMemoryFdKHR(handle_info);
                cudaExternalMemoryHandleDesc external_desc{
                    .type = cudaExternalMemoryHandleTypeOpaqueFd,
                    .handle = { .fd = memory_fd, },
                    .size = requirements.size,
                };
                if (cudaImportExternalMemory(&slot.external_memory, &external_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalMemory failed");

                vk::SemaphoreGetFdInfoKHR semaphore_handle_info{
                    .semaphore = *slot.timeline_semaphore,
                    .handleType = semaphore_handle_type,
                };
                const int semaphore_fd = renderer.vk_context().device.getSemaphoreFdKHR(semaphore_handle_info);
                cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                    .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
                    .handle = { .fd = semaphore_fd, },
                };
                if (cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalSemaphore failed");
#endif

                cudaExternalMemoryBufferDesc buffer_desc{
                    .offset = 0,
                    .size = viewer_runtime.field_bytes,
                };
                if (cudaExternalMemoryGetMappedBuffer(&slot.cuda_ptr, slot.external_memory, &buffer_desc) != cudaSuccess) throw std::runtime_error("cudaExternalMemoryGetMappedBuffer failed");

                vk::DescriptorBufferInfo field_info{
                    .buffer = *slot.buffer,
                    .offset = 0,
                    .range = viewer_runtime.field_bytes,
                };
                vk::WriteDescriptorSet field_write{
                    .dstSet = *slot.descriptor_set,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &field_info,
                };
                renderer.vk_context().device.updateDescriptorSets(field_write, {});
                snapshot_slots.push_back(std::move(slot));
            }
        };

        auto find_available_snapshot_slot = [&]() -> int {
            for (uint32_t slot_index = 0; slot_index < snapshot_slots.size(); ++slot_index) {
                auto& slot = snapshot_slots[slot_index];
                if (static_cast<int>(slot_index) == viewer_runtime.active_snapshot_slot) continue;
                if (slot.ready_generation != 0 && viewer_runtime.submit_serial < slot.last_used_submit_serial + renderer.frames_in_flight() + 1) continue;
                return static_cast<int>(slot_index);
            }
            return -1;
        };

        auto snapshot_current_field_to_slot = [&](const int slot_index, const char* tag) {
            nvtx3::scoped_range range{tag};
            const auto begin = std::chrono::steady_clock::now();
            auto& slot = snapshot_slots.at(static_cast<size_t>(slot_index));
            const auto grid = simulation.grid_desc();
            const auto& field = current_field();
            simulation.export_field(field.id, slot.cuda_ptr);
            cudaExternalSemaphoreSignalParams signal_params{};
            const uint64_t next_generation = viewer_runtime.snapshot_generation + 1;
            signal_params.params.fence.value = next_generation;
            if (cudaSignalExternalSemaphoresAsync(&slot.external_semaphore, &signal_params, 1, simulation.stream()) != cudaSuccess) throw std::runtime_error("cudaSignalExternalSemaphoresAsync failed");
            if (cudaStreamSynchronize(simulation.stream()) != cudaSuccess) throw std::runtime_error("cudaStreamSynchronize failed");
            slot.ready_generation = next_generation;
            slot.nx = static_cast<uint32_t>(grid.nx);
            slot.ny = static_cast<uint32_t>(grid.ny);
            slot.nz = static_cast<uint32_t>(grid.nz);
            slot.cell_size = grid.cell_size;
            slot.semantic = field.semantic;
            slot.label = field.label;
            viewer_runtime.snapshot_generation = next_generation;
            viewer_runtime.active_snapshot_slot = slot_index;
            viewer_runtime.steps_since_snapshot = 0;
            accumulate_snapshot(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
        };

        auto reset_backend = [&]() {
            nvtx3::scoped_range range{"smoke_app.reset_backend"};
            const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
            if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) throw std::runtime_error("smoke-visualizer requires Vulkan timeline semaphore support");
            int cuda_device_index = 0;
            if (cudaGetDevice(&cuda_device_index) != cudaSuccess) throw std::runtime_error("cudaGetDevice failed");
            int timeline_supported = 0;
            if (cudaDeviceGetAttribute(&timeline_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index) != cudaSuccess) throw std::runtime_error("cudaDeviceGetAttribute failed");
            if (timeline_supported == 0) throw std::runtime_error("CUDA timeline semaphore interop is required");

            simulation.rebuild();
            if (simulation.settings().emit_source) simulation.step(1);
            destroy_snapshot_slots();
            viewer_runtime.field_bytes = rgba_bytes(simulation.grid_desc());
            create_snapshot_slots();
            snapshot_current_field_to_slot(0, "smoke_app.reset_backend.initial_snapshot");
            apply_field_defaults(current_field());
            if (const auto field = active_snapshot()) renderer.frame_volume(*field);
        };

        bool force_snapshot_requested = false;
        reset_backend();

        while (!renderer.should_close()) {
            nvtx3::scoped_range frame_range{"smoke_app.frame"};
            renderer.begin_frame();

            bool reset_requested = false;
            bool field_changed = false;
            ImGui::Begin("Simulation");
            auto& settings = simulation.settings();
            auto& selected_field = settings.selected_field;
            selected_field = std::clamp(selected_field, 0, static_cast<int>(fields.size()) - 1);
            if (ImGui::BeginCombo("Field", fields[static_cast<size_t>(selected_field)].label.data())) {
                for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
                    const bool is_selected = selected_field == i;
                    if (ImGui::Selectable(fields[static_cast<size_t>(i)].label.data(), is_selected)) {
                        selected_field = i;
                        field_changed = true;
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("Pause Simulation", &playback.paused);
            if (ImGui::Button("Single Step")) playback.step_once = true;
            ImGui::SameLine();
            if (ImGui::Button("Reset Backend")) reset_requested = true;
            ImGui::SliderInt("Sim Steps / Frame", &playback.sim_steps_per_frame, 1, 8);
            ImGui::SliderInt("Snapshot Interval", &playback.snapshot_interval, 1, 8);
            ImGui::Separator();
            ImGui::TextUnformatted("Solver");
            ImGui::Text("Step calls: %llu", static_cast<unsigned long long>(simulation.stats().step_count));
            ImGui::Text("Last step call: %.3f ms", simulation.stats().last_step_call_ms);
            ImGui::Text("Avg step call: %.3f ms", simulation.stats().average_step_call_ms);
            ImGui::Text("Snapshot commits: %llu", static_cast<unsigned long long>(viewer_runtime.snapshot_count));
            ImGui::Text("Last snapshot: %.3f ms", viewer_runtime.last_snapshot_ms);
            ImGui::Text("Avg snapshot: %.3f ms", viewer_runtime.average_snapshot_ms);
            ImGui::Separator();

            ImGui::TextUnformatted("Grid / Time");
            if (ImGui::SliderInt("Grid X", &settings.config.nx, 16, 192)) reset_requested = true;
            if (ImGui::SliderInt("Grid Y", &settings.config.ny, 16, 192)) reset_requested = true;
            if (ImGui::SliderInt("Grid Z", &settings.config.nz, 16, 192)) reset_requested = true;
            if (ImGui::SliderFloat("Dt", &settings.config.dt, 1.0f / 240.0f, 1.0f / 24.0f, "%.5f")) reset_requested = true;
            if (ImGui::SliderFloat("Cell Size", &settings.config.cell_size, 0.25f, 2.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Viscosity", &settings.config.viscosity, 0.0f, 0.002f, "%.5f")) reset_requested = true;
            if (ImGui::SliderFloat("Density Diffusion", &settings.density_diffusion, 0.0f, 0.002f, "%.5f")) reset_requested = true;
            if (ImGui::SliderFloat("Dye Diffusion", &settings.dye_diffusion, 0.0f, 0.002f, "%.5f")) reset_requested = true;
            if (ImGui::SliderInt("Diffuse Iterations", &settings.config.diffuse_iterations, 1, 64)) reset_requested = true;
            if (ImGui::SliderInt("Pressure Iterations", &settings.config.pressure_iterations, 4, 192)) reset_requested = true;

            auto draw_boundary_combo = [&](const char* label, StableFluidsBoundaryFaceDesc& face) {
                int boundary = std::clamp(static_cast<int>(face.type), 0, static_cast<int>(boundary_labels.size()) - 1);
                if (ImGui::BeginCombo(label, boundary_labels[static_cast<size_t>(boundary)])) {
                    for (int i = 0; i < static_cast<int>(boundary_labels.size()); ++i) {
                        const bool is_selected = boundary == i;
                        if (ImGui::Selectable(boundary_labels[static_cast<size_t>(i)], is_selected)) {
                            boundary = i;
                            face.type = static_cast<uint32_t>(i);
                            reset_requested = true;
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            };

            ImGui::Separator();
            ImGui::TextUnformatted("Domain Boundary");
            draw_boundary_combo("Boundary X-", settings.config.domain_boundary.x_min);
            draw_boundary_combo("Boundary X+", settings.config.domain_boundary.x_max);
            draw_boundary_combo("Boundary Y-", settings.config.domain_boundary.y_min);
            draw_boundary_combo("Boundary Y+", settings.config.domain_boundary.y_max);
            draw_boundary_combo("Boundary Z-", settings.config.domain_boundary.z_min);
            draw_boundary_combo("Boundary Z+", settings.config.domain_boundary.z_max);
            if (ImGui::SliderFloat("Inflow Vel X-", &settings.config.domain_boundary.x_min.velocity, -4.0f, 4.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Inflow Vel X+", &settings.config.domain_boundary.x_max.velocity, -4.0f, 4.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Inflow Vel Y-", &settings.config.domain_boundary.y_min.velocity, -4.0f, 4.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Inflow Vel Y+", &settings.config.domain_boundary.y_max.velocity, -4.0f, 4.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Inflow Vel Z-", &settings.config.domain_boundary.z_min.velocity, -4.0f, 4.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Inflow Vel Z+", &settings.config.domain_boundary.z_max.velocity, -4.0f, 4.0f, "%.2f")) reset_requested = true;

            ImGui::Separator();
            ImGui::TextUnformatted("Forces");
            if (ImGui::SliderFloat("Density Buoyancy", &settings.density_buoyancy, 0.0f, 2.0f, "%.4f")) reset_requested = true;
            if (ImGui::SliderFloat("Uniform Force X", &settings.config.uniform_force_x, -2.0f, 2.0f, "%.3f")) reset_requested = true;
            if (ImGui::SliderFloat("Uniform Force Y", &settings.config.uniform_force_y, -2.0f, 2.0f, "%.3f")) reset_requested = true;
            if (ImGui::SliderFloat("Uniform Force Z", &settings.config.uniform_force_z, -2.0f, 2.0f, "%.3f")) reset_requested = true;

            ImGui::Separator();
            ImGui::TextUnformatted("Sources");
            if (ImGui::Checkbox("Emit Source", &settings.emit_source)) reset_requested = true;
            if (ImGui::SliderFloat("Source Radius", &settings.source_radius, 1.0f, 16.0f, "%.1f")) reset_requested = true;
            if (ImGui::SliderFloat("Density Amount", &settings.density_amount, 0.0f, 2.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Dye Amount", &settings.dye_amount, 0.0f, 2.0f, "%.2f")) reset_requested = true;
            if (ImGui::ColorEdit3("Source A Dye", &settings.source_a_r)) reset_requested = true;
            if (ImGui::ColorEdit3("Source B Dye", &settings.source_b_r)) reset_requested = true;
            if (ImGui::SliderFloat("Jet Speed", &settings.jet_speed, 0.2f, 200.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Upward Bias", &settings.upward_bias, -1.0f, 2.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Focus X", &settings.focus_x, 0.05f, 0.95f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Focus Y", &settings.focus_y, 0.05f, 0.95f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Focus Z", &settings.focus_z, 0.05f, 0.95f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Source A X", &settings.source_a_x, 0.03f, 0.97f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Source A Y", &settings.source_a_y, 0.03f, 0.97f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Source A Z", &settings.source_a_z, 0.03f, 0.97f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Source B X", &settings.source_b_x, 0.03f, 0.97f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Source B Y", &settings.source_b_y, 0.03f, 0.97f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Source B Z", &settings.source_b_z, 0.03f, 0.97f, "%.2f")) reset_requested = true;

            ImGui::Separator();
            ImGui::TextUnformatted("Collider");
            ImGui::Checkbox("Enable Collider", &settings.collider.enabled);
            int collider_type = std::clamp(settings.collider.type, 0, static_cast<int>(collider_type_labels.size()) - 1);
            if (ImGui::BeginCombo("Collider Type", collider_type_labels[static_cast<size_t>(collider_type)])) {
                for (int i = 0; i < static_cast<int>(collider_type_labels.size()); ++i) {
                    const bool is_selected = collider_type == i;
                    if (ImGui::Selectable(collider_type_labels[static_cast<size_t>(i)], is_selected)) {
                        settings.collider.type = i;
                        reset_requested = true;
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            int collider_boundary = std::clamp(static_cast<int>(settings.collider.boundary), 0, 1);
            if (ImGui::BeginCombo("Collider Boundary", boundary_labels[static_cast<size_t>(collider_boundary)])) {
                for (int i = 0; i < 2; ++i) {
                    const bool is_selected = collider_boundary == i;
                    if (ImGui::Selectable(boundary_labels[static_cast<size_t>(i)], is_selected)) {
                        settings.collider.boundary = static_cast<uint32_t>(i);
                        reset_requested = true;
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (ImGui::SliderFloat("Collider X", &settings.collider.center_x, 0.05f, 0.95f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Collider Y", &settings.collider.center_y, 0.05f, 0.95f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Collider Z", &settings.collider.center_z, 0.05f, 0.95f, "%.2f")) reset_requested = true;
            if (settings.collider.type == 0) {
                if (ImGui::SliderFloat("Collider Radius", &settings.collider.radius, 0.03f, 0.30f, "%.2f")) reset_requested = true;
            } else {
                if (ImGui::SliderFloat("Half Extent X", &settings.collider.half_extent_x, 0.03f, 0.30f, "%.2f")) reset_requested = true;
                if (ImGui::SliderFloat("Half Extent Y", &settings.collider.half_extent_y, 0.03f, 0.30f, "%.2f")) reset_requested = true;
                if (ImGui::SliderFloat("Half Extent Z", &settings.collider.half_extent_z, 0.03f, 0.30f, "%.2f")) reset_requested = true;
            }
            if (ImGui::SliderFloat("Collider Vel X", &settings.collider.velocity_x, -3.0f, 3.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Collider Vel Y", &settings.collider.velocity_y, -3.0f, 3.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Collider Vel Z", &settings.collider.velocity_z, -3.0f, 3.0f, "%.2f")) reset_requested = true;
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
                    simulation.step(playback.sim_steps_per_frame);
                    if (viewer_runtime.steps_since_snapshot < static_cast<uint32_t>(playback.snapshot_interval)) ++viewer_runtime.steps_since_snapshot;
                    if (viewer_runtime.steps_since_snapshot >= static_cast<uint32_t>(playback.snapshot_interval)) {
                        const int slot_index = find_available_snapshot_slot();
                        if (slot_index >= 0) {
                            snapshot_current_field_to_slot(slot_index, "smoke_app.simulation.snapshot");
                            force_snapshot_requested = false;
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
            const bool submitted = renderer.render_frame(field);
            if (submitted) {
                const uint64_t next_submit_serial = viewer_runtime.submit_serial + 1;
                if (viewer_runtime.active_snapshot_slot >= 0) snapshot_slots[static_cast<size_t>(viewer_runtime.active_snapshot_slot)].last_used_submit_serial = next_submit_serial;
                viewer_runtime.submit_serial = next_submit_serial;
            }
        }

        renderer.vk_context().device.waitIdle();
        destroy_snapshot_slots();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
