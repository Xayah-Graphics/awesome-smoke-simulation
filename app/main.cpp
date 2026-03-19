#include <cuda_runtime.h>
#include <imgui.h>
#include <nvtx3/nvtx3.hpp>
#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include <vulkan/vulkan_raii.hpp>

#include "visual-simulation-of-smoke.h"

import app;
import std;
import vk.memory;

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    struct PlaybackState {
        bool paused = false;
        bool step_once = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval = 2;
    };

    struct VisualSmokeSettings {
        VisualSimulationOfSmokeContextDesc desc = visual_simulation_of_smoke_context_desc_default();
        bool emit_source = true;
        float source_u = 0.5f;
        float source_v = 0.18f;
        float source_w = 0.5f;
        float source_radius = 5.0f;
        float density_amount = 0.85f;
        float temperature_amount = 1.35f;
        float velocity_x = 0.0f;
        float velocity_y = 1.2f;
        float velocity_z = 0.0f;

        VisualSmokeSettings() {
            desc.nx = 64;
            desc.ny = 96;
            desc.nz = 64;
            desc.dt = 1.0f / 90.0f;
            desc.cell_size = 1.0f;
            desc.ambient_temperature = 0.0f;
            desc.density_buoyancy = 0.045f;
            desc.temperature_buoyancy = 0.12f;
            desc.vorticity_epsilon = 2.0f;
            desc.pressure_iterations = 80;
            desc.use_monotonic_cubic = 1u;
        }
    };

    struct VisualSmokeRuntime {
        VisualSimulationOfSmokeContext* context = nullptr;
        cudaStream_t sim_stream = nullptr;
        uint64_t field_bytes = 0;
        uint64_t snapshot_generation = 0;
        uint64_t submit_serial = 0;
        uint32_t steps_since_snapshot = 0;
        int active_snapshot_slot = -1;
        uint64_t active_snapshot_generation = 0;
    };

    struct SnapshotSlot {
        vk::raii::Buffer buffer{nullptr};
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Semaphore timeline_semaphore{nullptr};
        vk::raii::DescriptorSet descriptor_set{nullptr};
        VisualSimulationOfSmokeBufferView buffer_view{};
        cudaExternalMemory_t external_memory = nullptr;
        cudaExternalSemaphore_t external_semaphore = nullptr;
        void* cuda_ptr = nullptr;
        uint64_t ready_generation = 0;
        uint64_t last_used_submit_serial = 0;
    };

    auto make_device_buffer_view(void* data, const uint64_t size_bytes) {
        return VisualSimulationOfSmokeBufferView{
            .data = data,
            .size_bytes = size_bytes,
            .format = VISUAL_SIMULATION_OF_SMOKE_BUFFER_FORMAT_F32,
            .memory_type = VISUAL_SIMULATION_OF_SMOKE_MEMORY_TYPE_CUDA_DEVICE,
        };
    }

    auto cuda_ok(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) {
            return;
        }
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }

    auto smoke_ok(const int32_t code, const char* what, const VisualSimulationOfSmokeContext* context = nullptr) {
        if (code == VISUAL_SIMULATION_OF_SMOKE_SUCCESS) {
            return;
        }

        const uint64_t message_length = context != nullptr ? visual_simulation_of_smoke_context_last_error_length(context) : visual_simulation_of_smoke_last_error_length();
        std::vector<char> message(static_cast<size_t>(message_length + 1), '\0');
        const int32_t copy_code = context != nullptr
            ? visual_simulation_of_smoke_copy_context_last_error(context, message.data(), static_cast<uint64_t>(message.size()))
            : visual_simulation_of_smoke_copy_last_error(message.data(), static_cast<uint64_t>(message.size()));

        std::string full = what;
        full += " failed";
        if (copy_code == VISUAL_SIMULATION_OF_SMOKE_SUCCESS && !message.empty() && message[0] != '\0') {
            full += ": ";
            full += message.data();
        }
        throw std::runtime_error(full);
    }

} // namespace

int main() {
    try {
        app::FieldRendererApp renderer;
        PlaybackState playback{};
        VisualSmokeSettings settings{};
        VisualSmokeRuntime runtime{};
        std::vector<vk::raii::DescriptorSet> density_descriptor_sets{};
        std::vector<SnapshotSlot> snapshot_slots{};

        auto make_source = [&]() {
            return VisualSimulationOfSmokeSourceDesc{
                .center_x = settings.source_u * static_cast<float>(settings.desc.nx),
                .center_y = settings.source_v * static_cast<float>(settings.desc.ny),
                .center_z = settings.source_w * static_cast<float>(settings.desc.nz),
                .radius = settings.source_radius,
                .density_amount = settings.density_amount,
                .temperature_amount = settings.temperature_amount,
                .velocity_x = settings.velocity_x,
                .velocity_y = settings.velocity_y,
                .velocity_z = settings.velocity_z,
            };
        };

        auto active_snapshot = [&]() -> std::optional<app::VolumeFieldView> {
            if (runtime.active_snapshot_slot < 0) {
                return std::nullopt;
            }

            const auto& slot = snapshot_slots.at(static_cast<size_t>(runtime.active_snapshot_slot));
            return app::VolumeFieldView{
                .descriptor_set = *slot.descriptor_set,
                .timeline_semaphore = *slot.timeline_semaphore,
                .ready_generation = slot.ready_generation,
                .nx = static_cast<uint32_t>(settings.desc.nx),
                .ny = static_cast<uint32_t>(settings.desc.ny),
                .nz = static_cast<uint32_t>(settings.desc.nz),
                .cell_size = settings.desc.cell_size,
            };
        };

        auto destroy_backend = [&]() {
            nvtx3::scoped_range range{"smoke_app.destroy_backend"};
            renderer.vk_context().device.waitIdle();

            if (runtime.sim_stream != nullptr) {
                cudaStreamSynchronize(runtime.sim_stream);
            }

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
                slot.buffer_view = {};
                slot.ready_generation = 0;
                slot.last_used_submit_serial = 0;
            }
            snapshot_slots.clear();

            if (runtime.sim_stream != nullptr) {
                cudaStreamDestroy(runtime.sim_stream);
                runtime.sim_stream = nullptr;
            }
            if (runtime.context != nullptr) {
                visual_simulation_of_smoke_context_destroy(runtime.context);
                runtime.context = nullptr;
            }

            runtime.field_bytes = 0;
            runtime.snapshot_generation = 0;
            runtime.submit_serial = 0;
            runtime.steps_since_snapshot = 0;
            runtime.active_snapshot_slot = -1;
            runtime.active_snapshot_generation = 0;
        };

        auto reset_backend = [&]() {
            nvtx3::scoped_range range{"smoke_app.reset_backend"};
            const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
            if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) {
                throw std::runtime_error("002-visual-simulation-of-smoke backend requires Vulkan timeline semaphore support");
            }

            int cuda_device_index = 0;
            cuda_ok(cudaGetDevice(&cuda_device_index), "cudaGetDevice");
            int cuda_timeline_semaphore_interop_supported = 0;
            cuda_ok(cudaDeviceGetAttribute(&cuda_timeline_semaphore_interop_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index), "cudaDeviceGetAttribute cudaDevAttrTimelineSemaphoreInteropSupported");
            if (cuda_timeline_semaphore_interop_supported == 0) {
                throw std::runtime_error("002-visual-simulation-of-smoke backend requires CUDA timeline semaphore interop support");
            }

            destroy_backend();
            density_descriptor_sets = renderer.allocate_density_descriptor_sets(snapshot_slot_count);

            runtime.context = visual_simulation_of_smoke_context_create(&settings.desc);
            if (runtime.context == nullptr) {
                smoke_ok(VISUAL_SIMULATION_OF_SMOKE_ERROR_RUNTIME, "visual_simulation_of_smoke_context_create");
            }

            cuda_ok(cudaStreamCreateWithFlags(&runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags sim_stream");
            runtime.field_bytes = visual_simulation_of_smoke_context_required_density_bytes(runtime.context);

            snapshot_slots.clear();
            snapshot_slots.reserve(snapshot_slot_count);

            for (uint32_t slot_index = 0; slot_index < snapshot_slot_count; ++slot_index) {
                SnapshotSlot slot{};
                slot.descriptor_set = std::move(density_descriptor_sets.at(slot_index));

                vk::SemaphoreTypeCreateInfo timeline_semaphore_ci{
                    .semaphoreType = vk::SemaphoreType::eTimeline,
                    .initialValue = 0,
                };
                vk::ExportSemaphoreCreateInfo export_semaphore_ci{
                    .pNext = &timeline_semaphore_ci,
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
                    .pNext = &external_buffer_ci,
                    .size = runtime.field_bytes,
                    .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                    .sharingMode = vk::SharingMode::eExclusive,
                };
                slot.buffer = vk::raii::Buffer{renderer.vk_context().device, buffer_ci};

                const vk::MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
                vk::ExportMemoryAllocateInfo export_memory_ci{
                    .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
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
                    .handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
                };
                HANDLE memory_handle = renderer.vk_context().device.getMemoryWin32HandleKHR(handle_info);
                if (memory_handle == nullptr) {
                    throw std::runtime_error("getMemoryWin32HandleKHR returned a null handle");
                }

                cudaExternalMemoryHandleDesc external_desc{};
                external_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                external_desc.handle.win32.handle = memory_handle;
                external_desc.size = requirements.size;
                cuda_ok(cudaImportExternalMemory(&slot.external_memory, &external_desc), "cudaImportExternalMemory");
                CloseHandle(memory_handle);

                cudaExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.offset = 0;
                buffer_desc.size = runtime.field_bytes;
                cuda_ok(cudaExternalMemoryGetMappedBuffer(&slot.cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");

                vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                    .semaphore = *slot.timeline_semaphore,
                    .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32,
                };
                HANDLE semaphore_handle = renderer.vk_context().device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
                if (semaphore_handle == nullptr) {
                    throw std::runtime_error("getSemaphoreWin32HandleKHR returned a null handle");
                }

                cudaExternalSemaphoreHandleDesc external_semaphore_desc{};
                external_semaphore_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
                external_semaphore_desc.handle.win32.handle = semaphore_handle;
                cuda_ok(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
                CloseHandle(semaphore_handle);
#else
                throw std::runtime_error("smoke-visualizer currently requires Windows external memory interop");
#endif

                slot.buffer_view = make_device_buffer_view(slot.cuda_ptr, runtime.field_bytes);

                vk::DescriptorBufferInfo density_info{
                    .buffer = *slot.buffer,
                    .offset = 0,
                    .range = runtime.field_bytes,
                };
                vk::WriteDescriptorSet density_write{
                    .dstSet = *slot.descriptor_set,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &density_info,
                };
                renderer.vk_context().device.updateDescriptorSets(density_write, {});
                snapshot_slots.push_back(std::move(slot));
            }

            {
                nvtx3::scoped_range init_range{"smoke_app.reset_backend.initialize"};
                smoke_ok(visual_simulation_of_smoke_clear_async(runtime.context, runtime.sim_stream), "visual_simulation_of_smoke_clear_async", runtime.context);
                if (settings.emit_source) {
                    const auto source = make_source();
                    smoke_ok(visual_simulation_of_smoke_add_source_async(runtime.context, &source, runtime.sim_stream), "visual_simulation_of_smoke_add_source_async", runtime.context);
                    smoke_ok(visual_simulation_of_smoke_step_async(runtime.context, runtime.sim_stream), "visual_simulation_of_smoke_step_async", runtime.context);
                }

                smoke_ok(visual_simulation_of_smoke_snapshot_density_async(runtime.context, snapshot_slots[0].buffer_view, runtime.sim_stream), "visual_simulation_of_smoke_snapshot_density_async", runtime.context);
                const uint64_t initial_generation = 1;
                cudaExternalSemaphoreSignalParams signal_params{};
                signal_params.params.fence.value = initial_generation;
                cuda_ok(cudaSignalExternalSemaphoresAsync(&snapshot_slots[0].external_semaphore, &signal_params, 1, runtime.sim_stream), "cudaSignalExternalSemaphoresAsync initial_snapshot");
                cuda_ok(cudaStreamSynchronize(runtime.sim_stream), "cudaStreamSynchronize initial_snapshot");
                runtime.snapshot_generation = initial_generation;
                snapshot_slots[0].ready_generation = initial_generation;
                runtime.active_snapshot_slot = 0;
                runtime.active_snapshot_generation = initial_generation;
                runtime.steps_since_snapshot = 0;
            }

            if (const auto field = active_snapshot()) {
                renderer.frame_volume(*field);
            }
        };

        reset_backend();

        while (!renderer.should_close()) {
            nvtx3::scoped_range frame_range{"smoke_app.frame"};
            renderer.begin_frame();

            bool reset_requested = false;
            ImGui::Begin("Simulation");
            ImGui::TextUnformatted("Backend: 002-visual-simulation-of-smoke");
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
            if (ImGui::SliderInt("Grid X", &settings.desc.nx, 16, 192)) reset_requested = true;
            if (ImGui::SliderInt("Grid Y", &settings.desc.ny, 16, 192)) reset_requested = true;
            if (ImGui::SliderInt("Grid Z", &settings.desc.nz, 16, 192)) reset_requested = true;
            if (ImGui::SliderFloat("Dt", &settings.desc.dt, 1.0f / 240.0f, 1.0f / 24.0f, "%.5f")) reset_requested = true;
            if (ImGui::SliderFloat("Cell Size", &settings.desc.cell_size, 0.25f, 2.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Ambient Temp", &settings.desc.ambient_temperature, -2.0f, 2.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Density Buoyancy", &settings.desc.density_buoyancy, 0.0f, 0.2f, "%.4f")) reset_requested = true;
            if (ImGui::SliderFloat("Temperature Buoyancy", &settings.desc.temperature_buoyancy, 0.0f, 0.4f, "%.4f")) reset_requested = true;
            if (ImGui::SliderFloat("Vorticity Epsilon", &settings.desc.vorticity_epsilon, 0.0f, 6.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderInt("Pressure Iterations", &settings.desc.pressure_iterations, 8, 160)) reset_requested = true;
            bool use_monotonic_cubic = settings.desc.use_monotonic_cubic != 0u;
            if (ImGui::Checkbox("Monotonic Cubic", &use_monotonic_cubic)) {
                settings.desc.use_monotonic_cubic = use_monotonic_cubic ? 1u : 0u;
                reset_requested = true;
            }
            ImGui::Separator();
            ImGui::Checkbox("Emit Source", &settings.emit_source);
            ImGui::SliderFloat("Source U", &settings.source_u, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Source V", &settings.source_v, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Source W", &settings.source_w, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Source Radius", &settings.source_radius, 1.0f, 16.0f, "%.1f");
            ImGui::SliderFloat("Density Amount", &settings.density_amount, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Temperature Amount", &settings.temperature_amount, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Velocity X", &settings.velocity_x, -3.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Velocity Y", &settings.velocity_y, -3.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Velocity Z", &settings.velocity_z, -3.0f, 3.0f, "%.2f");
            ImGui::End();

            if (reset_requested) {
                reset_backend();
            } else {
                const bool run_simulation = !playback.paused || playback.step_once;
                if (run_simulation) {
                    nvtx3::scoped_range sim_range{"smoke_app.simulation"};
                    for (int sim_step = 0; sim_step < playback.sim_steps_per_frame; ++sim_step) {
                        if (settings.emit_source) {
                            nvtx3::scoped_range source_range{"smoke_app.simulation.add_source"};
                            const auto source = make_source();
                            smoke_ok(visual_simulation_of_smoke_add_source_async(runtime.context, &source, runtime.sim_stream), "visual_simulation_of_smoke_add_source_async", runtime.context);
                        }

                        {
                            nvtx3::scoped_range step_range{"smoke_app.simulation.step"};
                            smoke_ok(visual_simulation_of_smoke_step_async(runtime.context, runtime.sim_stream), "visual_simulation_of_smoke_step_async", runtime.context);
                        }

                        if (runtime.steps_since_snapshot < static_cast<uint32_t>(playback.snapshot_interval)) {
                            ++runtime.steps_since_snapshot;
                        }
                        if (runtime.steps_since_snapshot >= static_cast<uint32_t>(playback.snapshot_interval)) {
                            int snapshot_slot_index = -1;
                            for (uint32_t slot_index = 0; slot_index < snapshot_slots.size(); ++slot_index) {
                                auto& slot = snapshot_slots[slot_index];
                                if (static_cast<int>(slot_index) == runtime.active_snapshot_slot) {
                                    continue;
                                }
                                if (slot.ready_generation != 0 && runtime.submit_serial < slot.last_used_submit_serial + renderer.frames_in_flight() + 1) {
                                    continue;
                                }
                                snapshot_slot_index = static_cast<int>(slot_index);
                                break;
                            }

                            if (snapshot_slot_index >= 0) {
                                nvtx3::scoped_range snapshot_range{"smoke_app.simulation.snapshot"};
                                auto& slot = snapshot_slots[static_cast<size_t>(snapshot_slot_index)];
                                smoke_ok(visual_simulation_of_smoke_snapshot_density_async(runtime.context, slot.buffer_view, runtime.sim_stream), "visual_simulation_of_smoke_snapshot_density_async", runtime.context);
                                const uint64_t next_generation = runtime.snapshot_generation + 1;
                                cudaExternalSemaphoreSignalParams signal_params{};
                                signal_params.params.fence.value = next_generation;
                                cuda_ok(cudaSignalExternalSemaphoresAsync(&slot.external_semaphore, &signal_params, 1, runtime.sim_stream), "cudaSignalExternalSemaphoresAsync snapshot");
                                cuda_ok(cudaStreamSynchronize(runtime.sim_stream), "cudaStreamSynchronize snapshot");
                                slot.ready_generation = next_generation;
                                runtime.snapshot_generation = next_generation;
                                runtime.active_snapshot_generation = next_generation;
                                runtime.active_snapshot_slot = snapshot_slot_index;
                                runtime.steps_since_snapshot = 0;
                            }
                        }
                    }
                }
            }

            playback.step_once = false;

            const auto field = active_snapshot();
            renderer.draw_renderer_ui(field);

            bool submitted = false;
            {
                nvtx3::scoped_range render_range{"smoke_app.render"};
                submitted = renderer.render_frame(field);
            }
            if (submitted) {
                const uint64_t next_submit_serial = runtime.submit_serial + 1;
                if (runtime.active_snapshot_slot >= 0) {
                    snapshot_slots[static_cast<size_t>(runtime.active_snapshot_slot)].last_used_submit_serial = next_submit_serial;
                }
                runtime.submit_serial = next_submit_serial;
            }
        }

        destroy_backend();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
