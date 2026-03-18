#include <cuda_runtime.h>
#include <imgui.h>
#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include <vulkan/vulkan_raii.hpp>

#include "stable_fluids.h"

import app;
import std;
import vk.memory;

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    struct PlaybackState {
        bool paused = false;
        bool step_once = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval = 3;
    };

    struct StableFluidsSettings {
        StableFluidsContextDesc desc = stable_fluids_context_desc_default();
        bool emit_density = true;
        bool emit_force = true;
        float source_u = 0.5f;
        float source_v = 0.22f;
        float source_w = 0.5f;
        float density_radius = 5.0f;
        float density_amount = 1.2f;
        float force_radius = 6.0f;
        float force_x = 0.0f;
        float force_y = 1.8f;
        float force_z = 0.0f;

        StableFluidsSettings() {
            desc.nx = 96;
            desc.ny = 96;
            desc.nz = 96;
            desc.dt = 1.0f / 90.0f;
            desc.viscosity = 0.00012f;
            desc.diffusion = 0.00003f;
            desc.diffuse_iterations = 8;
            desc.pressure_iterations = 24;
            desc.pressure_tolerance = 1.0e-5f;
        }
    };

    struct StableFluidsRuntime {
        StableFluidsContext* context = nullptr;
        StableFluidsFieldSetDesc fields{};
        float* density = nullptr;
        float* velocity_x = nullptr;
        float* velocity_y = nullptr;
        float* velocity_z = nullptr;
        cudaStream_t sim_stream = nullptr;
        cudaStream_t snapshot_stream = nullptr;
        cudaEvent_t step_complete_event = nullptr;
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
        StableFluidsBufferView buffer_view{};
        cudaExternalMemory_t external_memory = nullptr;
        cudaExternalSemaphore_t external_semaphore = nullptr;
        void* cuda_ptr = nullptr;
        bool pending = false;
        uint64_t pending_generation = 0;
        uint64_t ready_generation = 0;
        uint64_t last_used_submit_serial = 0;
    };

    auto make_device_buffer_view(void* data, const uint64_t size_bytes) {
        StableFluidsBufferView view{};
        view.data = data;
        view.size_bytes = size_bytes;
        view.format = STABLE_FLUIDS_BUFFER_FORMAT_F32;
        view.memory_type = STABLE_FLUIDS_MEMORY_TYPE_CUDA_DEVICE;
        return view;
    }

    auto cuda_ok(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) {
            return;
        }
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }

    auto stable_ok(const int32_t code, const char* what, const StableFluidsContext* context = nullptr) {
        if (code == STABLE_FLUIDS_SUCCESS) {
            return;
        }

        const uint64_t message_length =
            context != nullptr ? stable_fluids_context_last_error_length(context) : stable_fluids_last_error_length();
        std::vector<char> message(static_cast<size_t>(message_length + 1), '\0');
        const int32_t copy_code = context != nullptr
            ? stable_fluids_copy_context_last_error(context, message.data(), static_cast<uint64_t>(message.size()))
            : stable_fluids_copy_last_error(message.data(), static_cast<uint64_t>(message.size()));

        std::string full = what;
        full += " failed";
        if (copy_code == STABLE_FLUIDS_SUCCESS && !message.empty() && message[0] != '\0') {
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
        StableFluidsSettings settings{};
        StableFluidsRuntime runtime{};
        std::vector<vk::raii::DescriptorSet> density_descriptor_sets{};
        std::vector<SnapshotSlot> snapshot_slots{};

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
            renderer.vk_context().device.waitIdle();

            if (runtime.snapshot_stream != nullptr) {
                cudaStreamSynchronize(runtime.snapshot_stream);
            }
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
                slot.pending = false;
                slot.pending_generation = 0;
                slot.ready_generation = 0;
                slot.last_used_submit_serial = 0;
            }
            snapshot_slots.clear();

            if (runtime.step_complete_event != nullptr) {
                cudaEventDestroy(runtime.step_complete_event);
                runtime.step_complete_event = nullptr;
            }
            if (runtime.snapshot_stream != nullptr) {
                cudaStreamDestroy(runtime.snapshot_stream);
                runtime.snapshot_stream = nullptr;
            }
            if (runtime.sim_stream != nullptr) {
                cudaStreamDestroy(runtime.sim_stream);
                runtime.sim_stream = nullptr;
            }
            if (runtime.density != nullptr) {
                cudaFree(runtime.density);
                runtime.density = nullptr;
            }
            if (runtime.velocity_x != nullptr) {
                cudaFree(runtime.velocity_x);
                runtime.velocity_x = nullptr;
            }
            if (runtime.velocity_y != nullptr) {
                cudaFree(runtime.velocity_y);
                runtime.velocity_y = nullptr;
            }
            if (runtime.velocity_z != nullptr) {
                cudaFree(runtime.velocity_z);
                runtime.velocity_z = nullptr;
            }
            if (runtime.context != nullptr) {
                stable_fluids_context_destroy(runtime.context);
                runtime.context = nullptr;
            }

            runtime.fields = {};
            runtime.field_bytes = 0;
            runtime.snapshot_generation = 0;
            runtime.submit_serial = 0;
            runtime.steps_since_snapshot = 0;
            runtime.active_snapshot_slot = -1;
            runtime.active_snapshot_generation = 0;
        };

        auto reset_backend = [&]() {
            const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
            if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) {
                throw std::runtime_error("001-stable-fluids backend requires Vulkan timeline semaphore support");
            }

            int cuda_device_index = 0;
            cuda_ok(cudaGetDevice(&cuda_device_index), "cudaGetDevice");
            int cuda_timeline_semaphore_interop_supported = 0;
            cuda_ok(
                cudaDeviceGetAttribute(
                    &cuda_timeline_semaphore_interop_supported,
                    cudaDevAttrTimelineSemaphoreInteropSupported,
                    cuda_device_index),
                "cudaDeviceGetAttribute cudaDevAttrTimelineSemaphoreInteropSupported");
            if (cuda_timeline_semaphore_interop_supported == 0) {
                throw std::runtime_error("001-stable-fluids backend requires CUDA timeline semaphore interop support");
            }

            destroy_backend();
            density_descriptor_sets = renderer.allocate_density_descriptor_sets(snapshot_slot_count);

            runtime.context = stable_fluids_context_create(&settings.desc);
            if (runtime.context == nullptr) {
                stable_ok(STABLE_FLUIDS_ERROR_RUNTIME, "stable_fluids_context_create", nullptr);
            }

            runtime.field_bytes = stable_fluids_context_required_scalar_field_bytes(runtime.context);

            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&runtime.density), runtime.field_bytes), "cudaMalloc density");
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&runtime.velocity_x), runtime.field_bytes), "cudaMalloc velocity_x");
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&runtime.velocity_y), runtime.field_bytes), "cudaMalloc velocity_y");
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&runtime.velocity_z), runtime.field_bytes), "cudaMalloc velocity_z");

            runtime.fields.density = make_device_buffer_view(runtime.density, runtime.field_bytes);
            runtime.fields.velocity_x = make_device_buffer_view(runtime.velocity_x, runtime.field_bytes);
            runtime.fields.velocity_y = make_device_buffer_view(runtime.velocity_y, runtime.field_bytes);
            runtime.fields.velocity_z = make_device_buffer_view(runtime.velocity_z, runtime.field_bytes);

            cuda_ok(cudaStreamCreateWithFlags(&runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags sim_stream");
            cuda_ok(cudaStreamCreateWithFlags(&runtime.snapshot_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags snapshot_stream");
            cuda_ok(cudaEventCreateWithFlags(&runtime.step_complete_event, cudaEventDisableTiming), "cudaEventCreateWithFlags step_complete_event");

            stable_ok(stable_fluids_fields_clear_async(runtime.context, &runtime.fields, runtime.sim_stream), "stable_fluids_fields_clear_async", runtime.context);
            cuda_ok(cudaStreamSynchronize(runtime.sim_stream), "cudaStreamSynchronize clear");

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

            const float source_x = settings.source_u * static_cast<float>(std::max(settings.desc.nx - 1, 0));
            const float source_y = settings.source_v * static_cast<float>(std::max(settings.desc.ny - 1, 0));
            const float source_z = settings.source_w * static_cast<float>(std::max(settings.desc.nz - 1, 0));

            StableFluidsDensitySplatDesc density_splat{};
            density_splat.center_x = source_x;
            density_splat.center_y = source_y;
            density_splat.center_z = source_z;
            density_splat.radius = settings.density_radius;
            density_splat.amount = settings.density_amount;
            stable_ok(stable_fluids_fields_add_density_splat_async(runtime.context, &runtime.fields, &density_splat, runtime.sim_stream), "stable_fluids_fields_add_density_splat_async", runtime.context);

            StableFluidsForceSplatDesc force_splat{};
            force_splat.center_x = source_x;
            force_splat.center_y = source_y;
            force_splat.center_z = source_z;
            force_splat.radius = settings.force_radius;
            force_splat.force_x = settings.force_x;
            force_splat.force_y = settings.force_y;
            force_splat.force_z = settings.force_z;
            stable_ok(stable_fluids_fields_add_force_splat_async(runtime.context, &runtime.fields, &force_splat, runtime.sim_stream), "stable_fluids_fields_add_force_splat_async", runtime.context);
            stable_ok(stable_fluids_fields_step_async(runtime.context, &runtime.fields, runtime.sim_stream), "stable_fluids_fields_step_async", runtime.context);
            cuda_ok(cudaEventRecord(runtime.step_complete_event, runtime.sim_stream), "cudaEventRecord initial_step_complete_event");
            cuda_ok(cudaStreamWaitEvent(runtime.snapshot_stream, runtime.step_complete_event, 0), "cudaStreamWaitEvent initial_snapshot");
            stable_ok(stable_fluids_fields_snapshot_density_async(runtime.context, &runtime.fields, snapshot_slots[0].buffer_view, runtime.snapshot_stream), "stable_fluids_fields_snapshot_density_async", runtime.context);

            const uint64_t initial_generation = runtime.snapshot_generation + 1;
            cudaExternalSemaphoreSignalParams initial_signal_params{};
            initial_signal_params.params.fence.value = initial_generation;
            cuda_ok(
                cudaSignalExternalSemaphoresAsync(
                    &snapshot_slots[0].external_semaphore,
                    &initial_signal_params,
                    1,
                    runtime.snapshot_stream),
                "cudaSignalExternalSemaphoresAsync initial_snapshot");
            cuda_ok(cudaStreamSynchronize(runtime.snapshot_stream), "cudaStreamSynchronize initial_snapshot");
            runtime.snapshot_generation = initial_generation;
            snapshot_slots[0].ready_generation = initial_generation;
            runtime.active_snapshot_slot = 0;
            runtime.active_snapshot_generation = snapshot_slots[0].ready_generation;
            runtime.steps_since_snapshot = 0;

            if (const auto field = active_snapshot()) {
                renderer.frame_volume(*field);
            }
        };

        reset_backend();

        while (!renderer.should_close()) {
            renderer.begin_frame();

            bool reset_requested = false;
            ImGui::Begin("Simulation");
            ImGui::TextUnformatted("Backend: 001-stable-fluids");
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
            if (ImGui::SliderInt("Grid X", &settings.desc.nx, 32, 256)) reset_requested = true;
            if (ImGui::SliderInt("Grid Y", &settings.desc.ny, 32, 256)) reset_requested = true;
            if (ImGui::SliderInt("Grid Z", &settings.desc.nz, 32, 256)) reset_requested = true;
            if (ImGui::SliderFloat("Dt", &settings.desc.dt, 1.0f / 240.0f, 1.0f / 24.0f, "%.5f")) reset_requested = true;
            if (ImGui::SliderFloat("Cell Size", &settings.desc.cell_size, 0.25f, 2.0f, "%.2f")) reset_requested = true;
            if (ImGui::SliderFloat("Viscosity", &settings.desc.viscosity, 0.0f, 0.002f, "%.5f")) reset_requested = true;
            if (ImGui::SliderFloat("Diffusion", &settings.desc.diffusion, 0.0f, 0.002f, "%.5f")) reset_requested = true;
            if (ImGui::SliderInt("Diffuse Iterations", &settings.desc.diffuse_iterations, 1, 32)) reset_requested = true;
            if (ImGui::SliderInt("Pressure Iterations", &settings.desc.pressure_iterations, 1, 64)) reset_requested = true;
            if (ImGui::SliderFloat("Pressure Tolerance", &settings.desc.pressure_tolerance, 1.0e-7f, 1.0e-3f, "%.6f", ImGuiSliderFlags_Logarithmic)) reset_requested = true;
            ImGui::Separator();
            ImGui::Checkbox("Emit Density", &settings.emit_density);
            ImGui::Checkbox("Emit Force", &settings.emit_force);
            ImGui::SliderFloat("Source U", &settings.source_u, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Source V", &settings.source_v, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Source W", &settings.source_w, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Density Radius", &settings.density_radius, 1.0f, 16.0f, "%.1f");
            ImGui::SliderFloat("Density Amount", &settings.density_amount, 0.1f, 8.0f, "%.2f");
            ImGui::SliderFloat("Force Radius", &settings.force_radius, 1.0f, 20.0f, "%.1f");
            ImGui::SliderFloat("Force X", &settings.force_x, -4.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Force Y", &settings.force_y, -4.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Force Z", &settings.force_z, -4.0f, 4.0f, "%.2f");
            ImGui::End();

            if (reset_requested) {
                reset_backend();
            } else {
                const bool run_simulation = !playback.paused || playback.step_once;
                if (run_simulation) {
                    const float source_x = settings.source_u * static_cast<float>(std::max(settings.desc.nx - 1, 0));
                    const float source_y = settings.source_v * static_cast<float>(std::max(settings.desc.ny - 1, 0));
                    const float source_z = settings.source_w * static_cast<float>(std::max(settings.desc.nz - 1, 0));

                    for (int sim_step = 0; sim_step < playback.sim_steps_per_frame; ++sim_step) {
                        if (settings.emit_density) {
                            StableFluidsDensitySplatDesc density_splat{};
                            density_splat.center_x = source_x;
                            density_splat.center_y = source_y;
                            density_splat.center_z = source_z;
                            density_splat.radius = settings.density_radius;
                            density_splat.amount = settings.density_amount;
                            stable_ok(stable_fluids_fields_add_density_splat_async(runtime.context, &runtime.fields, &density_splat, runtime.sim_stream), "stable_fluids_fields_add_density_splat_async", runtime.context);
                        }

                        if (settings.emit_force) {
                            StableFluidsForceSplatDesc force_splat{};
                            force_splat.center_x = source_x;
                            force_splat.center_y = source_y;
                            force_splat.center_z = source_z;
                            force_splat.radius = settings.force_radius;
                            force_splat.force_x = settings.force_x;
                            force_splat.force_y = settings.force_y;
                            force_splat.force_z = settings.force_z;
                            stable_ok(stable_fluids_fields_add_force_splat_async(runtime.context, &runtime.fields, &force_splat, runtime.sim_stream), "stable_fluids_fields_add_force_splat_async", runtime.context);
                        }

                        stable_ok(stable_fluids_fields_step_async(runtime.context, &runtime.fields, runtime.sim_stream), "stable_fluids_fields_step_async", runtime.context);
                        cuda_ok(cudaEventRecord(runtime.step_complete_event, runtime.sim_stream), "cudaEventRecord step_complete_event");

                        if (runtime.steps_since_snapshot < static_cast<uint32_t>(playback.snapshot_interval)) {
                            ++runtime.steps_since_snapshot;
                        }
                        if (runtime.steps_since_snapshot >= static_cast<uint32_t>(playback.snapshot_interval)) {
                            int snapshot_slot_index = -1;
                            for (uint32_t slot_index = 0; slot_index < snapshot_slots.size(); ++slot_index) {
                                auto& slot = snapshot_slots[slot_index];
                                if (slot.pending) {
                                    continue;
                                }
                                if (static_cast<int>(slot_index) == runtime.active_snapshot_slot) {
                                    continue;
                                }
                                if (slot.ready_generation != 0 &&
                                    runtime.submit_serial < slot.last_used_submit_serial + renderer.frames_in_flight() + 1) {
                                    continue;
                                }
                                snapshot_slot_index = static_cast<int>(slot_index);
                                break;
                            }

                            if (snapshot_slot_index >= 0) {
                                auto& slot = snapshot_slots[static_cast<size_t>(snapshot_slot_index)];
                                const uint64_t next_generation = runtime.snapshot_generation + 1;
                                cuda_ok(cudaStreamWaitEvent(runtime.snapshot_stream, runtime.step_complete_event, 0), "cudaStreamWaitEvent snapshot_stream");
                                stable_ok(stable_fluids_fields_snapshot_density_async(runtime.context, &runtime.fields, slot.buffer_view, runtime.snapshot_stream), "stable_fluids_fields_snapshot_density_async", runtime.context);
                                slot.pending = true;
                                slot.pending_generation = next_generation;
                                cudaExternalSemaphoreSignalParams signal_params{};
                                signal_params.params.fence.value = next_generation;
                                cuda_ok(
                                    cudaSignalExternalSemaphoresAsync(
                                        &slot.external_semaphore,
                                        &signal_params,
                                        1,
                                        runtime.snapshot_stream),
                                    "cudaSignalExternalSemaphoresAsync snapshot");
                                runtime.snapshot_generation = next_generation;
                                runtime.steps_since_snapshot = 0;
                            }
                        }
                    }
                }
            }

            playback.step_once = false;

            for (size_t slot_index = 0; slot_index < snapshot_slots.size(); ++slot_index) {
                auto& slot = snapshot_slots[slot_index];
                if (!slot.pending) {
                    continue;
                }
                if (slot.timeline_semaphore.getCounterValue() >= slot.pending_generation) {
                    slot.pending = false;
                    slot.ready_generation = slot.pending_generation;
                    if (slot.ready_generation >= runtime.active_snapshot_generation) {
                        runtime.active_snapshot_generation = slot.ready_generation;
                        runtime.active_snapshot_slot = static_cast<int>(slot_index);
                    }
                }
            }

            const auto field = active_snapshot();
            renderer.draw_renderer_ui(field);

            const bool submitted = renderer.render_frame(field);
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
