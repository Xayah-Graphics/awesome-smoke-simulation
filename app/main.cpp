#include <cuda_runtime.h>
#include <imgui.h>
#include <nvtx3/nvtx3.hpp>
#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include <vulkan/vulkan_raii.hpp>

#include "stable-fluids.h"
#include "visual-simulation-of-smoke.h"

import app;
import std;
import vk.memory;

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    enum class BackendKind : uint32_t {
        StableFluids001 = 0,
        VisualSmoke002 = 1,
    };

    enum class FieldId : uint32_t {
        Density = 0,
        Temperature = 1,
        VelocityMagnitude = 2,
    };

    struct PlaybackState {
        bool paused = false;
        bool step_once = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval = 2;
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
        uint64_t field_bytes = 0;
        uint64_t snapshot_generation = 0;
        uint64_t submit_serial = 0;
        uint32_t steps_since_snapshot = 0;
        int active_snapshot_slot = -1;
    };

    struct SnapshotSlot {
        vk::raii::Buffer buffer{nullptr};
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Semaphore timeline_semaphore{nullptr};
        vk::raii::DescriptorSet descriptor_set{nullptr};
        cudaExternalMemory_t external_memory = nullptr;
        cudaExternalSemaphore_t external_semaphore = nullptr;
        void* cuda_ptr = nullptr;
        uint64_t ready_generation = 0;
        uint64_t last_used_submit_serial = 0;
    };

    struct StableSettings {
        StableFluidsContextDesc desc = stable_fluids_context_desc_default();
        int selected_field = 0;
        bool emit_density = true;
        bool emit_force = true;
        float source_u = 0.5f;
        float source_v = 0.33f;
        float source_w = 0.5f;
        float source_radius = 5.0f;
        float density_amount = 6.0f;
        float force_x = 1.25f;
        float force_y = 2.5f;
        float force_z = 0.75f;

        StableSettings() {
            desc.nx = 96;
            desc.ny = 96;
            desc.nz = 96;
            desc.dt = 1.0f / 60.0f;
            desc.cell_size = 1.0f;
            desc.viscosity = 0.00015f;
            desc.diffusion = 0.00005f;
            desc.diffuse_iterations = 24;
            desc.pressure_iterations = 96;
            desc.pressure_tolerance = 1.0e-5f;
        }
    };

    struct StableRuntime {
        StableFluidsContext* context = nullptr;
        cudaStream_t sim_stream = nullptr;
        float* density = nullptr;
        float* velocity_x = nullptr;
        float* velocity_y = nullptr;
        float* velocity_z = nullptr;
        StableFluidsFieldSetDesc fields{};
    };

    struct VisualSettings {
        VisualSimulationOfSmokeContextDesc desc = visual_simulation_of_smoke_context_desc_default();
        int selected_field = 0;
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

        VisualSettings() {
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

    struct VisualRuntime {
        VisualSimulationOfSmokeContext* context = nullptr;
        cudaStream_t sim_stream = nullptr;
    };

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

        const uint64_t message_length = context != nullptr ? stable_fluids_context_last_error_length(context) : stable_fluids_last_error_length();
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
        BackendKind backend_kind = BackendKind::StableFluids001;
        StableSettings stable_settings{};
        StableRuntime stable_runtime{};
        VisualSettings visual_settings{};
        VisualRuntime visual_runtime{};
        ViewerRuntime viewer_runtime{};
        std::vector<SnapshotSlot> snapshot_slots{};

        auto stable_make_buffer_view = [](void* data, const uint64_t size_bytes) {
            return StableFluidsBufferView{
                .data = data,
                .size_bytes = size_bytes,
                .format = STABLE_FLUIDS_BUFFER_FORMAT_F32,
                .memory_type = STABLE_FLUIDS_MEMORY_TYPE_CUDA_DEVICE,
            };
        };

        auto visual_make_buffer_view = [](void* data, const uint64_t size_bytes) {
            return VisualSimulationOfSmokeBufferView{
                .data = data,
                .size_bytes = size_bytes,
                .format = VISUAL_SIMULATION_OF_SMOKE_BUFFER_FORMAT_F32,
                .memory_type = VISUAL_SIMULATION_OF_SMOKE_MEMORY_TYPE_CUDA_DEVICE,
            };
        };

        auto current_fields = [&]() -> std::span<const FieldChoice> {
            return backend_kind == BackendKind::StableFluids001
                ? std::span<const FieldChoice>(stable_fields)
                : std::span<const FieldChoice>(visual_fields);
        };

        auto current_field_index = [&]() -> int& {
            return backend_kind == BackendKind::StableFluids001 ? stable_settings.selected_field : visual_settings.selected_field;
        };

        auto current_field = [&]() -> const FieldChoice& {
            auto fields = current_fields();
            auto& selected = current_field_index();
            selected = std::clamp(selected, 0, static_cast<int>(fields.size()) - 1);
            return fields[static_cast<size_t>(selected)];
        };

        auto current_stream = [&]() -> cudaStream_t {
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
                render.mode = app::RenderMode::Smoke;
                render.density_scale = 1.0f;
                render.absorption = 1.4f;
            } else if (field.id == FieldId::Temperature) {
                render.mode = app::RenderMode::Scalar;
                render.scalar_min = 0.0f;
                render.scalar_max = 2.0f;
                render.scalar_opacity = 2.0f;
                render.scalar_low_r = 0.08f;
                render.scalar_low_g = 0.16f;
                render.scalar_low_b = 0.45f;
                render.scalar_high_r = 0.98f;
                render.scalar_high_g = 0.42f;
                render.scalar_high_b = 0.12f;
            } else {
                render.mode = app::RenderMode::Scalar;
                render.scalar_min = 0.0f;
                render.scalar_max = 3.0f;
                render.scalar_opacity = 1.6f;
                render.scalar_low_r = 0.04f;
                render.scalar_low_g = 0.18f;
                render.scalar_low_b = 0.36f;
                render.scalar_high_r = 0.74f;
                render.scalar_high_g = 0.94f;
                render.scalar_high_b = 0.96f;
            }
        };

        auto active_snapshot = [&]() -> std::optional<app::ScalarFieldView> {
            if (viewer_runtime.active_snapshot_slot < 0) {
                return std::nullopt;
            }

            const auto& field = current_field();
            const auto grid = current_grid();
            const auto& slot = snapshot_slots.at(static_cast<size_t>(viewer_runtime.active_snapshot_slot));
            return app::ScalarFieldView{
                .descriptor_set = *slot.descriptor_set,
                .timeline_semaphore = *slot.timeline_semaphore,
                .ready_generation = slot.ready_generation,
                .nx = grid.nx,
                .ny = grid.ny,
                .nz = grid.nz,
                .cell_size = grid.cell_size,
                .semantic = field.semantic,
                .label = field.label,
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
                slot.ready_generation = 0;
                slot.last_used_submit_serial = 0;
            }
            snapshot_slots.clear();
            viewer_runtime.field_bytes = 0;
            viewer_runtime.snapshot_generation = 0;
            viewer_runtime.submit_serial = 0;
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
            stable_runtime.fields = {};
            if (stable_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(stable_runtime.sim_stream);
                stable_runtime.sim_stream = nullptr;
            }
            if (stable_runtime.context != nullptr) {
                stable_fluids_context_destroy(stable_runtime.context);
                stable_runtime.context = nullptr;
            }
        };

        auto destroy_visual_backend = [&]() {
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamSynchronize(visual_runtime.sim_stream);
            }
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(visual_runtime.sim_stream);
                visual_runtime.sim_stream = nullptr;
            }
            if (visual_runtime.context != nullptr) {
                visual_simulation_of_smoke_context_destroy(visual_runtime.context);
                visual_runtime.context = nullptr;
            }
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
                    .size = viewer_runtime.field_bytes,
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
                buffer_desc.size = viewer_runtime.field_bytes;
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

        auto stable_density_splat = [&]() {
            return StableFluidsDensitySplatDesc{
                .center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx),
                .center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny),
                .center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz),
                .radius = stable_settings.source_radius,
                .amount = stable_settings.density_amount,
            };
        };

        auto stable_force_splat = [&]() {
            return StableFluidsForceSplatDesc{
                .center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx),
                .center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny),
                .center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz),
                .radius = stable_settings.source_radius,
                .force_x = stable_settings.force_x,
                .force_y = stable_settings.force_y,
                .force_z = stable_settings.force_z,
            };
        };

        auto visual_source = [&]() {
            return VisualSimulationOfSmokeSourceDesc{
                .center_x = visual_settings.source_u * static_cast<float>(visual_settings.desc.nx),
                .center_y = visual_settings.source_v * static_cast<float>(visual_settings.desc.ny),
                .center_z = visual_settings.source_w * static_cast<float>(visual_settings.desc.nz),
                .radius = visual_settings.source_radius,
                .density_amount = visual_settings.density_amount,
                .temperature_amount = visual_settings.temperature_amount,
                .velocity_x = visual_settings.velocity_x,
                .velocity_y = visual_settings.velocity_y,
                .velocity_z = visual_settings.velocity_z,
            };
        };

        auto snapshot_current_field_to_slot = [&](const int slot_index, const char* tag) {
            nvtx3::scoped_range range{tag};
            auto& slot = snapshot_slots.at(static_cast<size_t>(slot_index));
            const auto& field = current_field();

            if (backend_kind == BackendKind::StableFluids001) {
                const auto destination = stable_make_buffer_view(slot.cuda_ptr, viewer_runtime.field_bytes);
                if (field.id == FieldId::Density) {
                    stable_ok(stable_fluids_fields_snapshot_density_async(stable_runtime.context, &stable_runtime.fields, destination, stable_runtime.sim_stream), "stable_fluids_fields_snapshot_density_async", stable_runtime.context);
                } else {
                    stable_ok(stable_fluids_fields_snapshot_velocity_magnitude_async(stable_runtime.context, &stable_runtime.fields, destination, stable_runtime.sim_stream), "stable_fluids_fields_snapshot_velocity_magnitude_async", stable_runtime.context);
                }
            } else {
                const auto destination = visual_make_buffer_view(slot.cuda_ptr, viewer_runtime.field_bytes);
                if (field.id == FieldId::Density) {
                    smoke_ok(visual_simulation_of_smoke_snapshot_density_async(visual_runtime.context, destination, visual_runtime.sim_stream), "visual_simulation_of_smoke_snapshot_density_async", visual_runtime.context);
                } else if (field.id == FieldId::Temperature) {
                    smoke_ok(visual_simulation_of_smoke_snapshot_temperature_async(visual_runtime.context, destination, visual_runtime.sim_stream), "visual_simulation_of_smoke_snapshot_temperature_async", visual_runtime.context);
                } else {
                    smoke_ok(visual_simulation_of_smoke_snapshot_velocity_magnitude_async(visual_runtime.context, destination, visual_runtime.sim_stream), "visual_simulation_of_smoke_snapshot_velocity_magnitude_async", visual_runtime.context);
                }
            }

            const uint64_t next_generation = viewer_runtime.snapshot_generation + 1;
            cudaExternalSemaphoreSignalParams signal_params{};
            signal_params.params.fence.value = next_generation;
            cuda_ok(cudaSignalExternalSemaphoresAsync(&slot.external_semaphore, &signal_params, 1, current_stream()), "cudaSignalExternalSemaphoresAsync snapshot");
            cuda_ok(cudaStreamSynchronize(current_stream()), "cudaStreamSynchronize snapshot");
            slot.ready_generation = next_generation;
            viewer_runtime.snapshot_generation = next_generation;
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
                stable_runtime.context = stable_fluids_context_create(&stable_settings.desc);
                if (stable_runtime.context == nullptr) {
                    stable_ok(STABLE_FLUIDS_ERROR_RUNTIME, "stable_fluids_context_create");
                }

                viewer_runtime.field_bytes = stable_fluids_context_required_scalar_field_bytes(stable_runtime.context);
                cuda_ok(cudaStreamCreateWithFlags(&stable_runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags stable_stream");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.density), viewer_runtime.field_bytes), "cudaMalloc stable density");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_x), viewer_runtime.field_bytes), "cudaMalloc stable velocity_x");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_y), viewer_runtime.field_bytes), "cudaMalloc stable velocity_y");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_z), viewer_runtime.field_bytes), "cudaMalloc stable velocity_z");
                stable_runtime.fields = StableFluidsFieldSetDesc{
                    .density = stable_make_buffer_view(stable_runtime.density, viewer_runtime.field_bytes),
                    .velocity_x = stable_make_buffer_view(stable_runtime.velocity_x, viewer_runtime.field_bytes),
                    .velocity_y = stable_make_buffer_view(stable_runtime.velocity_y, viewer_runtime.field_bytes),
                    .velocity_z = stable_make_buffer_view(stable_runtime.velocity_z, viewer_runtime.field_bytes),
                };
            } else {
                visual_runtime.context = visual_simulation_of_smoke_context_create(&visual_settings.desc);
                if (visual_runtime.context == nullptr) {
                    smoke_ok(VISUAL_SIMULATION_OF_SMOKE_ERROR_RUNTIME, "visual_simulation_of_smoke_context_create");
                }

                viewer_runtime.field_bytes = visual_simulation_of_smoke_context_required_density_bytes(visual_runtime.context);
                cuda_ok(cudaStreamCreateWithFlags(&visual_runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags visual_stream");
            }

            create_snapshot_slots();

            if (backend_kind == BackendKind::StableFluids001) {
                stable_ok(stable_fluids_fields_clear_async(stable_runtime.context, &stable_runtime.fields, stable_runtime.sim_stream), "stable_fluids_fields_clear_async", stable_runtime.context);
                if (stable_settings.emit_density) {
                    const auto splat = stable_density_splat();
                    stable_ok(stable_fluids_fields_add_density_splat_async(stable_runtime.context, &stable_runtime.fields, &splat, stable_runtime.sim_stream), "stable_fluids_fields_add_density_splat_async", stable_runtime.context);
                }
                if (stable_settings.emit_force) {
                    const auto splat = stable_force_splat();
                    stable_ok(stable_fluids_fields_add_force_splat_async(stable_runtime.context, &stable_runtime.fields, &splat, stable_runtime.sim_stream), "stable_fluids_fields_add_force_splat_async", stable_runtime.context);
                }
                if (stable_settings.emit_density || stable_settings.emit_force) {
                    stable_ok(stable_fluids_fields_step_async(stable_runtime.context, &stable_runtime.fields, stable_runtime.sim_stream), "stable_fluids_fields_step_async", stable_runtime.context);
                }
            } else {
                smoke_ok(visual_simulation_of_smoke_clear_async(visual_runtime.context, visual_runtime.sim_stream), "visual_simulation_of_smoke_clear_async", visual_runtime.context);
                if (visual_settings.emit_source) {
                    const auto source = visual_source();
                    smoke_ok(visual_simulation_of_smoke_add_source_async(visual_runtime.context, &source, visual_runtime.sim_stream), "visual_simulation_of_smoke_add_source_async", visual_runtime.context);
                    smoke_ok(visual_simulation_of_smoke_step_async(visual_runtime.context, visual_runtime.sim_stream), "visual_simulation_of_smoke_step_async", visual_runtime.context);
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
            bool field_changed = false;

            ImGui::Begin("Simulation");
            int backend_index = static_cast<int>(backend_kind);
            const char* backend_labels[] = {
                "001-stable-fluids",
                "002-visual-simulation-of-smoke",
            };
            if (ImGui::Combo("Backend", &backend_index, backend_labels, 2)) {
                backend_kind = static_cast<BackendKind>(backend_index);
                reset_requested = true;
                field_changed = true;
            }

            const auto fields = current_fields();
            auto& selected_field = current_field_index();
            selected_field = std::clamp(selected_field, 0, static_cast<int>(fields.size()) - 1);
            if (ImGui::BeginCombo("Field", fields[static_cast<size_t>(selected_field)].label.data())) {
                for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
                    const bool is_selected = selected_field == i;
                    if (ImGui::Selectable(fields[static_cast<size_t>(i)].label.data(), is_selected)) {
                        selected_field = i;
                        field_changed = true;
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
                ImGui::Checkbox("Emit Density", &stable_settings.emit_density);
                ImGui::Checkbox("Emit Force", &stable_settings.emit_force);
                ImGui::SliderFloat("Source U", &stable_settings.source_u, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source V", &stable_settings.source_v, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source W", &stable_settings.source_w, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Source Radius", &stable_settings.source_radius, 1.0f, 16.0f, "%.1f");
                ImGui::SliderFloat("Density Amount", &stable_settings.density_amount, 0.0f, 12.0f, "%.2f");
                ImGui::SliderFloat("Force X", &stable_settings.force_x, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Force Y", &stable_settings.force_y, -4.0f, 4.0f, "%.2f");
                ImGui::SliderFloat("Force Z", &stable_settings.force_z, -4.0f, 4.0f, "%.2f");
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
                    reset_requested = true;
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
                            if (stable_settings.emit_density) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_density"};
                                const auto splat = stable_density_splat();
                                stable_ok(stable_fluids_fields_add_density_splat_async(stable_runtime.context, &stable_runtime.fields, &splat, stable_runtime.sim_stream), "stable_fluids_fields_add_density_splat_async", stable_runtime.context);
                            }
                            if (stable_settings.emit_force) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_force"};
                                const auto splat = stable_force_splat();
                                stable_ok(stable_fluids_fields_add_force_splat_async(stable_runtime.context, &stable_runtime.fields, &splat, stable_runtime.sim_stream), "stable_fluids_fields_add_force_splat_async", stable_runtime.context);
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                stable_ok(stable_fluids_fields_step_async(stable_runtime.context, &stable_runtime.fields, stable_runtime.sim_stream), "stable_fluids_fields_step_async", stable_runtime.context);
                            }
                        } else {
                            if (visual_settings.emit_source) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_source"};
                                const auto source = visual_source();
                                smoke_ok(visual_simulation_of_smoke_add_source_async(visual_runtime.context, &source, visual_runtime.sim_stream), "visual_simulation_of_smoke_add_source_async", visual_runtime.context);
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                smoke_ok(visual_simulation_of_smoke_step_async(visual_runtime.context, visual_runtime.sim_stream), "visual_simulation_of_smoke_step_async", visual_runtime.context);
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
