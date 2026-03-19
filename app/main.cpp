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

    struct StableStepConfig {
        int32_t nx = 96;
        int32_t ny = 96;
        int32_t nz = 96;
        float dt = 1.0f / 60.0f;
        float cell_size = 1.0f;
        float viscosity = 0.00015f;
        float diffusion = 0.00005f;
        int32_t diffuse_iterations = 24;
        int32_t pressure_iterations = 96;
        int32_t block_x = 8;
        int32_t block_y = 8;
        int32_t block_z = 8;
    };

    struct StableSettings {
        StableStepConfig desc{};
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

    };

    struct StableRuntime {
        cudaStream_t sim_stream = nullptr;
        void* workspace = nullptr;
        uint64_t workspace_bytes = 0;
        float* density = nullptr;
        float* velocity_x = nullptr;
        float* velocity_y = nullptr;
        float* velocity_z = nullptr;
    };

    struct VisualStepConfig {
        int32_t nx = 64;
        int32_t ny = 96;
        int32_t nz = 64;
        float dt = 1.0f / 90.0f;
        float cell_size = 1.0f;
        float ambient_temperature = 0.0f;
        float density_buoyancy = 0.045f;
        float temperature_buoyancy = 0.12f;
        float vorticity_epsilon = 2.0f;
        int32_t pressure_iterations = 80;
        int32_t block_x = 8;
        int32_t block_y = 8;
        int32_t block_z = 4;
        uint32_t use_monotonic_cubic = 1u;
    };

    struct VisualSettings {
        VisualStepConfig desc{};
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

    };

    struct VisualRuntime {
        cudaStream_t sim_stream = nullptr;
        void* workspace = nullptr;
        uint64_t workspace_bytes = 0;
        float* density = nullptr;
        float* temperature = nullptr;
        float* velocity_x = nullptr;
        float* velocity_y = nullptr;
        float* velocity_z = nullptr;
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
        VisualSettings visual_settings{};
        VisualRuntime visual_runtime{};
        ViewerRuntime viewer_runtime{};
        std::vector<SnapshotSlot> snapshot_slots{};

        auto scalar_bytes = [](const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
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
            if (stable_runtime.workspace != nullptr) {
                cudaFree(stable_runtime.workspace);
                stable_runtime.workspace = nullptr;
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
            stable_runtime.workspace_bytes = 0;
            if (stable_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(stable_runtime.sim_stream);
                stable_runtime.sim_stream = nullptr;
            }
        };

        auto destroy_visual_backend = [&]() {
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamSynchronize(visual_runtime.sim_stream);
            }
            if (visual_runtime.workspace != nullptr) {
                cudaFree(visual_runtime.workspace);
                visual_runtime.workspace = nullptr;
            }
            if (visual_runtime.density != nullptr) {
                cudaFree(visual_runtime.density);
                visual_runtime.density = nullptr;
            }
            if (visual_runtime.temperature != nullptr) {
                cudaFree(visual_runtime.temperature);
                visual_runtime.temperature = nullptr;
            }
            if (visual_runtime.velocity_x != nullptr) {
                cudaFree(visual_runtime.velocity_x);
                visual_runtime.velocity_x = nullptr;
            }
            if (visual_runtime.velocity_y != nullptr) {
                cudaFree(visual_runtime.velocity_y);
                visual_runtime.velocity_y = nullptr;
            }
            if (visual_runtime.velocity_z != nullptr) {
                cudaFree(visual_runtime.velocity_z);
                visual_runtime.velocity_z = nullptr;
            }
            visual_runtime.workspace_bytes = 0;
            if (visual_runtime.sim_stream != nullptr) {
                cudaStreamDestroy(visual_runtime.sim_stream);
                visual_runtime.sim_stream = nullptr;
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

        auto snapshot_current_field_to_slot = [&](const int slot_index, const char* tag) {
            nvtx3::scoped_range range{tag};
            auto& slot = snapshot_slots.at(static_cast<size_t>(slot_index));
            const auto& field = current_field();
            const auto grid = current_grid();

            if (backend_kind == BackendKind::StableFluids001) {
                if (field.id == FieldId::Density) {
                    cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, stable_runtime.density, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, stable_runtime.sim_stream), "cudaMemcpyAsync stable density snapshot");
                } else {
                    stable_ok(stable_fluids_compute_velocity_magnitude_async(stable_runtime.velocity_x, viewer_runtime.field_bytes, stable_runtime.velocity_y, viewer_runtime.field_bytes, stable_runtime.velocity_z, viewer_runtime.field_bytes, slot.cuda_ptr, viewer_runtime.field_bytes, static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz), grid.cell_size, stable_runtime.sim_stream), "stable_fluids_compute_velocity_magnitude_async");
                }
            } else {
                if (field.id == FieldId::Density) {
                    cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.density, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual density snapshot");
                } else if (field.id == FieldId::Temperature) {
                    cuda_ok(cudaMemcpyAsync(slot.cuda_ptr, visual_runtime.temperature, viewer_runtime.field_bytes, cudaMemcpyDeviceToDevice, visual_runtime.sim_stream), "cudaMemcpyAsync visual temperature snapshot");
                } else {
                    smoke_ok(visual_simulation_of_smoke_compute_velocity_magnitude_async(visual_runtime.velocity_x, visual_simulation_of_smoke_velocity_x_bytes(static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz)), visual_runtime.velocity_y, visual_simulation_of_smoke_velocity_y_bytes(static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz)), visual_runtime.velocity_z, visual_simulation_of_smoke_velocity_z_bytes(static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz)), slot.cuda_ptr, viewer_runtime.field_bytes, static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny), static_cast<int32_t>(grid.nz), grid.cell_size, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream), "visual_simulation_of_smoke_compute_velocity_magnitude_async");
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
                viewer_runtime.field_bytes = scalar_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                stable_runtime.workspace_bytes = stable_fluids_workspace_bytes(stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz);
                cuda_ok(cudaStreamCreateWithFlags(&stable_runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags stable_stream");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.density), viewer_runtime.field_bytes), "cudaMalloc stable density");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_x), viewer_runtime.field_bytes), "cudaMalloc stable velocity_x");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_y), viewer_runtime.field_bytes), "cudaMalloc stable velocity_y");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&stable_runtime.velocity_z), viewer_runtime.field_bytes), "cudaMalloc stable velocity_z");
                cuda_ok(cudaMalloc(&stable_runtime.workspace, stable_runtime.workspace_bytes), "cudaMalloc stable workspace");
            } else {
                viewer_runtime.field_bytes = visual_simulation_of_smoke_scalar_field_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                visual_runtime.workspace_bytes = visual_simulation_of_smoke_workspace_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz);
                cuda_ok(cudaStreamCreateWithFlags(&visual_runtime.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags visual_stream");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.density), viewer_runtime.field_bytes), "cudaMalloc visual density");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.temperature), viewer_runtime.field_bytes), "cudaMalloc visual temperature");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.velocity_x), visual_simulation_of_smoke_velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz)), "cudaMalloc visual velocity_x");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.velocity_y), visual_simulation_of_smoke_velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz)), "cudaMalloc visual velocity_y");
                cuda_ok(cudaMalloc(reinterpret_cast<void**>(&visual_runtime.velocity_z), visual_simulation_of_smoke_velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz)), "cudaMalloc visual velocity_z");
                cuda_ok(cudaMalloc(&visual_runtime.workspace, visual_runtime.workspace_bytes), "cudaMalloc visual workspace");
            }

            create_snapshot_slots();

            if (backend_kind == BackendKind::StableFluids001) {
                stable_ok(stable_fluids_clear_async(stable_runtime.density, viewer_runtime.field_bytes, stable_runtime.velocity_x, viewer_runtime.field_bytes, stable_runtime.velocity_y, viewer_runtime.field_bytes, stable_runtime.velocity_z, viewer_runtime.field_bytes, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, stable_settings.desc.cell_size, stable_runtime.sim_stream), "stable_fluids_clear_async");
                if (stable_settings.emit_density) {
                    const float center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx);
                    const float center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny);
                    const float center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz);
                    stable_ok(stable_fluids_add_density_splat_async(stable_runtime.density, viewer_runtime.field_bytes, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, stable_settings.desc.cell_size, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount, stable_runtime.sim_stream), "stable_fluids_add_density_splat_async");
                }
                if (stable_settings.emit_force) {
                    const float center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx);
                    const float center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny);
                    const float center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz);
                    stable_ok(stable_fluids_add_force_splat_async(stable_runtime.velocity_x, viewer_runtime.field_bytes, stable_runtime.velocity_y, viewer_runtime.field_bytes, stable_runtime.velocity_z, viewer_runtime.field_bytes, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, stable_settings.desc.cell_size, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.force_x, stable_settings.force_y, stable_settings.force_z, stable_runtime.sim_stream), "stable_fluids_add_force_splat_async");
                }
                if (stable_settings.emit_density || stable_settings.emit_force) {
                    stable_ok(stable_fluids_step_async(stable_runtime.density, viewer_runtime.field_bytes, stable_runtime.velocity_x, viewer_runtime.field_bytes, stable_runtime.velocity_y, viewer_runtime.field_bytes, stable_runtime.velocity_z, viewer_runtime.field_bytes, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, stable_settings.desc.cell_size, stable_runtime.workspace, stable_runtime.workspace_bytes, stable_settings.desc.dt, stable_settings.desc.viscosity, stable_settings.desc.diffusion, stable_settings.desc.diffuse_iterations, stable_settings.desc.pressure_iterations, stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z, stable_runtime.sim_stream), "stable_fluids_step_async");
                }
            } else {
                smoke_ok(visual_simulation_of_smoke_clear_async(visual_runtime.density, viewer_runtime.field_bytes, visual_runtime.temperature, viewer_runtime.field_bytes, visual_runtime.velocity_x, visual_simulation_of_smoke_velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_y, visual_simulation_of_smoke_velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_z, visual_simulation_of_smoke_velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, visual_settings.desc.cell_size, visual_runtime.sim_stream), "visual_simulation_of_smoke_clear_async");
                if (visual_settings.emit_source) {
                    const float center_x = visual_settings.source_u * static_cast<float>(visual_settings.desc.nx);
                    const float center_y = visual_settings.source_v * static_cast<float>(visual_settings.desc.ny);
                    const float center_z = visual_settings.source_w * static_cast<float>(visual_settings.desc.nz);
                    smoke_ok(visual_simulation_of_smoke_add_source_async(visual_runtime.density, viewer_runtime.field_bytes, visual_runtime.temperature, viewer_runtime.field_bytes, visual_runtime.velocity_x, visual_simulation_of_smoke_velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_y, visual_simulation_of_smoke_velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_z, visual_simulation_of_smoke_velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, visual_settings.desc.cell_size, center_x, center_y, center_z, visual_settings.source_radius, visual_settings.density_amount, visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream), "visual_simulation_of_smoke_add_source_async");
                    smoke_ok(visual_simulation_of_smoke_step_async(visual_runtime.density, viewer_runtime.field_bytes, visual_runtime.temperature, viewer_runtime.field_bytes, visual_runtime.velocity_x, visual_simulation_of_smoke_velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_y, visual_simulation_of_smoke_velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_z, visual_simulation_of_smoke_velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, visual_settings.desc.cell_size, visual_runtime.workspace, visual_runtime.workspace_bytes, visual_settings.desc.dt, visual_settings.desc.ambient_temperature, visual_settings.desc.density_buoyancy, visual_settings.desc.temperature_buoyancy, visual_settings.desc.vorticity_epsilon, visual_settings.desc.pressure_iterations, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_settings.desc.use_monotonic_cubic, visual_runtime.sim_stream), "visual_simulation_of_smoke_step_async");
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
                                const float center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx);
                                const float center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny);
                                const float center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz);
                                stable_ok(stable_fluids_add_density_splat_async(stable_runtime.density, viewer_runtime.field_bytes, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, stable_settings.desc.cell_size, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.density_amount, stable_runtime.sim_stream), "stable_fluids_add_density_splat_async");
                            }
                            if (stable_settings.emit_force) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_force"};
                                const float center_x = stable_settings.source_u * static_cast<float>(stable_settings.desc.nx);
                                const float center_y = stable_settings.source_v * static_cast<float>(stable_settings.desc.ny);
                                const float center_z = stable_settings.source_w * static_cast<float>(stable_settings.desc.nz);
                                stable_ok(stable_fluids_add_force_splat_async(stable_runtime.velocity_x, viewer_runtime.field_bytes, stable_runtime.velocity_y, viewer_runtime.field_bytes, stable_runtime.velocity_z, viewer_runtime.field_bytes, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, stable_settings.desc.cell_size, center_x, center_y, center_z, stable_settings.source_radius, stable_settings.force_x, stable_settings.force_y, stable_settings.force_z, stable_runtime.sim_stream), "stable_fluids_add_force_splat_async");
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                stable_ok(stable_fluids_step_async(stable_runtime.density, viewer_runtime.field_bytes, stable_runtime.velocity_x, viewer_runtime.field_bytes, stable_runtime.velocity_y, viewer_runtime.field_bytes, stable_runtime.velocity_z, viewer_runtime.field_bytes, stable_settings.desc.nx, stable_settings.desc.ny, stable_settings.desc.nz, stable_settings.desc.cell_size, stable_runtime.workspace, stable_runtime.workspace_bytes, stable_settings.desc.dt, stable_settings.desc.viscosity, stable_settings.desc.diffusion, stable_settings.desc.diffuse_iterations, stable_settings.desc.pressure_iterations, stable_settings.desc.block_x, stable_settings.desc.block_y, stable_settings.desc.block_z, stable_runtime.sim_stream), "stable_fluids_step_async");
                            }
                        } else {
                            if (visual_settings.emit_source) {
                                nvtx3::scoped_range range{"smoke_app.simulation.add_source"};
                                const float center_x = visual_settings.source_u * static_cast<float>(visual_settings.desc.nx);
                                const float center_y = visual_settings.source_v * static_cast<float>(visual_settings.desc.ny);
                                const float center_z = visual_settings.source_w * static_cast<float>(visual_settings.desc.nz);
                                smoke_ok(visual_simulation_of_smoke_add_source_async(visual_runtime.density, viewer_runtime.field_bytes, visual_runtime.temperature, viewer_runtime.field_bytes, visual_runtime.velocity_x, visual_simulation_of_smoke_velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_y, visual_simulation_of_smoke_velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_z, visual_simulation_of_smoke_velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, visual_settings.desc.cell_size, center_x, center_y, center_z, visual_settings.source_radius, visual_settings.density_amount, visual_settings.temperature_amount, visual_settings.velocity_x, visual_settings.velocity_y, visual_settings.velocity_z, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_runtime.sim_stream), "visual_simulation_of_smoke_add_source_async");
                            }
                            {
                                nvtx3::scoped_range range{"smoke_app.simulation.step"};
                                smoke_ok(visual_simulation_of_smoke_step_async(visual_runtime.density, viewer_runtime.field_bytes, visual_runtime.temperature, viewer_runtime.field_bytes, visual_runtime.velocity_x, visual_simulation_of_smoke_velocity_x_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_y, visual_simulation_of_smoke_velocity_y_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_runtime.velocity_z, visual_simulation_of_smoke_velocity_z_bytes(visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz), visual_settings.desc.nx, visual_settings.desc.ny, visual_settings.desc.nz, visual_settings.desc.cell_size, visual_runtime.workspace, visual_runtime.workspace_bytes, visual_settings.desc.dt, visual_settings.desc.ambient_temperature, visual_settings.desc.density_buoyancy, visual_settings.desc.temperature_buoyancy, visual_settings.desc.vorticity_epsilon, visual_settings.desc.pressure_iterations, visual_settings.desc.block_x, visual_settings.desc.block_y, visual_settings.desc.block_z, visual_settings.desc.use_monotonic_cubic, visual_runtime.sim_stream), "visual_simulation_of_smoke_step_async");
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
