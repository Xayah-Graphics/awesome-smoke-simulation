#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <imgui.h>
#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include <vulkan/vulkan_raii.hpp>

#include "stable_fluids.h"

import std;
import vk.camera;
import vk.context;
import vk.frame;
import vk.imgui;
import vk.math;
import vk.memory;
import vk.pipeline;
import vk.swapchain;

namespace {

    constexpr uint32_t frames_in_flight = 2;
    constexpr uint32_t snapshot_slot_count = 4;

    struct alignas(16) uvec4 {
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t w;
    };

    struct alignas(16) PushConstants {
        vk::math::vec4 eye{};
        vk::math::vec4 right{};
        vk::math::vec4 up{};
        vk::math::vec4 forward{};
        vk::math::vec4 volume_min{};
        vk::math::vec4 volume_max{};
        vk::math::vec4 smoke_color{};
        vk::math::vec4 params0{};
        uvec4 params1{};
    };

    static_assert(sizeof(uvec4) == 16);
    static_assert(sizeof(PushConstants) == 144);

    struct UiState {
        bool placement_debug_mode = false;
        bool paused = false;
        bool step_once = false;
        bool emit_density = true;
        bool emit_force = true;
        bool reset_fields = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval = 2;
        int march_steps = 96;
        float density_radius = 5.0f;
        float density_amount = 1.2f;
        float force_radius = 6.0f;
        float force_x = 0.0f;
        float force_y = 1.8f;
        float force_z = 0.0f;
        float density_scale = 1.0f;
        float absorption = 1.4f;
        float smoke_r = 0.88f;
        float smoke_g = 0.92f;
        float smoke_b = 0.96f;
        float source_x = 64.0f;
        float source_y = 28.0f;
        float source_z = 64.0f;
    };

    struct WindowState {
        bool* resize_requested = nullptr;
        float scroll = 0.0f;
        bool first_mouse = true;
        double last_x = 0.0;
        double last_y = 0.0;
    };

    struct SimulationState {
        StableFluidsContext* context = nullptr;
        StableFluidsFieldSetDesc fields{};
        StableFluidsContextDesc desc{};
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
        vk::DescriptorSet descriptor_set{nullptr};
        StableFluidsBufferView buffer_view{};
        cudaExternalMemory_t external_memory = nullptr;
        cudaExternalSemaphore_t external_semaphore = nullptr;
        void* cuda_ptr = nullptr;
        bool pending = false;
        uint64_t pending_generation = 0;
        uint64_t ready_generation = 0;
        uint64_t last_used_submit_serial = 0;
    };

} // namespace

int main() {
    using namespace vk;

    auto cuda_ok = [](cudaError_t status, const char* what) {
        if (status == cudaSuccess) {
            return;
        }
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    };

    auto stable_ok = [](int32_t code, const char* what, const StableFluidsContext* context = nullptr) {
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
    };

    auto make_device_buffer_view = [](void* data, const uint64_t size_bytes) {
        StableFluidsBufferView view{};
        view.data = data;
        view.size_bytes = size_bytes;
        view.format = STABLE_FLUIDS_BUFFER_FORMAT_F32;
        view.memory_type = STABLE_FLUIDS_MEMORY_TYPE_CUDA_DEVICE;
        return view;
    };

    auto [vkctx, sctx] = context::setup_vk_context_glfw("awesome-smoke-simulation", "smoke-visualizer");
    GLFWwindow* window = sctx.window.get();

    const auto timeline_features = vkctx.physical_device.getFeatures2<PhysicalDeviceFeatures2, PhysicalDeviceVulkan12Features>();
    if (!timeline_features.get<PhysicalDeviceVulkan12Features>().timelineSemaphore) {
        throw std::runtime_error("smoke-visualizer requires Vulkan timeline semaphore support");
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
        throw std::runtime_error("smoke-visualizer requires CUDA timeline semaphore interop support");
    }

    WindowState window_state{};
    window_state.resize_requested = &sctx.resize_requested;

    glfwSetWindowUserPointer(window, &window_state);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* raw_window, int, int) {
        auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(raw_window));
        if (state != nullptr && state->resize_requested != nullptr) {
            *state->resize_requested = true;
        }
    });
    glfwSetScrollCallback(window, [](GLFWwindow* raw_window, double, double yoffset) {
        auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(raw_window));
        if (state != nullptr) {
            state->scroll += static_cast<float>(yoffset);
        }
    });

    glfwGetCursorPos(window, &window_state.last_x, &window_state.last_y);

    auto sc = swapchain::setup_swapchain(vkctx, sctx);
    auto frames = frame::create_frame_system(vkctx, sc, frames_in_flight);
    auto imgui_sys = imgui::create(vkctx, window, sc.format, frames_in_flight, static_cast<uint32_t>(sc.images.size()));

    camera::Camera camera;
    camera::CameraConfig camera_config{};
    camera_config.orbit_rotate_sens = 0.005f;
    camera_config.orbit_zoom_sens = 0.12f;
    camera.set_config(camera_config);
    camera.home();

    UiState ui{};
    ui.paused = false;
    ui.absorption = 1.4f;
    ui.density_scale = 1.0f;
    ui.snapshot_interval = 3;
    ui.march_steps = 64;
    ui.smoke_r = 0.92f;
    ui.smoke_g = 0.94f;
    ui.smoke_b = 0.98f;
    ui.source_x = 64.0f;
    ui.source_y = 28.0f;
    ui.source_z = 64.0f;

    SimulationState sim{};
    sim.desc = stable_fluids_context_desc_default();
    sim.desc.nx = 96;
    sim.desc.ny = 96;
    sim.desc.nz = 96;
    sim.desc.dt = 1.0f / 90.0f;
    sim.desc.viscosity = 0.00012f;
    sim.desc.diffusion = 0.00003f;
    sim.desc.diffuse_iterations = 8;
    sim.desc.pressure_iterations = 24;
    sim.desc.pressure_tolerance = 1.0e-5f;

    std::vector<SnapshotSlot> snapshot_slots;

    DescriptorSetLayoutBinding density_binding{
        .binding = 0,
        .descriptorType = DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = ShaderStageFlagBits::eFragment,
    };
    DescriptorSetLayoutCreateInfo density_layout_ci{
        .bindingCount = 1,
        .pBindings = &density_binding,
    };
    raii::DescriptorSetLayout density_set_layout{vkctx.device, density_layout_ci};

    DescriptorPoolSize density_pool_size{
        .type = DescriptorType::eStorageBuffer,
        .descriptorCount = snapshot_slot_count,
    };
    DescriptorPoolCreateInfo density_pool_ci{
        .flags = DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = snapshot_slot_count,
        .poolSizeCount = 1,
        .pPoolSizes = &density_pool_size,
    };
    raii::DescriptorPool density_descriptor_pool{vkctx.device, density_pool_ci};

    std::vector<DescriptorSetLayout> density_layouts(snapshot_slot_count, *density_set_layout);
    DescriptorSetAllocateInfo density_alloc_info{
        .descriptorPool = *density_descriptor_pool,
        .descriptorSetCount = static_cast<uint32_t>(density_layouts.size()),
        .pSetLayouts = density_layouts.data(),
    };
    auto density_descriptor_sets = vkctx.device.allocateDescriptorSets(density_alloc_info);

    const std::filesystem::path shader_path = std::filesystem::path(SMOKE_SIM_SHADER_DIR) / "smoke_volume.spv";
    auto smoke_shader_spv = pipeline::read_file_bytes(shader_path.string());
    auto smoke_shader_module = pipeline::load_shader_module(vkctx.device, smoke_shader_spv);

    std::array<DescriptorSetLayout, 1> pipeline_set_layouts{*density_set_layout};
    pipeline::GraphicsPipelineDesc pipeline_desc{};
    pipeline_desc.color_format = sc.format;
    pipeline_desc.use_depth = false;
    pipeline_desc.use_blend = false;
    pipeline_desc.topology = PrimitiveTopology::eTriangleList;
    pipeline_desc.cull = CullModeFlagBits::eNone;
    pipeline_desc.push_constant_bytes = sizeof(PushConstants);
    pipeline_desc.push_constant_stages = ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment;
    pipeline_desc.set_layouts = pipeline_set_layouts;

    pipeline::VertexInput empty_vertex_input{};
    auto smoke_pipeline = pipeline::create_graphics_pipeline(vkctx.device, empty_vertex_input, pipeline_desc, smoke_shader_module, "vs_main", "fs_main");

    auto destroy_snapshot_slots = [&] {
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
    };

    auto destroy_simulation = [&] {
        if (sim.snapshot_stream != nullptr) {
            cudaStreamSynchronize(sim.snapshot_stream);
        }
        if (sim.sim_stream != nullptr) {
            cudaStreamSynchronize(sim.sim_stream);
        }

        destroy_snapshot_slots();

        if (sim.step_complete_event != nullptr) {
            cudaEventDestroy(sim.step_complete_event);
            sim.step_complete_event = nullptr;
        }
        if (sim.snapshot_stream != nullptr) {
            cudaStreamDestroy(sim.snapshot_stream);
            sim.snapshot_stream = nullptr;
        }
        if (sim.sim_stream != nullptr) {
            cudaStreamDestroy(sim.sim_stream);
            sim.sim_stream = nullptr;
        }

        cudaFree(sim.density);
        cudaFree(sim.velocity_x);
        cudaFree(sim.velocity_y);
        cudaFree(sim.velocity_z);
        sim.density = nullptr;
        sim.velocity_x = nullptr;
        sim.velocity_y = nullptr;
        sim.velocity_z = nullptr;

        if (sim.context != nullptr) {
            stable_fluids_context_destroy(sim.context);
            sim.context = nullptr;
        }

        sim.fields = {};
        sim.field_bytes = 0;
        sim.snapshot_generation = 0;
        sim.submit_serial = 0;
        sim.steps_since_snapshot = 0;
        sim.active_snapshot_slot = -1;
        sim.active_snapshot_generation = 0;
    };

    auto recreate_simulation = [&] {
        destroy_simulation();

        sim.context = stable_fluids_context_create(&sim.desc);
        if (sim.context == nullptr) {
            stable_ok(STABLE_FLUIDS_ERROR_RUNTIME, "stable_fluids_context_create", nullptr);
        }

        sim.field_bytes = stable_fluids_context_required_scalar_field_bytes(sim.context);

        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim.density), sim.field_bytes), "cudaMalloc density");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim.velocity_x), sim.field_bytes), "cudaMalloc velocity_x");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim.velocity_y), sim.field_bytes), "cudaMalloc velocity_y");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim.velocity_z), sim.field_bytes), "cudaMalloc velocity_z");

        sim.fields.density = make_device_buffer_view(sim.density, sim.field_bytes);
        sim.fields.velocity_x = make_device_buffer_view(sim.velocity_x, sim.field_bytes);
        sim.fields.velocity_y = make_device_buffer_view(sim.velocity_y, sim.field_bytes);
        sim.fields.velocity_z = make_device_buffer_view(sim.velocity_z, sim.field_bytes);

        cuda_ok(cudaStreamCreateWithFlags(&sim.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags sim_stream");
        cuda_ok(cudaStreamCreateWithFlags(&sim.snapshot_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags snapshot_stream");
        cuda_ok(cudaEventCreateWithFlags(&sim.step_complete_event, cudaEventDisableTiming), "cudaEventCreateWithFlags step_complete_event");

        stable_ok(stable_fluids_fields_clear_async(sim.context, &sim.fields, sim.sim_stream), "stable_fluids_fields_clear_async", sim.context);
        cuda_ok(cudaStreamSynchronize(sim.sim_stream), "cudaStreamSynchronize clear");

        snapshot_slots.clear();
        snapshot_slots.reserve(snapshot_slot_count);

        for (uint32_t slot_index = 0; slot_index < snapshot_slot_count; ++slot_index) {
            SnapshotSlot slot{};
            slot.descriptor_set = density_descriptor_sets.at(slot_index);

            SemaphoreTypeCreateInfo timeline_semaphore_ci{
                .semaphoreType = SemaphoreType::eTimeline,
                .initialValue = 0,
            };
            ExportSemaphoreCreateInfo export_semaphore_ci{
                .pNext = &timeline_semaphore_ci,
                .handleTypes = ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32,
            };
            SemaphoreCreateInfo semaphore_ci{
                .pNext = &export_semaphore_ci,
            };
            slot.timeline_semaphore = raii::Semaphore{vkctx.device, semaphore_ci};

            ExternalMemoryBufferCreateInfo external_buffer_ci{
                .handleTypes = ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
            };
            BufferCreateInfo buffer_ci{
                .pNext = &external_buffer_ci,
                .size = sim.field_bytes,
                .usage = BufferUsageFlagBits::eStorageBuffer | BufferUsageFlagBits::eTransferDst | BufferUsageFlagBits::eTransferSrc,
                .sharingMode = SharingMode::eExclusive,
            };
            slot.buffer = raii::Buffer{vkctx.device, buffer_ci};

            const MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
            ExportMemoryAllocateInfo export_memory_ci{
                .handleTypes = ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
            };
            MemoryAllocateInfo alloc_ci{
                .pNext = &export_memory_ci,
                .allocationSize = requirements.size,
                .memoryTypeIndex = memory::find_memory_type(vkctx.physical_device, requirements.memoryTypeBits, MemoryPropertyFlagBits::eDeviceLocal),
            };
            slot.memory = raii::DeviceMemory{vkctx.device, alloc_ci};
            slot.buffer.bindMemory(*slot.memory, 0);

#if defined(_WIN32)
            MemoryGetWin32HandleInfoKHR handle_info{
                .memory = *slot.memory,
                .handleType = ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
            };
            HANDLE memory_handle = vkctx.device.getMemoryWin32HandleKHR(handle_info);
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
            buffer_desc.size = sim.field_bytes;
            cuda_ok(cudaExternalMemoryGetMappedBuffer(&slot.cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");

            SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                .semaphore = *slot.timeline_semaphore,
                .handleType = ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32,
            };
            HANDLE semaphore_handle = vkctx.device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
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

            slot.buffer_view = make_device_buffer_view(slot.cuda_ptr, sim.field_bytes);

            DescriptorBufferInfo density_info{
                .buffer = *slot.buffer,
                .offset = 0,
                .range = sim.field_bytes,
            };
            WriteDescriptorSet density_write{
                .dstSet = slot.descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = DescriptorType::eStorageBuffer,
                .pBufferInfo = &density_info,
            };
            vkctx.device.updateDescriptorSets(density_write, {});

            snapshot_slots.push_back(std::move(slot));
        }

        ui.source_x = static_cast<float>(sim.desc.nx) * 0.5f;
        ui.source_y = static_cast<float>(sim.desc.ny) * 0.22f;
        ui.source_z = static_cast<float>(sim.desc.nz) * 0.5f;

        camera::CameraState camera_state = camera.state();
        camera_state.mode = camera::Mode::Orbit;
        camera_state.orbit.target = {
            sim.desc.nx * sim.desc.cell_size * 0.5f,
            sim.desc.ny * sim.desc.cell_size * 0.5f,
            sim.desc.nz * sim.desc.cell_size * 0.5f,
            0.0f,
        };
        camera_state.orbit.distance = std::max({sim.desc.nx, sim.desc.ny, sim.desc.nz}) * sim.desc.cell_size * 2.15f;
        camera_state.orbit.yaw_rad = -0.78539816339f;
        camera_state.orbit.pitch_rad = -0.43633231299f;
        camera.set_state(camera_state);

        const size_t scalar_count = static_cast<size_t>(sim.field_bytes / sizeof(float));
        std::vector<float> filled_density(scalar_count, ui.placement_debug_mode ? 1.0f : 0.0f);
        std::vector<float> zero_field(scalar_count, 0.0f);

        cuda_ok(cudaMemcpyAsync(sim.density, filled_density.data(), sim.field_bytes, cudaMemcpyHostToDevice, sim.sim_stream), "cudaMemcpyAsync initial density");
        cuda_ok(cudaMemcpyAsync(sim.velocity_x, zero_field.data(), sim.field_bytes, cudaMemcpyHostToDevice, sim.sim_stream), "cudaMemcpyAsync initial velocity_x");
        cuda_ok(cudaMemcpyAsync(sim.velocity_y, zero_field.data(), sim.field_bytes, cudaMemcpyHostToDevice, sim.sim_stream), "cudaMemcpyAsync initial velocity_y");
        cuda_ok(cudaMemcpyAsync(sim.velocity_z, zero_field.data(), sim.field_bytes, cudaMemcpyHostToDevice, sim.sim_stream), "cudaMemcpyAsync initial velocity_z");
        cuda_ok(cudaStreamSynchronize(sim.sim_stream), "cudaStreamSynchronize initial field upload");

        if (!ui.placement_debug_mode) {
            StableFluidsDensitySplatDesc density_splat{};
            density_splat.center_x = ui.source_x;
            density_splat.center_y = ui.source_y;
            density_splat.center_z = ui.source_z;
            density_splat.radius = ui.density_radius;
            density_splat.amount = ui.density_amount;
            stable_ok(stable_fluids_fields_add_density_splat_async(sim.context, &sim.fields, &density_splat, sim.sim_stream), "stable_fluids_fields_add_density_splat_async", sim.context);

            StableFluidsForceSplatDesc force_splat{};
            force_splat.center_x = ui.source_x;
            force_splat.center_y = ui.source_y;
            force_splat.center_z = ui.source_z;
            force_splat.radius = ui.force_radius;
            force_splat.force_x = ui.force_x;
            force_splat.force_y = ui.force_y;
            force_splat.force_z = ui.force_z;
            stable_ok(stable_fluids_fields_add_force_splat_async(sim.context, &sim.fields, &force_splat, sim.sim_stream), "stable_fluids_fields_add_force_splat_async", sim.context);
            stable_ok(stable_fluids_fields_step_async(sim.context, &sim.fields, sim.sim_stream), "stable_fluids_fields_step_async", sim.context);
            cuda_ok(cudaEventRecord(sim.step_complete_event, sim.sim_stream), "cudaEventRecord initial_step_complete_event");
            cuda_ok(cudaStreamWaitEvent(sim.snapshot_stream, sim.step_complete_event, 0), "cudaStreamWaitEvent initial_snapshot");
            stable_ok(stable_fluids_fields_snapshot_density_async(sim.context, &sim.fields, snapshot_slots[0].buffer_view, sim.snapshot_stream), "stable_fluids_fields_snapshot_density_async", sim.context);
        } else {
            cuda_ok(cudaMemcpyAsync(snapshot_slots[0].cuda_ptr, filled_density.data(), sim.field_bytes, cudaMemcpyHostToDevice, sim.snapshot_stream), "cudaMemcpyAsync placement snapshot");
        }

        const uint64_t initial_generation = sim.snapshot_generation + 1;
        cudaExternalSemaphoreSignalParams initial_signal_params{};
        initial_signal_params.params.fence.value = initial_generation;
        cuda_ok(
            cudaSignalExternalSemaphoresAsync(
                &snapshot_slots[0].external_semaphore,
                &initial_signal_params,
                1,
                sim.snapshot_stream),
            "cudaSignalExternalSemaphoresAsync initial_snapshot");
        cuda_ok(cudaStreamSynchronize(sim.snapshot_stream), "cudaStreamSynchronize initial_snapshot");
        sim.snapshot_generation = initial_generation;
        snapshot_slots[0].ready_generation = initial_generation;
        sim.active_snapshot_slot = 0;
        sim.active_snapshot_generation = snapshot_slots[0].ready_generation;
        sim.steps_since_snapshot = 0;
    };

    auto recreate_swapchain = [&] {
        swapchain::recreate_swapchain(vkctx, sctx, sc);
        frame::on_swapchain_recreated(vkctx, sc, frames);
        const uint32_t image_count = static_cast<uint32_t>(sc.images.size());
        if (imgui_sys.image_count != image_count || imgui_sys.min_image_count != image_count || imgui_sys.color_format != sc.format) {
            const bool docking = imgui_sys.docking;
            const bool viewports = imgui_sys.viewports;
            imgui::shutdown(imgui_sys);
            imgui_sys = imgui::create(vkctx, window, sc.format, image_count, image_count, docking, viewports);
        }
        sctx.resize_requested = false;
    };

    recreate_simulation();

    auto last_frame_time = std::chrono::steady_clock::now();
    uint32_t frame_index = 0;

    try {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            const auto now = std::chrono::steady_clock::now();
            const float dt_seconds = std::chrono::duration<float>(now - last_frame_time).count();
            last_frame_time = now;

            if (sctx.resize_requested) {
                recreate_swapchain();
            }

            imgui::begin_frame();

            double mouse_x = 0.0;
            double mouse_y = 0.0;
            glfwGetCursorPos(window, &mouse_x, &mouse_y);

            float mouse_dx = 0.0f;
            float mouse_dy = 0.0f;
            if (window_state.first_mouse) {
                window_state.first_mouse = false;
            } else {
                mouse_dx = static_cast<float>(mouse_x - window_state.last_x);
                mouse_dy = static_cast<float>(mouse_y - window_state.last_y);
            }
            window_state.last_x = mouse_x;
            window_state.last_y = mouse_y;

            auto& io = ImGui::GetIO();
            camera::CameraInput camera_input{};
            if (!io.WantCaptureMouse) {
                camera_input.lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
                camera_input.mmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
                camera_input.rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
                camera_input.mouse_dx = mouse_dx;
                camera_input.mouse_dy = mouse_dy;
                camera_input.scroll = window_state.scroll;
            }
            if (!io.WantCaptureKeyboard) {
                camera_input.forward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
                camera_input.backward = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
                camera_input.left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
                camera_input.right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
                camera_input.up = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;
                camera_input.down = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
                camera_input.shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
                camera_input.ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
                camera_input.alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
                camera_input.space = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
            }
            window_state.scroll = 0.0f;

            camera.update(dt_seconds, sc.extent.width, sc.extent.height, camera_input);

            ImGui::Begin("Smoke Control");
            const bool placement_mode_changed = ImGui::Checkbox("Placement Debug Mode", &ui.placement_debug_mode);
            if (ui.placement_debug_mode) {
                ImGui::TextUnformatted("Simulation disabled. Rendering a fully filled debug volume.");
            }
            ImGui::Checkbox("Pause Simulation", &ui.paused);
            if (ImGui::Button("Single Step")) {
                ui.step_once = true;
            }
            if (ImGui::Button("Reset Fields")) {
                ui.reset_fields = true;
            }
            ImGui::Separator();
            ImGui::SliderInt("Sim Steps / Frame", &ui.sim_steps_per_frame, 1, 8);
            ImGui::SliderInt("Snapshot Interval", &ui.snapshot_interval, 1, 8);
            ImGui::SliderInt("March Steps", &ui.march_steps, 24, 192);
            ImGui::SliderFloat("Density Scale", &ui.density_scale, 0.05f, 4.0f, "%.2f");
            ImGui::SliderFloat("Absorption", &ui.absorption, 0.1f, 6.0f, "%.2f");
            ImGui::ColorEdit3("Smoke Color", &ui.smoke_r);
            ImGui::Separator();
            ImGui::Checkbox("Emit Density", &ui.emit_density);
            ImGui::Checkbox("Emit Force", &ui.emit_force);
            ImGui::SliderFloat("Source X", &ui.source_x, 0.0f, static_cast<float>(sim.desc.nx - 1), "%.1f");
            ImGui::SliderFloat("Source Y", &ui.source_y, 0.0f, static_cast<float>(sim.desc.ny - 1), "%.1f");
            ImGui::SliderFloat("Source Z", &ui.source_z, 0.0f, static_cast<float>(sim.desc.nz - 1), "%.1f");
            ImGui::SliderFloat("Density Radius", &ui.density_radius, 1.0f, 16.0f, "%.1f");
            ImGui::SliderFloat("Density Amount", &ui.density_amount, 0.1f, 8.0f, "%.2f");
            ImGui::SliderFloat("Force Radius", &ui.force_radius, 1.0f, 20.0f, "%.1f");
            ImGui::SliderFloat("Force X", &ui.force_x, -4.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Force Y", &ui.force_y, -4.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Force Z", &ui.force_z, -4.0f, 4.0f, "%.2f");
            ImGui::Separator();
            ImGui::Text("Grid: %d x %d x %d", sim.desc.nx, sim.desc.ny, sim.desc.nz);
            ImGui::Text("Active snapshot: %d", sim.active_snapshot_slot);
            ImGui::Text("Snapshot generation: %llu", static_cast<unsigned long long>(sim.active_snapshot_generation));
            ImGui::End();

            if (placement_mode_changed) {
                recreate_simulation();
                ui.step_once = false;
                continue;
            }

            if (ui.reset_fields) {
                recreate_simulation();
                ui.reset_fields = false;
                ui.step_once = false;
                continue;
            }

            const bool run_simulation = !ui.placement_debug_mode && (!ui.paused || ui.step_once);
            if (run_simulation) {
                for (int sim_step = 0; sim_step < ui.sim_steps_per_frame; ++sim_step) {
                    if (ui.emit_density) {
                        StableFluidsDensitySplatDesc density_splat{};
                        density_splat.center_x = ui.source_x;
                        density_splat.center_y = ui.source_y;
                        density_splat.center_z = ui.source_z;
                        density_splat.radius = ui.density_radius;
                        density_splat.amount = ui.density_amount;
                        stable_ok(stable_fluids_fields_add_density_splat_async(sim.context, &sim.fields, &density_splat, sim.sim_stream), "stable_fluids_fields_add_density_splat_async", sim.context);
                    }

                    if (ui.emit_force) {
                        StableFluidsForceSplatDesc force_splat{};
                        force_splat.center_x = ui.source_x;
                        force_splat.center_y = ui.source_y;
                        force_splat.center_z = ui.source_z;
                        force_splat.radius = ui.force_radius;
                        force_splat.force_x = ui.force_x;
                        force_splat.force_y = ui.force_y;
                        force_splat.force_z = ui.force_z;
                        stable_ok(stable_fluids_fields_add_force_splat_async(sim.context, &sim.fields, &force_splat, sim.sim_stream), "stable_fluids_fields_add_force_splat_async", sim.context);
                    }

                    stable_ok(stable_fluids_fields_step_async(sim.context, &sim.fields, sim.sim_stream), "stable_fluids_fields_step_async", sim.context);
                    cuda_ok(cudaEventRecord(sim.step_complete_event, sim.sim_stream), "cudaEventRecord step_complete_event");

                    if (sim.steps_since_snapshot < static_cast<uint32_t>(ui.snapshot_interval)) {
                        ++sim.steps_since_snapshot;
                    }
                    if (sim.steps_since_snapshot >= static_cast<uint32_t>(ui.snapshot_interval)) {
                        int snapshot_slot_index = -1;
                        for (uint32_t slot_index = 0; slot_index < snapshot_slots.size(); ++slot_index) {
                            auto& slot = snapshot_slots[slot_index];
                            if (slot.pending) {
                                continue;
                            }
                            if (static_cast<int>(slot_index) == sim.active_snapshot_slot) {
                                continue;
                            }
                            if (slot.ready_generation != 0 &&
                                sim.submit_serial < slot.last_used_submit_serial + frames.frames_in_flight + 1) {
                                continue;
                            }
                            snapshot_slot_index = static_cast<int>(slot_index);
                            break;
                        }

                        if (snapshot_slot_index >= 0) {
                            auto& slot = snapshot_slots[static_cast<size_t>(snapshot_slot_index)];
                            const uint64_t next_generation = sim.snapshot_generation + 1;
                            cuda_ok(cudaStreamWaitEvent(sim.snapshot_stream, sim.step_complete_event, 0), "cudaStreamWaitEvent snapshot_stream");
                            stable_ok(stable_fluids_fields_snapshot_density_async(sim.context, &sim.fields, slot.buffer_view, sim.snapshot_stream), "stable_fluids_fields_snapshot_density_async", sim.context);
                            slot.pending = true;
                            slot.pending_generation = next_generation;
                            cudaExternalSemaphoreSignalParams signal_params{};
                            signal_params.params.fence.value = next_generation;
                            cuda_ok(
                                cudaSignalExternalSemaphoresAsync(
                                    &slot.external_semaphore,
                                    &signal_params,
                                    1,
                                    sim.snapshot_stream),
                                "cudaSignalExternalSemaphoresAsync snapshot");
                            sim.snapshot_generation = next_generation;
                            sim.steps_since_snapshot = 0;
                        }
                    }
                }
            }
            ui.step_once = false;

            for (size_t slot_index = 0; slot_index < snapshot_slots.size(); ++slot_index) {
                auto& slot = snapshot_slots[slot_index];
                if (!slot.pending) {
                    continue;
                }
                if (slot.timeline_semaphore.getCounterValue() >= slot.pending_generation) {
                    slot.pending = false;
                    slot.ready_generation = slot.pending_generation;
                    if (slot.ready_generation >= sim.active_snapshot_generation) {
                        sim.active_snapshot_generation = slot.ready_generation;
                        sim.active_snapshot_slot = static_cast<int>(slot_index);
                    }
                }
            }

            auto acquire_result = frame::begin_frame(vkctx, sc, frames, frame_index);
            if (acquire_result.need_recreate) {
                recreate_swapchain();
                continue;
            }

            frame::begin_commands(frames, frame_index);
            auto& cmd = frame::cmd(frames, frame_index);

            const uint32_t image_index = acquire_result.image_index;
            const ImageLayout previous_layout = frames.swapchain_image_layout[image_index];
            const ImageMemoryBarrier2 to_color_barrier{
                .srcStageMask = previous_layout == ImageLayout::eUndefined ? PipelineStageFlagBits2::eNone : PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = previous_layout == ImageLayout::eUndefined ? AccessFlags2{} : (AccessFlagBits2::eMemoryRead | AccessFlagBits2::eMemoryWrite),
                .dstStageMask = PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = previous_layout,
                .newLayout = ImageLayout::eColorAttachmentOptimal,
                .image = sc.images[image_index],
                .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmd.pipelineBarrier2(DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &to_color_barrier,
            });
            frames.swapchain_image_layout[image_index] = ImageLayout::eColorAttachmentOptimal;

            ClearValue clear_value{};
            clear_value.color = ClearColorValue{std::array<float, 4>{0.02f, 0.025f, 0.035f, 1.0f}};
            RenderingAttachmentInfo color_attachment{
                .imageView = *sc.image_views[image_index],
                .imageLayout = ImageLayout::eColorAttachmentOptimal,
                .loadOp = AttachmentLoadOp::eClear,
                .storeOp = AttachmentStoreOp::eStore,
                .clearValue = clear_value,
            };
            RenderingInfo rendering_info{
                .renderArea = Rect2D{Offset2D{0, 0}, sc.extent},
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &color_attachment,
            };

            cmd.beginRendering(rendering_info);
            cmd.setViewport(0, Viewport{
                0.0f,
                0.0f,
                static_cast<float>(sc.extent.width),
                static_cast<float>(sc.extent.height),
                0.0f,
                1.0f,
            });
            cmd.setScissor(0, Rect2D{{0, 0}, sc.extent});

            if (sim.active_snapshot_slot >= 0) {
                const auto& matrices = camera.matrices();
                const float half_fov_tan = std::tan(camera.config().fov_y_rad * 0.5f);
                PushConstants push{};
                push.eye = {matrices.eye.x, matrices.eye.y, matrices.eye.z, 1.0f};
                push.right = {matrices.right.x, matrices.right.y, matrices.right.z, 0.0f};
                push.up = {matrices.up.x, matrices.up.y, matrices.up.z, 0.0f};
                push.forward = {-matrices.forward.x, -matrices.forward.y, -matrices.forward.z, 0.0f};
                push.volume_min = {0.0f, 0.0f, 0.0f, 0.0f};
                push.volume_max = {
                    sim.desc.nx * sim.desc.cell_size,
                    sim.desc.ny * sim.desc.cell_size,
                    sim.desc.nz * sim.desc.cell_size,
                    0.0f,
                };
                push.smoke_color = {ui.smoke_r, ui.smoke_g, ui.smoke_b, 1.0f};
                push.params0 = {
                    static_cast<float>(sc.extent.width) / static_cast<float>(std::max(sc.extent.height, 1u)),
                    half_fov_tan,
                    ui.density_scale,
                    ui.absorption,
                };
                push.params1 = {
                    static_cast<uint32_t>(sim.desc.nx),
                    static_cast<uint32_t>(sim.desc.ny),
                    static_cast<uint32_t>(sim.desc.nz),
                    static_cast<uint32_t>(ui.march_steps),
                };

                cmd.bindPipeline(PipelineBindPoint::eGraphics, *smoke_pipeline.pipeline);
                cmd.bindDescriptorSets(PipelineBindPoint::eGraphics, *smoke_pipeline.layout, 0, {snapshot_slots[static_cast<size_t>(sim.active_snapshot_slot)].descriptor_set}, {});
                const ArrayProxy<const PushConstants> push_block(1, &push);
                cmd.pushConstants(*smoke_pipeline.layout, ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment, 0, push_block);
                cmd.draw(3, 1, 0, 0);
            }
            cmd.endRendering();

            imgui::draw_mini_axis_gizmo(camera.matrices().c2w);
            imgui::render(imgui_sys, cmd, sc.extent, *sc.image_views[image_index], ImageLayout::eColorAttachmentOptimal);

            const ImageMemoryBarrier2 to_present_barrier{
                .srcStageMask = PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = PipelineStageFlagBits2::eBottomOfPipe,
                .oldLayout = ImageLayout::eColorAttachmentOptimal,
                .newLayout = ImageLayout::ePresentSrcKHR,
                .image = sc.images[image_index],
                .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmd.pipelineBarrier2(DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &to_present_barrier,
            });
            frames.swapchain_image_layout[image_index] = ImageLayout::ePresentSrcKHR;

            std::array<SemaphoreSubmitInfo, 1> snapshot_waits{};
            std::span<const SemaphoreSubmitInfo> extra_waits{};
            const uint64_t submit_serial = sim.submit_serial + 1;
            if (sim.active_snapshot_slot >= 0) {
                auto& active_slot = snapshot_slots[static_cast<size_t>(sim.active_snapshot_slot)];
                active_slot.last_used_submit_serial = submit_serial;
                snapshot_waits[0] = SemaphoreSubmitInfo{
                    .semaphore = *active_slot.timeline_semaphore,
                    .value = active_slot.ready_generation,
                    .stageMask = PipelineStageFlagBits2::eFragmentShader,
                };
                extra_waits = std::span<const SemaphoreSubmitInfo>(snapshot_waits.data(), snapshot_waits.size());
            }
            const bool need_recreate = frame::end_frame(vkctx, sc, frames, frame_index, image_index, extra_waits);
            sim.submit_serial = submit_serial;

            imgui::end_frame();

            if (need_recreate || sctx.resize_requested) {
                recreate_swapchain();
            }

            frame_index = (frame_index + 1) % frames.frames_in_flight;
        }
    } catch (...) {
        vkctx.device.waitIdle();
        destroy_simulation();
        imgui::shutdown(imgui_sys);
        throw;
    }

    vkctx.device.waitIdle();
    destroy_simulation();
    imgui::shutdown(imgui_sys);
    return 0;
}
