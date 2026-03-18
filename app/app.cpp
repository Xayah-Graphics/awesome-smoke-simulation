module;

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

module app;

import std;
import vk.camera;
import vk.context;
import vk.frame;
import vk.imgui;
import vk.math;
import vk.memory;
import vk.pipeline;
import vk.swapchain;

namespace app {

    SmokeApp::SmokeApp() {
        using namespace vk;

        auto [vkctx, sctx] = context::setup_vk_context_glfw("awesome-smoke-simulation", "smoke-visualizer");
        vkctx_ = std::move(vkctx);
        sctx_ = std::move(sctx);
        window_ = sctx_.window.get();

        const auto timeline_features = vkctx_.physical_device.getFeatures2<PhysicalDeviceFeatures2, PhysicalDeviceVulkan12Features>();
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

        window_state_.resize_requested = &sctx_.resize_requested;
        glfwSetWindowUserPointer(window_, &window_state_);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* raw_window, int, int) {
            auto* state = static_cast<detail::WindowState*>(glfwGetWindowUserPointer(raw_window));
            if (state != nullptr && state->resize_requested != nullptr) {
                *state->resize_requested = true;
            }
        });
        glfwSetScrollCallback(window_, [](GLFWwindow* raw_window, double, double yoffset) {
            auto* state = static_cast<detail::WindowState*>(glfwGetWindowUserPointer(raw_window));
            if (state != nullptr) {
                state->scroll += static_cast<float>(yoffset);
            }
        });
        glfwGetCursorPos(window_, &window_state_.last_x, &window_state_.last_y);

        sc_ = swapchain::setup_swapchain(vkctx_, sctx_);
        frames_ = frame::create_frame_system(vkctx_, sc_, detail::frames_in_flight);
        imgui_sys_ = imgui::create(vkctx_, window_, sc_.format, detail::frames_in_flight, static_cast<uint32_t>(sc_.images.size()));

        camera::CameraConfig camera_config{};
        camera_config.orbit_rotate_sens = 0.005f;
        camera_config.orbit_zoom_sens = 0.12f;
        camera_.set_config(camera_config);
        camera_.home();

        sim_.desc = stable_fluids_context_desc_default();
        sim_.desc.nx = 96;
        sim_.desc.ny = 96;
        sim_.desc.nz = 96;
        sim_.desc.dt = 1.0f / 90.0f;
        sim_.desc.viscosity = 0.00012f;
        sim_.desc.diffusion = 0.00003f;
        sim_.desc.diffuse_iterations = 8;
        sim_.desc.pressure_iterations = 24;
        sim_.desc.pressure_tolerance = 1.0e-5f;

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
        density_set_layout_ = raii::DescriptorSetLayout{vkctx_.device, density_layout_ci};

        DescriptorPoolSize density_pool_size{
            .type = DescriptorType::eStorageBuffer,
            .descriptorCount = detail::snapshot_slot_count,
        };
        DescriptorPoolCreateInfo density_pool_ci{
            .flags = DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = detail::snapshot_slot_count,
            .poolSizeCount = 1,
            .pPoolSizes = &density_pool_size,
        };
        density_descriptor_pool_ = raii::DescriptorPool{vkctx_.device, density_pool_ci};

        std::vector<DescriptorSetLayout> density_layouts(detail::snapshot_slot_count, *density_set_layout_);
        DescriptorSetAllocateInfo density_alloc_info{
            .descriptorPool = *density_descriptor_pool_,
            .descriptorSetCount = static_cast<uint32_t>(density_layouts.size()),
            .pSetLayouts = density_layouts.data(),
        };
        density_descriptor_sets_ = vkctx_.device.allocateDescriptorSets(density_alloc_info);

        const std::filesystem::path shader_path = std::filesystem::path(SMOKE_SIM_SHADER_DIR) / "smoke_volume.spv";
        const auto smoke_shader_spv = pipeline::read_file_bytes(shader_path.string());
        smoke_shader_module_ = pipeline::load_shader_module(vkctx_.device, smoke_shader_spv);

        std::array<DescriptorSetLayout, 1> pipeline_set_layouts{*density_set_layout_};
        pipeline::GraphicsPipelineDesc pipeline_desc{};
        pipeline_desc.color_format = sc_.format;
        pipeline_desc.use_depth = false;
        pipeline_desc.use_blend = false;
        pipeline_desc.topology = PrimitiveTopology::eTriangleList;
        pipeline_desc.cull = CullModeFlagBits::eNone;
        pipeline_desc.push_constant_bytes = sizeof(detail::PushConstants);
        pipeline_desc.push_constant_stages = ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment;
        pipeline_desc.set_layouts = pipeline_set_layouts;

        pipeline::VertexInput empty_vertex_input{};
        smoke_pipeline_ = pipeline::create_graphics_pipeline(vkctx_.device, empty_vertex_input, pipeline_desc, smoke_shader_module_, "vs_main", "fs_main");

        recreate_simulation();
        last_frame_time_ = std::chrono::steady_clock::now();
    }

    SmokeApp::~SmokeApp() {
        try {
            vkctx_.device.waitIdle();
        } catch (...) {
        }

        destroy_simulation();
        if (imgui_sys_.initialized) {
            vk::imgui::shutdown(imgui_sys_);
        }
    }

    int SmokeApp::run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            const auto now = std::chrono::steady_clock::now();
            const float dt_seconds = std::chrono::duration<float>(now - last_frame_time_).count();
            last_frame_time_ = now;

            if (sctx_.resize_requested) {
                recreate_swapchain();
            }

            vk::imgui::begin_frame();
            collect_camera_input(dt_seconds);

            const bool placement_mode_changed = draw_ui();
            if (placement_mode_changed) {
                recreate_simulation();
                ui_.step_once = false;
                vk::imgui::end_frame();
                continue;
            }

            if (ui_.reset_fields) {
                recreate_simulation();
                ui_.reset_fields = false;
                ui_.step_once = false;
                vk::imgui::end_frame();
                continue;
            }

            step_simulation();
            ui_.step_once = false;
            update_ready_snapshots();
            render_frame();
            vk::imgui::end_frame();
        }

        return 0;
    }

    void SmokeApp::cuda_ok(const cudaError_t status, const char* what) const {
        if (status == cudaSuccess) {
            return;
        }
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }

    void SmokeApp::stable_ok(const int32_t code, const char* what, const StableFluidsContext* context) const {
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

    StableFluidsBufferView SmokeApp::make_device_buffer_view(void* data, const uint64_t size_bytes) {
        StableFluidsBufferView view{};
        view.data = data;
        view.size_bytes = size_bytes;
        view.format = STABLE_FLUIDS_BUFFER_FORMAT_F32;
        view.memory_type = STABLE_FLUIDS_MEMORY_TYPE_CUDA_DEVICE;
        return view;
    }

    void SmokeApp::destroy_snapshot_slots() {
        for (auto& slot : snapshot_slots_) {
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
        snapshot_slots_.clear();
    }

    void SmokeApp::destroy_simulation() {
        if (sim_.snapshot_stream != nullptr) {
            cudaStreamSynchronize(sim_.snapshot_stream);
        }
        if (sim_.sim_stream != nullptr) {
            cudaStreamSynchronize(sim_.sim_stream);
        }

        destroy_snapshot_slots();

        if (sim_.step_complete_event != nullptr) {
            cudaEventDestroy(sim_.step_complete_event);
            sim_.step_complete_event = nullptr;
        }
        if (sim_.snapshot_stream != nullptr) {
            cudaStreamDestroy(sim_.snapshot_stream);
            sim_.snapshot_stream = nullptr;
        }
        if (sim_.sim_stream != nullptr) {
            cudaStreamDestroy(sim_.sim_stream);
            sim_.sim_stream = nullptr;
        }
        if (sim_.density != nullptr) {
            cudaFree(sim_.density);
            sim_.density = nullptr;
        }
        if (sim_.velocity_x != nullptr) {
            cudaFree(sim_.velocity_x);
            sim_.velocity_x = nullptr;
        }
        if (sim_.velocity_y != nullptr) {
            cudaFree(sim_.velocity_y);
            sim_.velocity_y = nullptr;
        }
        if (sim_.velocity_z != nullptr) {
            cudaFree(sim_.velocity_z);
            sim_.velocity_z = nullptr;
        }
        if (sim_.context != nullptr) {
            stable_fluids_context_destroy(sim_.context);
            sim_.context = nullptr;
        }

        sim_.fields = {};
        sim_.field_bytes = 0;
        sim_.snapshot_generation = 0;
        sim_.submit_serial = 0;
        sim_.steps_since_snapshot = 0;
        sim_.active_snapshot_slot = -1;
        sim_.active_snapshot_generation = 0;
    }

    void SmokeApp::recreate_simulation() {
        using namespace vk;

        destroy_simulation();

        sim_.context = stable_fluids_context_create(&sim_.desc);
        if (sim_.context == nullptr) {
            stable_ok(STABLE_FLUIDS_ERROR_RUNTIME, "stable_fluids_context_create", nullptr);
        }

        sim_.field_bytes = stable_fluids_context_required_scalar_field_bytes(sim_.context);

        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim_.density), sim_.field_bytes), "cudaMalloc density");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim_.velocity_x), sim_.field_bytes), "cudaMalloc velocity_x");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim_.velocity_y), sim_.field_bytes), "cudaMalloc velocity_y");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sim_.velocity_z), sim_.field_bytes), "cudaMalloc velocity_z");

        sim_.fields.density = make_device_buffer_view(sim_.density, sim_.field_bytes);
        sim_.fields.velocity_x = make_device_buffer_view(sim_.velocity_x, sim_.field_bytes);
        sim_.fields.velocity_y = make_device_buffer_view(sim_.velocity_y, sim_.field_bytes);
        sim_.fields.velocity_z = make_device_buffer_view(sim_.velocity_z, sim_.field_bytes);

        cuda_ok(cudaStreamCreateWithFlags(&sim_.sim_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags sim_stream");
        cuda_ok(cudaStreamCreateWithFlags(&sim_.snapshot_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags snapshot_stream");
        cuda_ok(cudaEventCreateWithFlags(&sim_.step_complete_event, cudaEventDisableTiming), "cudaEventCreateWithFlags step_complete_event");

        stable_ok(stable_fluids_fields_clear_async(sim_.context, &sim_.fields, sim_.sim_stream), "stable_fluids_fields_clear_async", sim_.context);
        cuda_ok(cudaStreamSynchronize(sim_.sim_stream), "cudaStreamSynchronize clear");

        snapshot_slots_.clear();
        snapshot_slots_.reserve(detail::snapshot_slot_count);

        for (uint32_t slot_index = 0; slot_index < detail::snapshot_slot_count; ++slot_index) {
            detail::SnapshotSlot slot{};
            slot.descriptor_set = density_descriptor_sets_.at(slot_index);

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
            slot.timeline_semaphore = raii::Semaphore{vkctx_.device, semaphore_ci};

            ExternalMemoryBufferCreateInfo external_buffer_ci{
                .handleTypes = ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
            };
            BufferCreateInfo buffer_ci{
                .pNext = &external_buffer_ci,
                .size = sim_.field_bytes,
                .usage = BufferUsageFlagBits::eStorageBuffer | BufferUsageFlagBits::eTransferDst | BufferUsageFlagBits::eTransferSrc,
                .sharingMode = SharingMode::eExclusive,
            };
            slot.buffer = raii::Buffer{vkctx_.device, buffer_ci};

            const MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
            ExportMemoryAllocateInfo export_memory_ci{
                .handleTypes = ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
            };
            MemoryAllocateInfo alloc_ci{
                .pNext = &export_memory_ci,
                .allocationSize = requirements.size,
                .memoryTypeIndex = memory::find_memory_type(vkctx_.physical_device, requirements.memoryTypeBits, MemoryPropertyFlagBits::eDeviceLocal),
            };
            slot.memory = raii::DeviceMemory{vkctx_.device, alloc_ci};
            slot.buffer.bindMemory(*slot.memory, 0);

#if defined(_WIN32)
            MemoryGetWin32HandleInfoKHR handle_info{
                .memory = *slot.memory,
                .handleType = ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
            };
            HANDLE memory_handle = vkctx_.device.getMemoryWin32HandleKHR(handle_info);
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
            buffer_desc.size = sim_.field_bytes;
            cuda_ok(cudaExternalMemoryGetMappedBuffer(&slot.cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");

            SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                .semaphore = *slot.timeline_semaphore,
                .handleType = ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32,
            };
            HANDLE semaphore_handle = vkctx_.device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
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

            slot.buffer_view = make_device_buffer_view(slot.cuda_ptr, sim_.field_bytes);

            DescriptorBufferInfo density_info{
                .buffer = *slot.buffer,
                .offset = 0,
                .range = sim_.field_bytes,
            };
            WriteDescriptorSet density_write{
                .dstSet = slot.descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = DescriptorType::eStorageBuffer,
                .pBufferInfo = &density_info,
            };
            vkctx_.device.updateDescriptorSets(density_write, {});

            snapshot_slots_.push_back(std::move(slot));
        }

        ui_.source_x = static_cast<float>(sim_.desc.nx) * 0.5f;
        ui_.source_y = static_cast<float>(sim_.desc.ny) * 0.22f;
        ui_.source_z = static_cast<float>(sim_.desc.nz) * 0.5f;

        camera::CameraState camera_state = camera_.state();
        camera_state.mode = camera::Mode::Orbit;
        camera_state.orbit.target = {
            sim_.desc.nx * sim_.desc.cell_size * 0.5f,
            sim_.desc.ny * sim_.desc.cell_size * 0.5f,
            sim_.desc.nz * sim_.desc.cell_size * 0.5f,
            0.0f,
        };
        camera_state.orbit.distance = std::max({sim_.desc.nx, sim_.desc.ny, sim_.desc.nz}) * sim_.desc.cell_size * 2.15f;
        camera_state.orbit.yaw_rad = -0.78539816339f;
        camera_state.orbit.pitch_rad = -0.43633231299f;
        camera_.set_state(camera_state);

        const size_t scalar_count = static_cast<size_t>(sim_.field_bytes / sizeof(float));
        std::vector<float> filled_density(scalar_count, ui_.placement_debug_mode ? 1.0f : 0.0f);
        std::vector<float> zero_field(scalar_count, 0.0f);

        cuda_ok(cudaMemcpyAsync(sim_.density, filled_density.data(), sim_.field_bytes, cudaMemcpyHostToDevice, sim_.sim_stream), "cudaMemcpyAsync initial density");
        cuda_ok(cudaMemcpyAsync(sim_.velocity_x, zero_field.data(), sim_.field_bytes, cudaMemcpyHostToDevice, sim_.sim_stream), "cudaMemcpyAsync initial velocity_x");
        cuda_ok(cudaMemcpyAsync(sim_.velocity_y, zero_field.data(), sim_.field_bytes, cudaMemcpyHostToDevice, sim_.sim_stream), "cudaMemcpyAsync initial velocity_y");
        cuda_ok(cudaMemcpyAsync(sim_.velocity_z, zero_field.data(), sim_.field_bytes, cudaMemcpyHostToDevice, sim_.sim_stream), "cudaMemcpyAsync initial velocity_z");
        cuda_ok(cudaStreamSynchronize(sim_.sim_stream), "cudaStreamSynchronize initial field upload");

        if (!ui_.placement_debug_mode) {
            StableFluidsDensitySplatDesc density_splat{};
            density_splat.center_x = ui_.source_x;
            density_splat.center_y = ui_.source_y;
            density_splat.center_z = ui_.source_z;
            density_splat.radius = ui_.density_radius;
            density_splat.amount = ui_.density_amount;
            stable_ok(stable_fluids_fields_add_density_splat_async(sim_.context, &sim_.fields, &density_splat, sim_.sim_stream), "stable_fluids_fields_add_density_splat_async", sim_.context);

            StableFluidsForceSplatDesc force_splat{};
            force_splat.center_x = ui_.source_x;
            force_splat.center_y = ui_.source_y;
            force_splat.center_z = ui_.source_z;
            force_splat.radius = ui_.force_radius;
            force_splat.force_x = ui_.force_x;
            force_splat.force_y = ui_.force_y;
            force_splat.force_z = ui_.force_z;
            stable_ok(stable_fluids_fields_add_force_splat_async(sim_.context, &sim_.fields, &force_splat, sim_.sim_stream), "stable_fluids_fields_add_force_splat_async", sim_.context);
            stable_ok(stable_fluids_fields_step_async(sim_.context, &sim_.fields, sim_.sim_stream), "stable_fluids_fields_step_async", sim_.context);
            cuda_ok(cudaEventRecord(sim_.step_complete_event, sim_.sim_stream), "cudaEventRecord initial_step_complete_event");
            cuda_ok(cudaStreamWaitEvent(sim_.snapshot_stream, sim_.step_complete_event, 0), "cudaStreamWaitEvent initial_snapshot");
            stable_ok(stable_fluids_fields_snapshot_density_async(sim_.context, &sim_.fields, snapshot_slots_[0].buffer_view, sim_.snapshot_stream), "stable_fluids_fields_snapshot_density_async", sim_.context);
        } else {
            cuda_ok(cudaMemcpyAsync(snapshot_slots_[0].cuda_ptr, filled_density.data(), sim_.field_bytes, cudaMemcpyHostToDevice, sim_.snapshot_stream), "cudaMemcpyAsync placement snapshot");
        }

        const uint64_t initial_generation = sim_.snapshot_generation + 1;
        cudaExternalSemaphoreSignalParams initial_signal_params{};
        initial_signal_params.params.fence.value = initial_generation;
        cuda_ok(
            cudaSignalExternalSemaphoresAsync(
                &snapshot_slots_[0].external_semaphore,
                &initial_signal_params,
                1,
                sim_.snapshot_stream),
            "cudaSignalExternalSemaphoresAsync initial_snapshot");
        cuda_ok(cudaStreamSynchronize(sim_.snapshot_stream), "cudaStreamSynchronize initial_snapshot");
        sim_.snapshot_generation = initial_generation;
        snapshot_slots_[0].ready_generation = initial_generation;
        sim_.active_snapshot_slot = 0;
        sim_.active_snapshot_generation = snapshot_slots_[0].ready_generation;
        sim_.steps_since_snapshot = 0;
    }

    void SmokeApp::recreate_swapchain() {
        vk::swapchain::recreate_swapchain(vkctx_, sctx_, sc_);
        vk::frame::on_swapchain_recreated(vkctx_, sc_, frames_);
        const uint32_t image_count = static_cast<uint32_t>(sc_.images.size());
        if (imgui_sys_.image_count != image_count || imgui_sys_.min_image_count != image_count || imgui_sys_.color_format != sc_.format) {
            const bool docking = imgui_sys_.docking;
            const bool viewports = imgui_sys_.viewports;
            vk::imgui::shutdown(imgui_sys_);
            imgui_sys_ = vk::imgui::create(vkctx_, window_, sc_.format, image_count, image_count, docking, viewports);
        }
        sctx_.resize_requested = false;
    }

    void SmokeApp::collect_camera_input(const float dt_seconds) {
        double mouse_x = 0.0;
        double mouse_y = 0.0;
        glfwGetCursorPos(window_, &mouse_x, &mouse_y);

        float mouse_dx = 0.0f;
        float mouse_dy = 0.0f;
        if (window_state_.first_mouse) {
            window_state_.first_mouse = false;
        } else {
            mouse_dx = static_cast<float>(mouse_x - window_state_.last_x);
            mouse_dy = static_cast<float>(mouse_y - window_state_.last_y);
        }
        window_state_.last_x = mouse_x;
        window_state_.last_y = mouse_y;

        auto& io = ImGui::GetIO();
        vk::camera::CameraInput camera_input{};
        if (!io.WantCaptureMouse) {
            camera_input.lmb = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            camera_input.mmb = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
            camera_input.rmb = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
            camera_input.mouse_dx = mouse_dx;
            camera_input.mouse_dy = mouse_dy;
            camera_input.scroll = window_state_.scroll;
        }
        if (!io.WantCaptureKeyboard) {
            camera_input.forward = glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS;
            camera_input.backward = glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS;
            camera_input.left = glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS;
            camera_input.right = glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS;
            camera_input.up = glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS;
            camera_input.down = glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS;
            camera_input.shift = glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
            camera_input.ctrl = glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            camera_input.alt = glfwGetKey(window_, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
            camera_input.space = glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS;
        }
        window_state_.scroll = 0.0f;

        camera_.update(dt_seconds, sc_.extent.width, sc_.extent.height, camera_input);
    }

    bool SmokeApp::draw_ui() {
        ImGui::Begin("Smoke Control");
        const bool placement_mode_changed = ImGui::Checkbox("Placement Debug Mode", &ui_.placement_debug_mode);
        if (ui_.placement_debug_mode) {
            ImGui::TextUnformatted("Simulation disabled. Rendering a fully filled debug volume.");
        }
        ImGui::Checkbox("Pause Simulation", &ui_.paused);
        if (ImGui::Button("Single Step")) {
            ui_.step_once = true;
        }
        if (ImGui::Button("Reset Fields")) {
            ui_.reset_fields = true;
        }
        ImGui::Separator();
        ImGui::SliderInt("Sim Steps / Frame", &ui_.sim_steps_per_frame, 1, 8);
        ImGui::SliderInt("Snapshot Interval", &ui_.snapshot_interval, 1, 8);
        ImGui::SliderInt("March Steps", &ui_.march_steps, 24, 192);
        ImGui::SliderFloat("Density Scale", &ui_.density_scale, 0.05f, 4.0f, "%.2f");
        ImGui::SliderFloat("Absorption", &ui_.absorption, 0.1f, 6.0f, "%.2f");
        ImGui::ColorEdit3("Smoke Color", &ui_.smoke_r);
        ImGui::Separator();
        ImGui::Checkbox("Emit Density", &ui_.emit_density);
        ImGui::Checkbox("Emit Force", &ui_.emit_force);
        ImGui::SliderFloat("Source X", &ui_.source_x, 0.0f, static_cast<float>(sim_.desc.nx - 1), "%.1f");
        ImGui::SliderFloat("Source Y", &ui_.source_y, 0.0f, static_cast<float>(sim_.desc.ny - 1), "%.1f");
        ImGui::SliderFloat("Source Z", &ui_.source_z, 0.0f, static_cast<float>(sim_.desc.nz - 1), "%.1f");
        ImGui::SliderFloat("Density Radius", &ui_.density_radius, 1.0f, 16.0f, "%.1f");
        ImGui::SliderFloat("Density Amount", &ui_.density_amount, 0.1f, 8.0f, "%.2f");
        ImGui::SliderFloat("Force Radius", &ui_.force_radius, 1.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("Force X", &ui_.force_x, -4.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Force Y", &ui_.force_y, -4.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Force Z", &ui_.force_z, -4.0f, 4.0f, "%.2f");
        ImGui::Separator();
        ImGui::Text("Grid: %d x %d x %d", sim_.desc.nx, sim_.desc.ny, sim_.desc.nz);
        ImGui::Text("Active snapshot: %d", sim_.active_snapshot_slot);
        ImGui::Text("Snapshot generation: %llu", static_cast<unsigned long long>(sim_.active_snapshot_generation));
        ImGui::End();
        return placement_mode_changed;
    }

    void SmokeApp::step_simulation() {
        const bool run_simulation = !ui_.placement_debug_mode && (!ui_.paused || ui_.step_once);
        if (!run_simulation) {
            return;
        }

        for (int sim_step = 0; sim_step < ui_.sim_steps_per_frame; ++sim_step) {
            if (ui_.emit_density) {
                StableFluidsDensitySplatDesc density_splat{};
                density_splat.center_x = ui_.source_x;
                density_splat.center_y = ui_.source_y;
                density_splat.center_z = ui_.source_z;
                density_splat.radius = ui_.density_radius;
                density_splat.amount = ui_.density_amount;
                stable_ok(stable_fluids_fields_add_density_splat_async(sim_.context, &sim_.fields, &density_splat, sim_.sim_stream), "stable_fluids_fields_add_density_splat_async", sim_.context);
            }

            if (ui_.emit_force) {
                StableFluidsForceSplatDesc force_splat{};
                force_splat.center_x = ui_.source_x;
                force_splat.center_y = ui_.source_y;
                force_splat.center_z = ui_.source_z;
                force_splat.radius = ui_.force_radius;
                force_splat.force_x = ui_.force_x;
                force_splat.force_y = ui_.force_y;
                force_splat.force_z = ui_.force_z;
                stable_ok(stable_fluids_fields_add_force_splat_async(sim_.context, &sim_.fields, &force_splat, sim_.sim_stream), "stable_fluids_fields_add_force_splat_async", sim_.context);
            }

            stable_ok(stable_fluids_fields_step_async(sim_.context, &sim_.fields, sim_.sim_stream), "stable_fluids_fields_step_async", sim_.context);
            cuda_ok(cudaEventRecord(sim_.step_complete_event, sim_.sim_stream), "cudaEventRecord step_complete_event");

            if (sim_.steps_since_snapshot < static_cast<uint32_t>(ui_.snapshot_interval)) {
                ++sim_.steps_since_snapshot;
            }
            if (sim_.steps_since_snapshot >= static_cast<uint32_t>(ui_.snapshot_interval)) {
                int snapshot_slot_index = -1;
                for (uint32_t slot_index = 0; slot_index < snapshot_slots_.size(); ++slot_index) {
                    auto& slot = snapshot_slots_[slot_index];
                    if (slot.pending) {
                        continue;
                    }
                    if (static_cast<int>(slot_index) == sim_.active_snapshot_slot) {
                        continue;
                    }
                    if (slot.ready_generation != 0 &&
                        sim_.submit_serial < slot.last_used_submit_serial + frames_.frames_in_flight + 1) {
                        continue;
                    }
                    snapshot_slot_index = static_cast<int>(slot_index);
                    break;
                }

                if (snapshot_slot_index >= 0) {
                    auto& slot = snapshot_slots_[static_cast<size_t>(snapshot_slot_index)];
                    const uint64_t next_generation = sim_.snapshot_generation + 1;
                    cuda_ok(cudaStreamWaitEvent(sim_.snapshot_stream, sim_.step_complete_event, 0), "cudaStreamWaitEvent snapshot_stream");
                    stable_ok(stable_fluids_fields_snapshot_density_async(sim_.context, &sim_.fields, slot.buffer_view, sim_.snapshot_stream), "stable_fluids_fields_snapshot_density_async", sim_.context);
                    slot.pending = true;
                    slot.pending_generation = next_generation;
                    cudaExternalSemaphoreSignalParams signal_params{};
                    signal_params.params.fence.value = next_generation;
                    cuda_ok(
                        cudaSignalExternalSemaphoresAsync(
                            &slot.external_semaphore,
                            &signal_params,
                            1,
                            sim_.snapshot_stream),
                        "cudaSignalExternalSemaphoresAsync snapshot");
                    sim_.snapshot_generation = next_generation;
                    sim_.steps_since_snapshot = 0;
                }
            }
        }
    }

    void SmokeApp::update_ready_snapshots() {
        for (size_t slot_index = 0; slot_index < snapshot_slots_.size(); ++slot_index) {
            auto& slot = snapshot_slots_[slot_index];
            if (!slot.pending) {
                continue;
            }
            if (slot.timeline_semaphore.getCounterValue() >= slot.pending_generation) {
                slot.pending = false;
                slot.ready_generation = slot.pending_generation;
                if (slot.ready_generation >= sim_.active_snapshot_generation) {
                    sim_.active_snapshot_generation = slot.ready_generation;
                    sim_.active_snapshot_slot = static_cast<int>(slot_index);
                }
            }
        }
    }

    void SmokeApp::render_frame() {
        using namespace vk;

        const auto acquire_result = frame::begin_frame(vkctx_, sc_, frames_, frame_index_);
        if (acquire_result.need_recreate) {
            recreate_swapchain();
            return;
        }

        frame::begin_commands(frames_, frame_index_);
        auto& cmd = frame::cmd(frames_, frame_index_);

        const uint32_t image_index = acquire_result.image_index;
        const ImageLayout previous_layout = frames_.swapchain_image_layout[image_index];
        const ImageMemoryBarrier2 to_color_barrier{
            .srcStageMask = previous_layout == ImageLayout::eUndefined ? PipelineStageFlagBits2::eNone : PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = previous_layout == ImageLayout::eUndefined ? AccessFlags2{} : (AccessFlagBits2::eMemoryRead | AccessFlagBits2::eMemoryWrite),
            .dstStageMask = PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = previous_layout,
            .newLayout = ImageLayout::eColorAttachmentOptimal,
            .image = sc_.images[image_index],
            .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2(DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_color_barrier,
        });
        frames_.swapchain_image_layout[image_index] = ImageLayout::eColorAttachmentOptimal;

        ClearValue clear_value{};
        clear_value.color = ClearColorValue{std::array<float, 4>{0.02f, 0.025f, 0.035f, 1.0f}};
        RenderingAttachmentInfo color_attachment{
            .imageView = *sc_.image_views[image_index],
            .imageLayout = ImageLayout::eColorAttachmentOptimal,
            .loadOp = AttachmentLoadOp::eClear,
            .storeOp = AttachmentStoreOp::eStore,
            .clearValue = clear_value,
        };
        RenderingInfo rendering_info{
            .renderArea = Rect2D{Offset2D{0, 0}, sc_.extent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment,
        };

        cmd.beginRendering(rendering_info);
        cmd.setViewport(0, Viewport{
            0.0f,
            0.0f,
            static_cast<float>(sc_.extent.width),
            static_cast<float>(sc_.extent.height),
            0.0f,
            1.0f,
        });
        cmd.setScissor(0, Rect2D{{0, 0}, sc_.extent});

        if (sim_.active_snapshot_slot >= 0) {
            const auto& matrices = camera_.matrices();
            const float half_fov_tan = std::tan(camera_.config().fov_y_rad * 0.5f);
            detail::PushConstants push{};
            push.eye = {matrices.eye.x, matrices.eye.y, matrices.eye.z, 1.0f};
            push.right = {matrices.right.x, matrices.right.y, matrices.right.z, 0.0f};
            push.up = {matrices.up.x, matrices.up.y, matrices.up.z, 0.0f};
            push.forward = {matrices.forward.x, matrices.forward.y, matrices.forward.z, 0.0f};
            push.volume_min = {0.0f, 0.0f, 0.0f, 0.0f};
            push.volume_max = {
                sim_.desc.nx * sim_.desc.cell_size,
                sim_.desc.ny * sim_.desc.cell_size,
                sim_.desc.nz * sim_.desc.cell_size,
                0.0f,
            };
            push.smoke_color = {ui_.smoke_r, ui_.smoke_g, ui_.smoke_b, 1.0f};
            push.params0 = {
                static_cast<float>(sc_.extent.width) / static_cast<float>(std::max(sc_.extent.height, 1u)),
                half_fov_tan,
                ui_.density_scale,
                ui_.absorption,
            };
            push.params1 = {
                static_cast<uint32_t>(sim_.desc.nx),
                static_cast<uint32_t>(sim_.desc.ny),
                static_cast<uint32_t>(sim_.desc.nz),
                static_cast<uint32_t>(ui_.march_steps),
            };

            cmd.bindPipeline(PipelineBindPoint::eGraphics, *smoke_pipeline_.pipeline);
            cmd.bindDescriptorSets(PipelineBindPoint::eGraphics, *smoke_pipeline_.layout, 0, {snapshot_slots_[static_cast<size_t>(sim_.active_snapshot_slot)].descriptor_set}, {});
            const ArrayProxy<const detail::PushConstants> push_block(1, &push);
            cmd.pushConstants(*smoke_pipeline_.layout, ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment, 0, push_block);
            cmd.draw(3, 1, 0, 0);
        }
        cmd.endRendering();

        math::mat4 gizmo_c2w{};
        gizmo_c2w.c0 = {camera_.matrices().right.x, camera_.matrices().right.y, camera_.matrices().right.z, 0.0f};
        gizmo_c2w.c1 = {camera_.matrices().up.x, camera_.matrices().up.y, camera_.matrices().up.z, 0.0f};
        gizmo_c2w.c2 = {camera_.matrices().forward.x, camera_.matrices().forward.y, camera_.matrices().forward.z, 0.0f};
        gizmo_c2w.c3 = {camera_.matrices().eye.x, camera_.matrices().eye.y, camera_.matrices().eye.z, 1.0f};
        imgui::draw_mini_axis_gizmo(gizmo_c2w);
        imgui::render(imgui_sys_, cmd, sc_.extent, *sc_.image_views[image_index], ImageLayout::eColorAttachmentOptimal);

        const ImageMemoryBarrier2 to_present_barrier{
            .srcStageMask = PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = PipelineStageFlagBits2::eBottomOfPipe,
            .oldLayout = ImageLayout::eColorAttachmentOptimal,
            .newLayout = ImageLayout::ePresentSrcKHR,
            .image = sc_.images[image_index],
            .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2(DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_present_barrier,
        });
        frames_.swapchain_image_layout[image_index] = ImageLayout::ePresentSrcKHR;

        std::array<SemaphoreSubmitInfo, 1> snapshot_waits{};
        std::span<const SemaphoreSubmitInfo> extra_waits{};
        const uint64_t submit_serial = sim_.submit_serial + 1;
        if (sim_.active_snapshot_slot >= 0) {
            auto& active_slot = snapshot_slots_[static_cast<size_t>(sim_.active_snapshot_slot)];
            active_slot.last_used_submit_serial = submit_serial;
            snapshot_waits[0] = SemaphoreSubmitInfo{
                .semaphore = *active_slot.timeline_semaphore,
                .value = active_slot.ready_generation,
                .stageMask = PipelineStageFlagBits2::eFragmentShader,
            };
            extra_waits = std::span<const SemaphoreSubmitInfo>(snapshot_waits.data(), snapshot_waits.size());
        }

        const bool need_recreate = frame::end_frame(vkctx_, sc_, frames_, frame_index_, image_index, extra_waits);
        sim_.submit_serial = submit_serial;

        if (need_recreate || sctx_.resize_requested) {
            recreate_swapchain();
        }

        frame_index_ = (frame_index_ + 1) % frames_.frames_in_flight;
    }

} // namespace app
