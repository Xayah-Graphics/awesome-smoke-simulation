module;

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <nvtx3/nvtx3.hpp>
#include <vulkan/vulkan_raii.hpp>

module app;

import std;
import vk.camera;
import vk.context;
import vk.frame;
import vk.imgui;
import vk.math;
import vk.pipeline;
import vk.swapchain;

namespace app {

    FieldRendererApp::FieldRendererApp() {
        using namespace vk;

        auto [vkctx, sctx] = context::setup_vk_context_glfw("awesome-smoke-simulation", "smoke-visualizer");
        vkctx_ = std::move(vkctx);
        sctx_ = std::move(sctx);
        window_ = sctx_.window.get();

        window_state_.resize_requested = &sctx_.resize_requested;
        glfwSetWindowUserPointer(window_, &window_state_);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* raw_window, int, int) {
            auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(raw_window));
            if (state != nullptr && state->resize_requested != nullptr) {
                *state->resize_requested = true;
            }
        });
        glfwSetScrollCallback(window_, [](GLFWwindow* raw_window, double, double yoffset) {
            auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(raw_window));
            if (state != nullptr) {
                state->scroll += static_cast<float>(yoffset);
            }
        });
        glfwGetCursorPos(window_, &window_state_.last_x, &window_state_.last_y);

        sc_ = swapchain::setup_swapchain(vkctx_, sctx_);
        frames_ = frame::create_frame_system(vkctx_, sc_, frames_in_flight_value_);
        imgui_sys_ = imgui::create(vkctx_, window_, sc_.format, frames_in_flight_value_, static_cast<uint32_t>(sc_.images.size()));

        camera::CameraConfig camera_config{};
        camera_config.orbit_rotate_sens = 0.005f;
        camera_config.orbit_zoom_sens = 0.12f;
        camera_.set_config(camera_config);
        camera_.home();

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
            .descriptorCount = 128,
        };
        DescriptorPoolCreateInfo density_pool_ci{
            .flags = DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 128,
            .poolSizeCount = 1,
            .pPoolSizes = &density_pool_size,
        };
        density_descriptor_pool_ = raii::DescriptorPool{vkctx_.device, density_pool_ci};

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
        pipeline_desc.push_constant_bytes = sizeof(PushConstants);
        pipeline_desc.push_constant_stages = ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment;
        pipeline_desc.set_layouts = pipeline_set_layouts;

        pipeline::VertexInput empty_vertex_input{};
        smoke_pipeline_ = pipeline::create_graphics_pipeline(vkctx_.device, empty_vertex_input, pipeline_desc, smoke_shader_module_, "vs_main", "fs_main");
        last_frame_time_ = std::chrono::steady_clock::now();
    }

    FieldRendererApp::~FieldRendererApp() {
        try {
            vkctx_.device.waitIdle();
        } catch (...) {
        }

        if (imgui_sys_.initialized) {
            vk::imgui::shutdown(imgui_sys_);
        }
    }

    bool FieldRendererApp::should_close() const {
        return glfwWindowShouldClose(window_) != 0;
    }

    FrameInfo FieldRendererApp::begin_frame() {
        nvtx3::scoped_range range{"renderer.begin_frame"};
        glfwPollEvents();

        const auto now = std::chrono::steady_clock::now();
        const float dt_seconds = std::chrono::duration<float>(now - last_frame_time_).count();
        last_frame_time_ = now;
        if (dt_seconds > 0.0f) {
            const float instantaneous_fps = 1.0f / dt_seconds;
            render_fps_ = render_fps_ > 0.0f ? std::lerp(render_fps_, instantaneous_fps, 0.1f) : instantaneous_fps;
        }

        if (sctx_.resize_requested) {
            recreate_swapchain();
        }

        vk::imgui::begin_frame();
        collect_camera_input(dt_seconds);
        return FrameInfo{.dt_seconds = dt_seconds, .render_fps = render_fps_};
    }

    void FieldRendererApp::draw_renderer_ui(const std::optional<VolumeFieldView>& field) {
        ImGui::Begin("Renderer");
        ImGui::SliderInt("March Steps", &render_.march_steps, 24, 192);
        ImGui::SliderInt("Shadow Steps", &render_.shadow_steps, 4, 48);
        ImGui::SliderFloat("Density Scale", &render_.density_scale, 0.05f, 4.0f, "%.2f");
        ImGui::SliderFloat("Absorption", &render_.absorption, 0.1f, 6.0f, "%.2f");
        ImGui::ColorEdit3("Smoke Color", &render_.smoke_r);
        ImGui::SliderFloat("Light X", &render_.light_x, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Light Y", &render_.light_y, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Light Z", &render_.light_z, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Light Intensity", &render_.light_intensity, 0.0f, 6.0f, "%.2f");
        ImGui::SliderFloat("Ambient Light", &render_.ambient_light, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Shadow Strength", &render_.shadow_strength, 0.1f, 4.0f, "%.2f");
        ImGui::SliderFloat("Phase G", &render_.phase_g, -0.3f, 0.8f, "%.2f");
        if (field) {
            ImGui::Separator();
            ImGui::Text("Grid: %u x %u x %u", field->nx, field->ny, field->nz);
            ImGui::Text("Cell Size: %.3f", field->cell_size);
            ImGui::Text("Snapshot generation: %llu", static_cast<unsigned long long>(field->ready_generation));
        }
        ImGui::End();

        if (const ImGuiViewport* viewport = ImGui::GetMainViewport()) {
            ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + 12.0f, viewport->Pos.y + 12.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.35f);
            ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                             ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;
            ImGui::Begin("Render Stats Overlay", nullptr, overlay_flags);
            ImGui::Text("Render: %.1f FPS", render_fps_);
            ImGui::End();
        }
    }

    bool FieldRendererApp::render_frame(const std::optional<VolumeFieldView>& field) {
        nvtx3::scoped_range range{"renderer.render_frame"};
        using namespace vk;

        const auto acquire_result = frame::begin_frame(vkctx_, sc_, frames_, frame_index_);
        if (acquire_result.need_recreate) {
            recreate_swapchain();
            vk::imgui::end_frame();
            return false;
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

        if (field) {
            const auto& matrices = camera_.matrices();
            const float half_fov_tan = std::tan(camera_.config().fov_y_rad * 0.5f);
            const vk::math::vec3 light_dir = vk::math::normalize(vk::math::vec3{render_.light_x, render_.light_y, render_.light_z, 0.0f});
            PushConstants push{};
            push.eye = {matrices.eye.x, matrices.eye.y, matrices.eye.z, 1.0f};
            push.right = {matrices.right.x, matrices.right.y, matrices.right.z, 0.0f};
            push.up = {matrices.up.x, matrices.up.y, matrices.up.z, 0.0f};
            push.forward = {matrices.forward.x, matrices.forward.y, matrices.forward.z, 0.0f};
            push.volume_min = {0.0f, 0.0f, 0.0f, 0.0f};
            push.volume_max = {
                field->nx * field->cell_size,
                field->ny * field->cell_size,
                field->nz * field->cell_size,
                0.0f,
            };
            push.smoke_color = {render_.smoke_r, render_.smoke_g, render_.smoke_b, 1.0f};
            push.light_dir = {light_dir.x, light_dir.y, light_dir.z, render_.light_intensity};
            push.lighting_params = {render_.ambient_light, render_.shadow_strength, render_.phase_g, 0.0f};
            push.params0 = {
                static_cast<float>(sc_.extent.width) / static_cast<float>((std::max)(sc_.extent.height, 1u)),
                half_fov_tan,
                render_.density_scale,
                render_.absorption,
            };
            push.params1 = {
                field->nx,
                field->ny,
                field->nz,
                static_cast<uint32_t>(render_.march_steps),
            };
            push.params2 = {
                static_cast<uint32_t>(render_.shadow_steps),
                0u,
                0u,
                0u,
            };

            cmd.bindPipeline(PipelineBindPoint::eGraphics, *smoke_pipeline_.pipeline);
            cmd.bindDescriptorSets(PipelineBindPoint::eGraphics, *smoke_pipeline_.layout, 0, {field->descriptor_set}, {});
            const ArrayProxy<const PushConstants> push_block(1, &push);
            cmd.pushConstants(*smoke_pipeline_.layout, ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment, 0, push_block);
            cmd.draw(3, 1, 0, 0);
        }
        cmd.endRendering();

        vk::math::mat4 gizmo_c2w{};
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

        std::array<SemaphoreSubmitInfo, 1> volume_waits{};
        std::span<const SemaphoreSubmitInfo> extra_waits{};
        if (field) {
            volume_waits[0] = SemaphoreSubmitInfo{
                .semaphore = field->timeline_semaphore,
                .value = field->ready_generation,
                .stageMask = PipelineStageFlagBits2::eFragmentShader,
            };
            extra_waits = std::span<const SemaphoreSubmitInfo>(volume_waits.data(), volume_waits.size());
        }

        const bool need_recreate = frame::end_frame(vkctx_, sc_, frames_, frame_index_, image_index, extra_waits);
        if (need_recreate || sctx_.resize_requested) {
            recreate_swapchain();
        }

        frame_index_ = (frame_index_ + 1) % frames_.frames_in_flight;
        vk::imgui::end_frame();
        return true;
    }

    void FieldRendererApp::frame_volume(const VolumeFieldView& field) {
        vk::camera::CameraState camera_state = camera_.state();
        camera_state.mode = vk::camera::Mode::Orbit;
        camera_state.orbit.target = {
            field.nx * field.cell_size * 0.5f,
            field.ny * field.cell_size * 0.5f,
            field.nz * field.cell_size * 0.5f,
            0.0f,
        };
        camera_state.orbit.distance = (std::max)({field.nx, field.ny, field.nz}) * field.cell_size * 2.15f;
        camera_state.orbit.yaw_rad = -0.78539816339f;
        camera_state.orbit.pitch_rad = -0.43633231299f;
        camera_.set_state(camera_state);
    }

    RenderSettings& FieldRendererApp::render_settings() {
        return render_;
    }

    const RenderSettings& FieldRendererApp::render_settings() const {
        return render_;
    }

    const vk::context::VulkanContext& FieldRendererApp::vk_context() const {
        return vkctx_;
    }

    uint32_t FieldRendererApp::frames_in_flight() const {
        return frames_.frames_in_flight;
    }

    std::vector<vk::raii::DescriptorSet> FieldRendererApp::allocate_density_descriptor_sets(const uint32_t count) {
        std::vector<vk::DescriptorSetLayout> density_layouts(count, *density_set_layout_);
        vk::DescriptorSetAllocateInfo density_alloc_info{
            .descriptorPool = *density_descriptor_pool_,
            .descriptorSetCount = count,
            .pSetLayouts = density_layouts.data(),
        };
        return vkctx_.device.allocateDescriptorSets(density_alloc_info);
    }

    void FieldRendererApp::recreate_swapchain() {
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

    void FieldRendererApp::collect_camera_input(const float dt_seconds) {
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

} // namespace app
