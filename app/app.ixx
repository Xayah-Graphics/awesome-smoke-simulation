module;

#include <GLFW/glfw3.h>

#include <vulkan/vulkan_raii.hpp>

export module app;

import std;
import vk.camera;
import vk.context;
import vk.frame;
import vk.imgui;
import vk.math;
import vk.pipeline;
import vk.swapchain;

export namespace app {

    enum class FieldSemantic : uint32_t {
        Density           = 0,
        VelocityMagnitude = 1,
        DyeColor          = 2,
        GenericScalar     = 3,
    };

    enum class RenderMode : uint32_t {
        Smoke  = 0,
        Scalar = 1,
    };

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
        vk::math::vec4 color_a{};
        vk::math::vec4 color_b{};
        vk::math::vec4 params0{};
        uvec4 params1{};
        uvec4 params2{};
        vk::math::vec4 params3{};
    };

    struct RenderSettings {
        RenderMode mode       = RenderMode::Smoke;
        int march_steps       = 96;
        float density_scale   = 0.95f;
        float absorption      = 1.20f;
        bool show_bounds      = true;
        bool show_collider    = true;
        bool show_velocity_slice = false;
        uint32_t velocity_slice_axis = 2;
        float velocity_slice_position = 0.5f;
        int velocity_streamer_grid = 18;
        int velocity_streamer_steps = 28;
        float velocity_streamer_step = 0.85f;
        float velocity_streamer_min_speed = 0.04f;
        float velocity_streamer_thickness = 1.6f;
        float scalar_min      = 0.0f;
        float scalar_max      = 1.0f;
        float scalar_opacity  = 2.0f;
        float scalar_low_r    = 0.08f;
        float scalar_low_g    = 0.18f;
        float scalar_low_b    = 0.46f;
        float scalar_high_r   = 0.98f;
        float scalar_high_g   = 0.82f;
        float scalar_high_b   = 0.24f;
    };

    struct FrameInfo {
        float dt_seconds = 0.0f;
        float render_fps = 0.0f;
    };

    struct ScalarFieldView {
        vk::DescriptorSet descriptor_set{nullptr};
        vk::Semaphore timeline_semaphore{nullptr};
        uint64_t ready_generation = 0;
        uint32_t nx               = 0;
        uint32_t ny               = 0;
        uint32_t nz               = 0;
        float cell_size           = 1.0f;
        FieldSemantic semantic    = FieldSemantic::GenericScalar;
        std::string_view label{};
        bool collider_enabled     = false;
        uint32_t collider_type    = 0;
        float collider_center_x   = 0.0f;
        float collider_center_y   = 0.0f;
        float collider_center_z   = 0.0f;
        float collider_radius     = 0.0f;
        float collider_half_x     = 0.0f;
        float collider_half_y     = 0.0f;
        float collider_half_z     = 0.0f;
        const float* velocity_xyz = nullptr;
    };

    class FieldRendererApp {
    public:
        FieldRendererApp();
        ~FieldRendererApp();

        FieldRendererApp(const FieldRendererApp&)                = delete;
        FieldRendererApp& operator=(const FieldRendererApp&)     = delete;
        FieldRendererApp(FieldRendererApp&&) noexcept            = delete;
        FieldRendererApp& operator=(FieldRendererApp&&) noexcept = delete;

        [[nodiscard]] bool should_close() const;
        FrameInfo begin_frame();
        void draw_renderer_ui(const std::optional<ScalarFieldView>& field);
        bool render_frame(const std::optional<ScalarFieldView>& field);
        void frame_volume(const ScalarFieldView& field);

        [[nodiscard]] RenderSettings& render_settings();
        [[nodiscard]] const RenderSettings& render_settings() const;
        [[nodiscard]] const vk::context::VulkanContext& vk_context() const;
        [[nodiscard]] uint32_t frames_in_flight() const;
        [[nodiscard]] std::vector<vk::raii::DescriptorSet> allocate_field_descriptor_sets(uint32_t count);

    private:
        void recreate_swapchain();
        void collect_camera_input(float dt_seconds);

        static constexpr uint32_t frames_in_flight_value_ = 2;

        struct WindowState {
            bool* resize_requested = nullptr;
            float scroll           = 0.0f;
            bool first_mouse       = true;
            double last_x          = 0.0;
            double last_y          = 0.0;
        };

        vk::context::VulkanContext vkctx_{};
        vk::context::SurfaceContext sctx_{};
        GLFWwindow* window_ = nullptr;
        WindowState window_state_{};
        vk::swapchain::Swapchain sc_{};
        vk::frame::FrameSystem frames_{};
        vk::imgui::ImGuiSystem imgui_sys_{};
        vk::camera::Camera camera_{};
        vk::raii::DescriptorSetLayout field_set_layout_{nullptr};
        vk::raii::DescriptorPool field_descriptor_pool_{nullptr};
        vk::raii::ShaderModule smoke_shader_module_{nullptr};
        vk::pipeline::GraphicsPipeline smoke_pipeline_{};
        RenderSettings render_{};
        float render_fps_                                      = 0.0f;
        uint32_t frame_index_                                  = 0;
        std::chrono::steady_clock::time_point last_frame_time_ = std::chrono::steady_clock::now();
    };

} // namespace app
