module;

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include <vulkan/vulkan_raii.hpp>

#include "stable_fluids.h"

export module app;

import std;
import vk.camera;
import vk.context;
import vk.frame;
import vk.imgui;
import vk.math;
import vk.pipeline;
import vk.swapchain;

namespace app::detail {

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

    struct UiState {
        bool placement_debug_mode = false;
        bool paused = false;
        bool step_once = false;
        bool emit_density = true;
        bool emit_force = true;
        bool reset_fields = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval = 3;
        int march_steps = 64;
        float density_radius = 5.0f;
        float density_amount = 1.2f;
        float force_radius = 6.0f;
        float force_x = 0.0f;
        float force_y = 1.8f;
        float force_z = 0.0f;
        float density_scale = 1.0f;
        float absorption = 1.4f;
        float smoke_r = 0.92f;
        float smoke_g = 0.94f;
        float smoke_b = 0.98f;
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

} // namespace app::detail

export namespace app {

    class SmokeApp {
    public:
        SmokeApp();
        ~SmokeApp();

        SmokeApp(const SmokeApp&) = delete;
        SmokeApp& operator=(const SmokeApp&) = delete;
        SmokeApp(SmokeApp&&) noexcept = default;
        SmokeApp& operator=(SmokeApp&&) noexcept = default;

        int run();

    private:
        void cuda_ok(cudaError_t status, const char* what) const;
        void stable_ok(int32_t code, const char* what, const StableFluidsContext* context = nullptr) const;
        static StableFluidsBufferView make_device_buffer_view(void* data, uint64_t size_bytes);
        void destroy_snapshot_slots();
        void destroy_simulation();
        void recreate_simulation();
        void recreate_swapchain();
        void collect_camera_input(float dt_seconds);
        bool draw_ui();
        void step_simulation();
        void update_ready_snapshots();
        void render_frame();

        vk::context::VulkanContext vkctx_{};
        vk::context::SurfaceContext sctx_{};
        GLFWwindow* window_ = nullptr;
        detail::WindowState window_state_{};
        vk::swapchain::Swapchain sc_{};
        vk::frame::FrameSystem frames_{};
        vk::imgui::ImGuiSystem imgui_sys_{};
        vk::camera::Camera camera_{};
        detail::UiState ui_{};
        detail::SimulationState sim_{};
        std::vector<detail::SnapshotSlot> snapshot_slots_{};
        vk::raii::DescriptorSetLayout density_set_layout_{nullptr};
        vk::raii::DescriptorPool density_descriptor_pool_{nullptr};
        std::vector<vk::raii::DescriptorSet> density_descriptor_sets_{};
        vk::raii::ShaderModule smoke_shader_module_{nullptr};
        vk::pipeline::GraphicsPipeline smoke_pipeline_{};
        uint32_t frame_index_ = 0;
        std::chrono::steady_clock::time_point last_frame_time_ = std::chrono::steady_clock::now();
    };

} // namespace app
