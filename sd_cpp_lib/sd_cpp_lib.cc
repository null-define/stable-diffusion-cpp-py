#include "magic_enum/magic_enum.hpp"
#include "pybind11/pybind11.h"
#include "stable-diffusion.h"
#include "util.h"
#include <memory>

static sd_log_level_t g_cfg_log_level = SD_LOG_INFO;

/* Enables Printing the log level tag in color using ANSI escape codes */
void sd_log_cb(enum sd_log_level_t level, char const* log, void* data) {
  int tag_color;
  char const* level_str;
  FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

  if (!log || level < g_cfg_log_level) {
    return;
  }

  switch (level) {
    case SD_LOG_DEBUG:
      tag_color = 37;
      level_str = "DEBUG";
      break;
    case SD_LOG_INFO:
      tag_color = 34;
      level_str = "INFO";
      break;
    case SD_LOG_WARN:
      tag_color = 35;
      level_str = "WARN";
      break;
    case SD_LOG_ERROR:
      tag_color = 31;
      level_str = "ERROR";
      break;
    default: /* Potential future-proofing */
      tag_color = 33;
      level_str = "?????";
      break;
  }

  fprintf(out_stream, "[%-5s] ", level_str);

  fputs(log, out_stream);
  fflush(out_stream);
}

template <typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

namespace py = pybind11;

template <typename E, typename... Extra>
py::enum_<E> export_enum(py::handle const& scope, Extra&&... extra) {
  py::enum_<E> enum_type(scope, magic_enum::enum_type_name<E>().data(),
                         std::forward<Extra>(extra)...);
  for (auto const& [value, name] : magic_enum::enum_entries<E>()) {
    enum_type.value(name.data(), value);
  }
  return enum_type;
}

PYBIND11_MODULE(sd_cpp_lib, m) {
  m.doc() = "stable-diffusion.cpp python bind";  // optional module docstring
  py::class_<sd_image_t>(m, "sd_image_t", py::buffer_protocol())
      .def_buffer([](sd_image_t& m) -> py::buffer_info {
        return py::buffer_info(
            m.data,                                   /* Pointer to buffer */
            sizeof(uint8_t),                          /* Size of one scalar */
            py::format_descriptor<uint8_t>::format(), /* Python struct-style
                                                         format descriptor */
            3,                                        /* Number of dimensions */
            {m.width, m.height, m.channel},           /* Buffer dimensions */
            {sizeof(uint8_t) * m.channel *
                 m.height, /* Strides (in bytes) for each index */
             sizeof(uint8_t) * m.channel, sizeof(uint8_t)});
      });
  export_enum<sd_type_t>(m);
  export_enum<rng_type_t>(m);
  export_enum<schedule_t>(m);
  export_enum<sample_method_t>(m);
  export_enum<sd_log_level_t>(m);
  m.def("set_log_level",
        [](sd_log_level_t log_level) { g_cfg_log_level = log_level; });
  m.def(
      "new_sd_ctx",
      [](std::string const& model_path, std::string const& vae_path,
         std::string const& taesd_path, std::string const& control_net_path,
         std::string const& lora_model_dir, std::string const& embed_dir,
         std::string const& stacked_id_embed_dir, bool vae_decode_only,
         bool vae_tiling, bool free_params_immediately, int n_threads,
         sd_type_t wtype, rng_type_t rng_type, schedule_t s,
         bool keep_clip_on_cpu, bool keep_control_net_cpu,
         bool keep_vae_on_cpu) {
        sd_set_log_callback(sd_log_cb, nullptr);
        auto ptr = new_sd_ctx(
            model_path.c_str(), vae_path.c_str(), taesd_path.c_str(),
            control_net_path.c_str(), lora_model_dir.c_str(), embed_dir.c_str(),
            stacked_id_embed_dir.c_str(), vae_decode_only, vae_tiling,
            free_params_immediately, n_threads, wtype, rng_type, s,
            keep_clip_on_cpu, keep_control_net_cpu, keep_vae_on_cpu);
        return static_cast<py::capsule>(ptr);
      },
      "create stable diffusion context");
  m.def(
      "txt2img",
      [](py::capsule sd_ctx, std::string const& prompt,
         std::string const& negative_prompt, int clip_skip, float cfg_scale,
         int width, int height, sample_method_t sample_method, int sample_steps,
         int64_t seed, int batch_count, sd_image_t const* control_cond,
         float control_strength, float style_strength, bool normalize_input,
         std::string const& input_id_images_path) {
        return deleted_unique_ptr<sd_image_t>(
            txt2img(static_cast<sd_ctx_t*>(sd_ctx), prompt.c_str(),
                    negative_prompt.c_str(), clip_skip, cfg_scale, width,
                    height, sample_method, sample_steps, seed, batch_count,
                    control_cond, control_strength, style_strength,
                    normalize_input, input_id_images_path.c_str()),
            [](sd_image_t* p) { free(p->data); });
      },
      "text 2 image");
  m.def(
      "free_sd_ctx",
      [](py::capsule sd_ctx) { free_sd_ctx(static_cast<sd_ctx_t*>(sd_ctx)); },
      "release stable diffusion context");
  m.def(
      "new_upscaler_ctx",
      [](std::string const& esrgan_path, int n_threads, sd_type_t wtype) {
        return static_cast<py::capsule>(
            new_upscaler_ctx(esrgan_path.c_str(), n_threads, wtype));
      },
      "create upscaler context");
  m.def(
      "upscale",
      [](py::capsule upscaler_ctx, sd_image_t input_image,
         uint32_t upscale_factor) {
        return deleted_unique_ptr<sd_image_t>(
            new sd_image_t(upscale(static_cast<upscaler_ctx_t*>(upscaler_ctx),
                                   input_image, upscale_factor)),
            [](sd_image_t* p) { free(p->data); });
      },
      "upscale using upscaler context");
  m.def(
      "free_upscaler_ctx",
      [](py::capsule upscaler_ctx) {
        free_upscaler_ctx(static_cast<upscaler_ctx_t*>(upscaler_ctx));
      },
      "release upscaler context");
}