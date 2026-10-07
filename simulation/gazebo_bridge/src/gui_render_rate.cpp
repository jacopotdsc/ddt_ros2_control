// gzclient system plugin: caps the GUI render rate so the window does not steal
// CPU from gzserver and MPX. Physics, ros2_control and MPX rates are unchanged.
// The rate (Hz) is read from the TITA_GUI_FPS environment variable (default 20).

#include <cstdlib>
#include <iostream>
#include <string>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/gui/GuiEvents.hh>
#include <gazebo/gui/GuiIface.hh>
#include <gazebo/rendering/UserCamera.hh>

namespace gazebo
{
class TitaGuiRenderRate : public SystemPlugin
{
public:
  void Load(int /*_argc*/, char ** /*_argv*/) override
  {
    if (const char * env = std::getenv("TITA_GUI_FPS")) {
      rate_ = std::atof(env);
    }
    if (rate_ <= 0.0) rate_ = 20.0;
  }

  void Init() override
  {
    // The user camera exists only after the scene is created: apply on the
    // first render event that finds it. The connection is kept: resetting it
    // inside its own callback corrupts the event's connection list.
    connection_ = event::Events::ConnectPreRender([this]() { Apply(); });
  }

private:
  void Apply()
  {
    if (applied_) return;
    auto camera = gui::get_active_camera();
    if (!camera) return;
    applied_ = true;
    camera->SetRenderRate(rate_);
    gui::Events::setRenderRate(rate_);
    std::cout << "[TitaGuiRenderRate] GUI render rate limited to " << rate_ << " FPS" << std::endl;
  }

  double rate_{20.0};
  bool applied_{false};
  event::ConnectionPtr connection_;
};

GZ_REGISTER_SYSTEM_PLUGIN(TitaGuiRenderRate)
}  // namespace gazebo
