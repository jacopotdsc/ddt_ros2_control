from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
from rclpy import init, spin
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class FsmAudioNode(Node):
    def __init__(self) -> None:
        super().__init__("fsm_audio_node")
        self.declare_parameter("state_topic", "fsm_state")
        self.declare_parameter(
            "audio_service", "system/audio_control/play_audio_with_file_path"
        )
        self.declare_parameter("announce_initial_state", True)

        self._audio_root = (
            Path(get_package_share_directory("fsm_audio")) / "config" / "fsm_mp3"
        )
        self._announce_initial_state = bool(
            self.get_parameter("announce_initial_state").value
        )

        try:
            from tita_system_interfaces.srv import PlayAudioWithFilePath
        except ImportError as error:
            raise RuntimeError(
                "fsm_audio requires the robot's tita_system_interfaces package"
            ) from error

        service_name = self.get_parameter("audio_service").value
        self._audio_client = self.create_client(PlayAudioWithFilePath, service_name)
        self._last_state: str | None = None
        self._pending_audio_file: str | None = None
        self._playback_in_flight = False
        self._service_warning_logged = False

        state_topic = self.get_parameter("state_topic").value
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._state_subscription = self.create_subscription(
            String, state_topic, self._on_state, qos
        )
        self._retry_timer = self.create_timer(0.25, self._play_pending)
        self.get_logger().info(
            f"Listening for FSM transitions on {self.resolve_topic_name(state_topic)}"
        )

    def _on_state(self, message: String) -> None:
        state = message.data.strip()
        if not state or state == self._last_state:
            return

        is_initial_state = self._last_state is None
        self._last_state = state
        if Path(state).name != state or state in {".", ".."} or "\0" in state:
            self._pending_audio_file = None
            self.get_logger().warning(f"Invalid FSM state name for audio lookup: {state!r}")
            return
        if is_initial_state and not self._announce_initial_state:
            self.get_logger().debug(f"Received initial FSM state {state!r}; not announcing")
            return

        audio_path = self._audio_root / f"{state}.mp3"
        if not audio_path.is_file():
            self._pending_audio_file = None
            self.get_logger().info(
                f"No voice clip found for FSM state {state!r}: {audio_path}"
            )
            return

        self._pending_audio_file = str(audio_path.resolve())
        self._play_pending()

    def _play_pending(self) -> None:
        if self._pending_audio_file is None or self._playback_in_flight:
            return
        if not self._audio_client.service_is_ready():
            if not self._service_warning_logged:
                service_name = self.get_parameter("audio_service").value
                self.get_logger().warning(
                    f"Audio service {service_name} is not "
                    "available yet; the latest state announcement is queued"
                )
                self._service_warning_logged = True
            return

        from std_msgs.msg import String as StringMessage
        from tita_system_interfaces.srv import PlayAudioWithFilePath

        request = PlayAudioWithFilePath.Request()
        request.audio_file_path = StringMessage(data=self._pending_audio_file)
        audio_file = self._pending_audio_file
        self._pending_audio_file = None
        self._playback_in_flight = True

        future = self._audio_client.call_async(request)
        future.add_done_callback(lambda result: self._on_playback_done(result, audio_file))

    def _on_playback_done(self, future: Any, audio_file: str) -> None:
        self._playback_in_flight = False
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f"Audio service call failed for {audio_file!r}: {error}")
        else:
            if response.success:
                self.get_logger().info(f"Played FSM announcement {audio_file!r}")
            else:
                self.get_logger().error(
                    f"Audio service rejected FSM announcement {audio_file!r}"
                )
        self._play_pending()


def main(args: list[str] | None = None) -> None:
    init(args=args)
    node = FsmAudioNode()
    try:
        spin(node)
    finally:
        node.destroy_node()
        from rclpy import shutdown

        shutdown()
