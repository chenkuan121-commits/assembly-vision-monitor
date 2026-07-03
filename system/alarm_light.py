import os
import queue
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


DEFAULT_PORT = os.environ.get("VISION_CODEX_ALARM_PORT", "").strip() or None
DEFAULT_BAUDRATE = int(os.environ.get("VISION_CODEX_ALARM_BAUDRATE", "9600"))

LIGHT_MODES = {
    "off": 0x01,
    "green": 0x02,
    "yellow": 0x03,
    "red": 0x04,
}

BUZZER_MODES = {
    "off": 0x01,
    "on": 0x02,
}

FLASH_MODES = {
    "off": 0x01,
    "fast": 0x02,
    "medium": 0x03,
    "slow": 0x04,
}


def command_frame(light, buzzer="off", flash="off"):
    return bytes([
        0xFF,
        LIGHT_MODES[light],
        BUZZER_MODES[buzzer],
        FLASH_MODES[flash],
        0xAA,
    ])


def auto_detect_port():
    if list_ports is None:
        return None
    ports = list(list_ports.comports())
    if not ports:
        return None

    keywords = ("usb", "serial", "ch340", "cp210", "wch", "ftdi")
    candidates = []
    for port in ports:
        text = " ".join(
            str(value).lower()
            for value in (port.device, port.description, port.manufacturer, port.hwid)
            if value
        )
        if any(keyword in text for keyword in keywords):
            candidates.append(port)

    if len(candidates) == 1:
        return candidates[0].device
    if len(ports) == 1:
        return ports[0].device

    listed = ", ".join(f"{p.device}({p.description})" for p in (candidates or ports))
    print(
        "[AlarmLight] disabled: multiple serial ports found. "
        f"Set VISION_CODEX_ALARM_PORT to one of: {listed}"
    )
    return None


def list_serial_port_options():
    if list_ports is None:
        return []
    return [
        {
            "device": port.device,
            "description": port.description or "",
            "manufacturer": port.manufacturer or "",
            "hwid": port.hwid or "",
        }
        for port in list_ports.comports()
    ]


class AlarmLightController:
    """Non-blocking serial controller; green is steady whenever no alarm is active."""

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE, buzzer_enabled=True):
        self.port = port or auto_detect_port()
        self.baudrate = baudrate
        self.buzzer_enabled = bool(buzzer_enabled)
        self.enabled = serial is not None and self.port is not None
        self._ser = None
        self._serial_lock = threading.RLock()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop_event = threading.Event()
        self._forbidden_lock = threading.Lock()
        self._forbidden_requested = False
        self._steady_forbidden = None
        self._last_flash_by_key = {}
        self._thread.start()
        self._queue.put(("sync",))

    def set_buzzer_enabled(self, enabled):
        self.buzzer_enabled = bool(enabled)
        if self._is_forbidden_requested():
            self._steady_forbidden = None
            self._queue.put(("forbidden", True))

    def set_port(self, port):
        port = (port or "").strip() or auto_detect_port()
        with self._serial_lock:
            if self._ser is not None:
                try:
                    self._send("off", "off", "off")
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
            self.port = port
            self.enabled = serial is not None and self.port is not None
            self._steady_forbidden = None
        self._queue.put(("sync",))
        return self.enabled, self.port

    def set_forbidden_alarm(self, active):
        """Forbidden item: steady red light + buzzer until the item disappears."""
        active = bool(active)
        with self._forbidden_lock:
            self._forbidden_requested = active
        self._queue.put(("forbidden", active))

    def flash_red(self, key="event", times=3, interval=0.3, cooldown=2.0, buzzer=True):
        """Jump/AOI alarms: red light flashes a few times, optionally with buzzer."""
        now = time.time()
        last = self._last_flash_by_key.get(key, 0.0)
        if now - last < cooldown:
            return
        self._last_flash_by_key[key] = now
        self._queue.put(("flash_red", key, int(times), float(interval), bool(buzzer)))

    def stop(self):
        self._stop_event.set()
        self._queue.put(("stop",))
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _open(self):
        with self._serial_lock:
            if not self.enabled or self._ser is not None:
                return self._ser
            try:
                self._ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.2,
                    write_timeout=0.2,
                )
                time.sleep(0.2)
            except Exception as e:
                print(f"[AlarmLight] disabled: {e}")
                self.enabled = False
                self._ser = None
            return self._ser

    def _send(self, light, buzzer="off", flash="off"):
        with self._serial_lock:
            ser = self._open()
            if ser is None:
                return
            try:
                ser.reset_input_buffer()
                ser.write(command_frame(light, buzzer, flash))
                ser.flush()
            except Exception as e:
                print(f"[AlarmLight] write failed, disabled: {e}")
                self.enabled = False
                try:
                    ser.close()
                except Exception:
                    pass
                self._ser = None

    def _is_forbidden_requested(self):
        with self._forbidden_lock:
            return self._forbidden_requested

    def _sleep_interruptible(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self._stop_event.is_set() or self._is_forbidden_requested():
                return False
            time.sleep(min(0.05, end_time - time.time()))
        return True

    def _apply_steady_state(self):
        """Restore the persistent state after startup, port changes, or flashing."""
        active = self._is_forbidden_requested()
        self._steady_forbidden = active
        if active:
            buzzer = "on" if self.buzzer_enabled else "off"
            self._send("red", buzzer, "off")
        else:
            self._send("green", "off", "off")

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            action = item[0]
            if action == "stop":
                break

            if action in ("sync", "forbidden"):
                active = self._is_forbidden_requested()
                if action == "sync" or active != self._steady_forbidden:
                    self._apply_steady_state()
                continue

            if action == "flash_red":
                _, _, times, interval, buzzer = item
                if self._is_forbidden_requested():
                    continue
                buzzer_mode = "on" if buzzer and self.buzzer_enabled else "off"
                for _ in range(max(1, times)):
                    if self._stop_event.is_set() or self._is_forbidden_requested():
                        break
                    self._send("red", buzzer_mode, "off")
                    if not self._sleep_interruptible(interval):
                        break
                    self._send("off", "off", "off")
                    if not self._sleep_interruptible(interval):
                        break

                if not self._stop_event.is_set():
                    self._apply_steady_state()

        self._send("off", "off", "off")
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
