import socket
import struct

TCP_IP = '10.0.0.228'  # IP address of the ESP32
TCP_PORT = 10000  # Same port as the ESP32

OPEN_KEYWORDS = ['open', 'raise']
CLOSE_KEYWORDS = ['close', 'lower']


class Blinds:
    """Class that handles the smart blinds."""

    @staticmethod
    def handle(command_text: str) -> str:
        """Handles the smart blinds.
        Args:
            command_text: the command text
        Returns:
            the output of the command
        """
        if any(keyword in command_text for keyword in OPEN_KEYWORDS):
            Blinds.send_value(1)
            return 'Opening blinds'
        elif any(keyword in command_text for keyword in CLOSE_KEYWORDS):
            Blinds.send_value(0)
            return 'Closing blinds'
        else:
            return 'Unknown blinds command'

    @staticmethod
    def send_value(value: int) -> None:
        """Sends the given value to the blinds.
        Args:
            value: the value to send
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        try:
            ba = bytearray(struct.pack("b", value))
            s.send(ba)
        except:
            print("Error sending value")
        finally:
            s.close()
