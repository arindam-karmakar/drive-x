import serial
import serial.tools.list_ports

# Server
server_address = ('192.168.60.231', 45713)

server_ip = '192.168.53.253'

video_stream_address = (server_ip, 45710)
sensor_data_stream_address = (server_ip, 45720)

# Raspberry Pi
rpi_ip = "192.168.53.78"
rpi_port = "49214"

rpi = rpi_ip + ", " + rpi_port

# Arduino
arduino_serial_number = '75830303934351011210' # '75237333536351F0F0C1' is the serial number of my Arduino

def find_arduino(serial_number):
    for p in serial.tools.list_ports.comports():
        if p.serial_number == serial_number:
            return serial.Serial(p.device)

    raise IOError("Could not find the Arduino - is it plugged in ?!?")