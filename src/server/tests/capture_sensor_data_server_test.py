"""capture_sensor_data_server_test.py: Capture ultrasonic sensor data on server side"""

import socket
import time
import cv2

import sys
from path import Path

coreDir = Path(__file__).parent.parent.parent
sys.path.append(coreDir.abspath())

from core.utils import *

# image = cv2.imread('src/server/tools/white.png')
font = cv2.FONT_HERSHEY_SIMPLEX

class CaptureSensorDataTest(object):
    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(sensor_data_stream_address)
        self.sock.listen(1)
        self.connection, self.client_address = self.sock.accept()
        self.capture_data()
        

    def capture_data(self):
        try:
            print("Connection from: {0}".format(self.client_address))
            start = time.time()

            while True:
                sensor_data = float(self.connection.recv(1024))
                text = "Distance: {0:.1f} cm".format(sensor_data)
                print(text)

                image = cv2.imread('src/server/tools/white.png')
                cv2.putText(image, text, (50, 400), font, 4, (255, 0, 0), 5, cv2.LINE_AA)
                cv2.imshow('Sensor data', cv2.resize(image, (650, 175)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Test for 1 min
                # if time.time() - start > 300:
                #     break
        finally:
            self.connection.close()
            self.sock.close()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    CaptureSensorDataTest()