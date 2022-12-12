#include <AFMotor.h>

AF_DCMotor motor1(1, MOTOR12_1KHZ);
AF_DCMotor motor2(2, MOTOR12_1KHZ);
AF_DCMotor motor3(3, MOTOR34_1KHZ);
AF_DCMotor motor4(4, MOTOR34_1KHZ);

int command = 0;

void setMotorSpeed(uint8_t motor1speed, uint8_t motor2speed, uint8_t motor3speed, uint8_t motor4speed) {
  motor1.setSpeed(motor1speed);
  motor2.setSpeed(motor2speed);
  motor3.setSpeed(motor3speed);
  motor4.setSpeed(motor4speed);
}

void setup() {
  // put your setup code here, to run once:
  setMotorSpeed(150, 150, 150, 150);

  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0)
  {
    command = Serial.read();
  }
  else
  {
    reset();
  }

  drive(command);
}

void reset()
{
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
}

void forward(bool isHybridCmd) {
  if(!isHybridCmd) {
    setMotorSpeed(150, 150, 150, 150);
    motor1.run(FORWARD);
    motor2.run(FORWARD);
  }

  motor3.run(FORWARD);
  motor4.run(FORWARD);
}

void forward_right()
{
  setMotorSpeed(255, 255, 100, 150);
  motor1.run(FORWARD);
  forward(true);
}

void forward_left()
{
  setMotorSpeed(255, 255, 150, 100);
  motor2.run(FORWARD);
  forward(true);
}

void reverse(bool isHybridCmd) {
  if(!isHybridCmd) {
    setMotorSpeed(255, 255, 150, 150);
    motor3.run(BACKWARD);
    motor4.run(BACKWARD);
  }

  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
}

void reverse_right()
{
  setMotorSpeed(255, 215, 150, 150);
  motor4.run(BACKWARD);
  reverse(true);
}

void reverse_left()
{
  setMotorSpeed(215, 255, 150, 150);
  motor3.run(BACKWARD);
  reverse(true);
}

void right()
{
  setMotorSpeed(150, 150, 150, 150);
  motor1.run(FORWARD);
  motor4.run(FORWARD);
}

void left()
{
  setMotorSpeed(150, 150, 150, 150);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
}

void drive(int command)
{
  switch(command)
  {
    case 48: reset(); break;
    case 49: forward(false); break;
    case 50: reverse(false); break;
    case 51: right(); break;
    case 52: left(); break;
    case 53: forward_right(); break;
    case 54: forward_left(); break;
    case 55: reverse_right(); break;
    case 56: reverse_left(); break;
    default: break;
  }
}
