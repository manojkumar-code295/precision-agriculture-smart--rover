# Hardware Wiring & Electrical Safety Guide
**Plant Disease Detection AI – Precision Agriculture Smart Rover**

---

## ⚠️ Electrical Logic Warning

- **Raspberry Pi 4B operates on 3.3V logic on all 40 GPIO pins.**
- **5V logic inputs will permanently damage the Broadcom BCM2711 SoC.**
- Always power high-current inductive loads (DC motors, water pumps) using dedicated battery packs or buck converters, NOT directly from the Raspberry Pi 5V rail.

---

## 1. L298N Dual H-Bridge Motor Driver Pin Mapping

| L298N Terminal | Raspberry Pi 4 Pin (BCM) | Physical Header Pin | Function |
| :--- | :--- | :--- | :--- |
| **ENA** | GPIO 22 | Pin 15 | Left Motor Speed PWM (0-100%) |
| **IN1** | GPIO 17 | Pin 11 | Left Motor Direction 1 |
| **IN2** | GPIO 27 | Pin 13 | Left Motor Direction 2 |
| **IN3** | GPIO 23 | Pin 16 | Right Motor Direction 1 |
| **IN4** | GPIO 24 | Pin 18 | Right Motor Direction 2 |
| **ENB** | GPIO 25 | Pin 22 | Right Motor Speed PWM (0-100%) |
| **GND** | GND | Pin 6 / Pin 9 | Common Ground (Must be shared) |
| **+12V / Battery** | External Battery (+) | - | Motor Power Supply |

---

## 2. HC-SR04 Ultrasonic Sensor & Voltage Divider

The HC-SR04 requires 5V VCC to emit ultrasonic waves. Consequently, its **ECHO** pin outputs a 5V signal that must be stepped down to ~3.3V before connecting to the Raspberry Pi.

```text
HC-SR04 Sensor:
  VCC  --------> Raspberry Pi Pin 2 (5V Power)
  GND  --------> Raspberry Pi Pin 6 (Ground)
  TRIG --------> Raspberry Pi Pin 29 (GPIO 5)

  ECHO --------> [ 1 kΩ Resistor ] -------> Raspberry Pi Pin 31 (GPIO 6)
                                      |
                                      +----> [ 2 kΩ Resistor ] ----> GND
```

**Voltage Divider Formula:**  
$$V_{out} = 5V \times \frac{2000\,\Omega}{1000\,\Omega + 2000\,\Omega} = 3.33\,V$$

---

## 3. Sprayer Relay Module

| Relay Module Pin | Raspberry Pi 4 Pin (BCM) | Physical Header Pin | Function |
| :--- | :--- | :--- | :--- |
| **VCC** | 5V | Pin 4 | Relay Coil Power |
| **GND** | GND | Pin 14 | Ground |
| **IN** | GPIO 26 | Pin 37 | Sprayer Trigger Signal |
| **COM** | Pump Battery (+) | - | Switch Common |
| **NO** | 12V DC Pump (+) | - | Normally Open Contact |

---

## 4. Raspberry Pi Camera Module

1. Locate the **CAM / CSI** port between the HDMI micro ports and audio jack.
2. Gently pull the plastic clip upwards.
3. Insert the flat ribbon cable with the **silver contacts facing the HDMI ports** and blue backing facing the audio jack.
4. Push the plastic clip firmly down to lock the cable in place.
5. Verify connection in terminal:
   ```bash
   rpicam-hello --list-cameras
   ```
