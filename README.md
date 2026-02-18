# HapticTrain AI: Dual-Layered Athletic Intelligence

HapticTrain AI is a holistic performance system that bridges the gap between digital intelligence and physical execution. It uses real-time computer vision and wearable haptics to provide instant, tactile form correction while utilizing NLP to monitor athlete burnout and recovery.

## 🚀 Key Features

* **Real-Time Biomechanical Tracking:** Uses MediaPipe to extract 33-point skeletal coordinates at 30+ FPS for instant form analysis.
* **Tactile Feedback Loop:** Delivers sub-50ms haptic corrections via an ESP32-powered wearable to adjust form without visual distraction.
* **"Showstopper" NLP Sentinel:** Integrated Gemini 3 Flash API analyzes verbal check-ins to detect CNS fatigue and psychological stress.
* **Adaptive Logic Switch:** Automatically pivots between high-intensity coaching and recovery-focused de-loading based on physical and mental data.

## 🛠️ Tech Stack

### Software
* **Vision:** MediaPipe, OpenCV
* **Intelligence:** Google Gemini 3 Flash (NLP & Multimodal)
* **Backend:** Python (NumPy, Pandas)
* **Dashboard:** Streamlit, Plotly

### Hardware
* **Microcontroller:** ESP32 (Dual-core, BLE enabled)
* **Actuators:** ERM Vibration Motors
* **Communication:** Bluetooth Low Energy (BLE)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/haptictrain-ai.git](https://github.com/yourusername/haptictrain-ai.git)
   cd haptictrain-ai
