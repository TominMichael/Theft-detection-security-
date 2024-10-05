Theft Detection with Intel oneAPI

This project uses AI and computer vision techniques to detect weapons and masks in real-time video streams for theft prevention. Upon detection, the system sends alerts via Telegram, records the footage, and provides timely responses. The project leverages Intel's oneAPI, TensorFlow, OpenCV, Telegram API, and NumPy for real-time analysis and communication.
Features

    Real-time video stream analysis: Detects weapons and masks in live video streams using TensorFlow and OpenCV.
    AI-Powered Detection: Uses TensorFlow for accurate model predictions.
    Alert System: Sends instant alerts via Telegram API when a threat is detected.
    Footage Recording: Captures and stores video evidence upon detection.
    Optimized with Intel oneAPI: Enhanced performance leveraging Intel's oneAPI.

Prerequisites

To run this project, you will need:

    Intel oneAPI: For optimized AI inference and performance enhancements.
    Python 3.x
    TensorFlow
    OpenCV
    NumPy
    Telegram Bot API

You can install the necessary libraries by running:

bash

pip install tensorflow opencv-python-headless numpy python-telegram-bot

Setup
Step 1: Intel oneAPI Installation

To take advantage of Intel oneAPI, follow the installation steps from the official Intel documentation: Intel oneAPI Installation Guide.
Step 2: Setting up the Telegram Bot

    Create a new bot on Telegram by messaging BotFather.
    Copy the bot token provided by BotFather.
    Replace the TOKEN in the code with your bot token.

Step 3: Clone the Repository

bash

git clone https://github.com/yourusername/theft-detection-oneapi.git
cd theft-detection-oneapi

Step 4: Model Training (Optional)

If you want to train your model:

    Capture and label your dataset with frames of "weapons" and "no weapons."
    Use TensorFlow to train the detection model.

For model training, the repository includes a script train.py. Run the script to train the model.

bash

python train.py

Step 5: Running the Project

Make sure your webcam is connected or your video stream is set up, and execute:

bash

python main.py

This will start the real-time detection process.
Configuration

In the config.py file, you can configure:

    Telegram Bot Token: Replace TOKEN with your bot's API key.
    Recording Settings: Specify the directory where footage will be stored upon threat detection.
    Thresholds: Adjust sensitivity for mask and weapon detection.

Project Structure

bash

theft-detection-oneapi/
├── models/             # Trained models
├── data/               # Training datasets
├── src/
│   ├── main.py         # Main application code
│   ├── train.py        # Model training script
│   ├── detection.py    # Weapon and mask detection logic
│   ├── telegram_alert.py  # Telegram bot integration
├── config.py           # Configuration file for bot and thresholds
├── README.md           # Project documentation

Code Explanation

    main.py: Runs the real-time video analysis using OpenCV. Frames are passed through the TensorFlow model, and detections are analyzed. If a threat is detected, a message is sent to Telegram, and the footage is recorded.
    detection.py: Contains the logic for weapon and mask detection using the trained TensorFlow model.
    telegram_alert.py: Handles sending notifications to Telegram using the Telegram API.
    train.py: Trains the TensorFlow model with weapon and mask datasets.
    config.py: Configuration settings for Telegram bot tokens, recording, and detection thresholds.

Usage

    Real-time monitoring: Once started, the system will monitor live video streams and automatically send alerts if a potential threat is detected.
    The Telegram bot will provide notifications with details of the detection and the footage will be stored for further review.

Performance Optimization with Intel oneAPI

This project leverages Intel oneAPI to enhance AI inference performance, making the theft detection faster and more efficient, especially on Intel hardware. Intel's AI Kit provides better utilization of CPU and GPU resources, offering accelerated performance for TensorFlow models.
Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

    Intel oneAPI: For providing powerful AI optimization tools.
    TensorFlow: For AI-based model training and predictions.
    OpenCV: For image processing and real-time video analysis.
    Telegram API: For sending instant alerts during theft detection.



    
