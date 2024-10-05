# AI AI Security and Thief Detection

This project implements a real-time AI-based security system capable of detecting weapons and masks from a video stream. When a weapon is detected, the system records the video and sends an alert message via Telegram. This project can be utilized for security applications such as surveillance in sensitive areas.
Features

    Real-Time Detection: Detects weapons and masks in live video feeds.
    Telegram Alerts: Sends real-time alerts to a Telegram chat when a weapon or mask is detected.
    Video Recording: Automatically starts recording video footage when a weapon is detected.
    AI Model Integration: Uses a pre-trained TensorFlow model (.h5) for inference.
    Intel oneAPI: The project leverages Intel's oneAPI for enhanced performance and optimized inference across Intel hardware platforms.

Technologies Used

    Intel oneAPI: For optimized AI performance and inference acceleration on Intel-based platforms.
    TensorFlow/Keras: For loading the pre-trained AI model and running inference.
    OpenCV: For handling video capture, preprocessing, and displaying results.
    NumPy: For data manipulation and preparation.
    Telegram API: To send real-time alerts when specific objects (weapons or masks) are detected.
    Asyncio: To ensure non-blocking execution of Telegram alerts during live video processing.

Project Structure

    main.py: The primary script that handles real-time video processing, inference, and Telegram alerts.
    newnew.h5: The pre-trained TensorFlow/Keras model used for object detection.
    README.md: Documentation for the project.

Prerequisites

To run this project, ensure you have the following installed:

bash

pip install tensorflow opencv-python numpy python-telegram-bot intel-openvino

Intel oneAPI Setup

Ensure that you have Intel's oneAPI toolkit installed, which includes support for AI inference acceleration. You can download the toolkit from Intel's oneAPI website.
Telegram Setup

    Create a bot using BotFather on Telegram and obtain the bot token.
    Send a message to your bot to initiate communication.
    Retrieve your chat_id from Telegram (you can use the getUpdates API to do this).

Usage

    Clone the repository:

bash

git clone <repository-url>
cd <repository-directory>

    Replace the placeholders:
        Replace "YOUR_TELEGRAM_BOT_TOKEN" with your Telegram bot token.
        Replace "YOUR_CHAT_ID" with the chat ID of your Telegram account.

    Run the script:

bash

python main.py

How It Works

    The system captures video frames from your camera.
    Each frame is processed and passed through the pre-trained TensorFlow model for inference.
    Intel oneAPI enhances the inference process, optimizing it for Intel hardware to accelerate detection and reduce latency.
    Based on the prediction, if a weapon is detected:
        A Telegram alert is sent to notify about the detection.
        Video recording starts automatically.
    If a mask is detected, a separate Telegram alert is sent.
    The detection labels ("Weapon Detected", "Mask Detected", "None") are displayed in real-time on the video.

Customization

    Model: You can swap the newnew.h5 file with any other pre-trained model that suits your detection needs. Ensure that the input shape and preprocessing steps match the new model.
    Intel oneAPI: Leverage Intel's oneAPI for further optimization by fine-tuning the inference process to utilize Intel's hardware acceleration for deep learning workloads.
    Alert Messages: You can customize the alert messages in the script by changing the text in the bot.send_message() calls.

Troubleshooting

    No Telegram Messages: Ensure that the bot token and chat ID are correct. You may also need to interact with the bot by sending it a message before it can send you alerts.
    Model Accuracy: If the model's predictions are inaccurate, ensure that the input size, preprocessing, and normalization steps are aligned with how the model was trained.
    Performance Issues: Make sure you have Intel's oneAPI toolkit installed and configured properly to take advantage of AI acceleration. Refer to Intel's documentation if necessary.

Future Enhancements

    Add support for detecting additional objects beyond weapons and masks.
    Implement cloud storage integration for saving recorded videos.
    Enhance accuracy by fine-tuning the model on specific datasets for security-related use cases.
    Further optimize the system with Intel oneAPI for enhanced performance on Intel hardware platforms.

License

This project is licensed under the MIT License - see the LICENSE file for details.
