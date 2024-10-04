# CCTV Weapon or Thief Detection

## Overview

This project focuses on developing an AI-based system for detecting weapons or potential thieves through CCTV footage. The system uses computer vision, machine learning algorithms, and Intel's oneAPI to accelerate detection and improve performance. The goal is to enhance security measures in public and private spaces by providing early warning signals of potential threats.

## Features

- **Real-time detection:** Continuously monitors CCTV footage to detect the presence of weapons or suspicious behavior.
- **High-performance computing:** Utilizes Intel's oneAPI to optimize performance across heterogeneous hardware, including CPUs and GPUs.
- **Alert system:** Sends notifications when a weapon or suspicious activity is detected.
- **Customizable detection models:** Supports training with custom datasets for specific weapon types or thief behavior.
- **Scalable and adaptable:** Can be deployed on various hardware configurations, from local computers to cloud servers.
- **Integration options:** Works with messaging services to send alerts to phones, emails, or security systems.

## Getting Started

### Prerequisites

To set up and run the project, ensure you have the following installed:

- Python 3.8+
- OpenCV
- TensorFlow or PyTorch (Choose based on the model you're using)
- Numpy
- Scikit-learn
- Flask (for web deployment)
- Intel oneAPI Toolkit (for optimized performance)
- Twilio or any messaging service API for sending alerts

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cctv-weapon-thief-detection.git
   cd cctv-weapon-thief-detection
