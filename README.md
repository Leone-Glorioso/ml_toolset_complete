# The ML Engineer's Scrapyard: A Collection of Small Tutorials for Everything ML

Welcome to **The ML Engineer's Scrapyard**! This repository is a comprehensive, hands-on collection of Jupyter Notebook tutorials covering a massive spectrum of Machine Learning, Deep Learning, MLOps, and Robotics concepts. Whether you are building your first linear regression model or deploying a multi-agent reinforcement learning system, you will find a focused, analytical guide here.

## 📂 Repository Structure

All tutorials are contained within the root directory as interactive Jupyter Notebooks (`.ipynb`). To make navigating the "scrapyard" easier, the files are categorized by their overarching domain below:

### 🛠 Data Science Foundations & Preprocessing
* **`data_cleansing.ipynb`**: The foundation of ML. Covers techniques for handling missing data (imputation), outlier detection (IQR clipping), categorical encoding (one-hot), and feature scaling (Standardization).
* **`eda.ipynb`**: Explanatory Data Analysis. Teaches you how to profile a dataset, visualize distributions, and discover relationships and multicollinearity using statistical graphics.
* **`plotting.ipynb`**: A complete guide to the five essential plots every ML engineer needs (Distributions, Scatter, Boxplots, Heatmaps, Learning Curves) using Seaborn and Matplotlib.

### 📈 Classical Machine Learning
* **`linear_regression.ipynb`**: Regression Part 1. Introduces Simple Linear Regression, Ordinary Least Squares (OLS), and evaluation metrics like MSE and RMSE using synthetic real estate data.
* **`multilinear_regression.ipynb`**: Regression Part 2. Scales up to Multiple Linear Regression, focusing on matrix math, feature interpretability, and the importance of Adjusted R-Squared.
* **`logistic_regression.ipynb`**: Regression Part 3. The bridge to classification. Explains the Sigmoid activation function, Decision Boundaries, and the Confusion Matrix.
* **`clustering.ipynb`**: The ultimate guide to unsupervised learning. Compares centroid-based K-Means with density-based DBSCAN, evaluated via the Silhouette Score.

### 🧠 Deep Learning Fundamentals
* **`neural_networks.ipynb`**: A complete guide to building Neural Networks from scratch using **PyTorch**. Demonstrates how Multi-Layer Perceptrons (MLPs) and activation functions solve non-linear data (the "moons" dataset).
* **`neural_networks_tensorflow.ipynb`**: A parallel guide to Neural Networks, this time utilizing the **TensorFlow/Keras** Sequential API to separate concentric circles.

### 👁️ Computer Vision
* **`image_classification.ipynb`**: CV 101. Builds a Convolutional Neural Network (CNN) in TensorFlow to classify the CIFAR-10 dataset using spatial hierarchies (Convolutions and Pooling).
* **`image_segmentation.ipynb`**: CV 102. Moves to pixel-level semantic segmentation using a pre-trained DeepLabV3 model (ResNet-50 backbone) in PyTorch.
* **`morphological_operation.ipynb`**: The architectural level of pixel manipulation. Teaches Erosion, Dilation, Opening, Closing, and Morphological Gradients using OpenCV.

### 🗣️ Natural Language Processing & Generative AI
* **`transformers.ipynb`**: Demystifies the Transformer architecture (Tokenization, Embeddings, Attention). Demonstrates Sequence Classification (BERT) and Autoregressive Generation (GPT-2).
* **`llm.ipynb`**: A definitive guide to implementing Large Language Models locally. Explains tokenization, the forward pass, and decoding strategies (Greedy vs. Temperature Sampling) using GPT-2.
* **`rag.ipynb`**: Retrieval-Augmented Generation. Teaches how to ground LLMs in verifiable facts using Document Chunking, HuggingFace Embeddings, FAISS vector databases, and Prompt Augmentation.
* **`vae.ipynb`**: Variational Autoencoders. Explores Generative Deep Learning by building an Encoder/Decoder architecture with a reparameterization trick to generate fake handwritten digits (MNIST).
* **`multimodal_data.ipynb`**: Bridges vision and language using the **CLIP** model. Demonstrates Zero-Shot Image Classification and Text-to-Image Search using Cosine Similarity in a shared latent space.

### ⏳ Sequence Modeling (Time-Series)
* **`rnn.ipynb`**: An introduction to Recurrent Neural Networks (RNNs) in PyTorch, exploring hidden states and Backpropagation Through Time (BPTT) to predict a damped sine wave.
* **`lstm.ipynb`**: Introduces Long Short-Term Memory networks. Explains the mathematics of Forget, Input, and Output gates to solve the vanishing gradient problem in sequential data.

### 🎧 Audio & Speech Processing
* **`voice_to_text.ipynb`**: Automatic Speech Recognition (ASR). Builds a transcription pipeline using OpenAI's **Whisper** model, converting raw 16kHz waveforms to Mel-Spectrograms, and finally to text.
* **`text_to_voice.ipynb`**: Neural Speech Synthesis. Implements **SpeechT5**, demonstrating how to use speaker embeddings, spectrogram generation, and a HiFi-GAN Vocoder to generate realistic human audio.

### 🎮 Reinforcement Learning & AI Search
* **`qlearning.ipynb`**: Classical RL. Implements tabular Q-Learning and the Bellman Equation to navigate a deterministic grid (FrozenLake) balancing Exploration vs. Exploitation ($\epsilon$-greedy).
* **`ppo.ipynb`**: State-of-the-art RL. Introduces Proximal Policy Optimization (PPO), an Actor-Critic method, to solve the CartPole balancing problem using `stable-baselines3`.
* **`multiagentrl.ipynb`**: Multi-Agent RL (MARL). Explores Markov Games and the Non-Stationarity problem. Uses PettingZoo and SuperSuit parameter sharing to train cooperative agents.
* **`genetic_algorithms.ipynb`**: Evolutionary Computation. Mimics biological evolution (Initialization, Selection, Crossover, Mutation) to optimize the 0/1 Knapsack Problem.
* **`pathfinding.ipynb`**: Classical AI graph search. Implements Dijkstra's Algorithm and **A* Search** (with Manhattan distance heuristics) using `networkx` to navigate around obstacles.

### 🤖 Robotics & HCI
* **`ros.ipynb`**: Robot Operating System (ROS 2). Explains distributed graph architectures by building a simulated Lidar sensor (Publisher) and an AI brain (Subscriber) communicating via Topics in Python (`rclpy`).
* **`slam.ipynb`**: Simultaneous Localization and Mapping. Implements the **Extended Kalman Filter (EKF) SLAM** to fuse noisy motor odometry with noisy sensor measurements using Jacobian matrices.
* **`human_computer_interaction.ipynb`**: Next-gen HCI. Combines OpenCV and Google's **MediaPipe** to build a non-touch interface that translates 3D hand tracking landmarks into spatial computer commands.

### ⚙️ MLOps & Production
* **`cicd.ipynb`**: The intersection of Software Engineering and Data Science. Builds a simulated ML CI/CD pipeline using `scikit-learn` for modeling, `MLflow` for experiment tracking, `pytest` for automated data/model evaluation, and `FastAPI` for continuous deployment.

---

## 🚀 Getting Started

To run the notebooks in this scrapyard, it is highly recommended to set up a virtual environment (e.g., `venv` or `conda`). Most notebooks rely on the standard modern ML stack:

* `numpy`, `pandas`, `matplotlib`, `seaborn`
* `scikit-learn`
* `torch` (PyTorch) or `tensorflow`
* `transformers`, `datasets` (Hugging Face)
* `opencv-python`
* `gymnasium`, `stable-baselines3`

Simply launch Jupyter Notebook or Jupyter Lab in the `mlops_tutorial/` directory and start experimenting! Each notebook is designed to be self-contained, analytical, and highly educational.