# **CartPole DQN AI Agent** 🚀  

A **Deep Q-Learning AI Agent** for **CartPole-v1** using **TensorFlow & OpenAI Gym**.  
The agent learns to balance a pole through **reinforcement learning**, using **experience replay** and **target networks**.  

---

## 📌 **Overview**  
This project demonstrates how to train an **AI agent** using **Deep Q-Learning (DQN)** to play **CartPole-v1**, a classic reinforcement learning environment.  
The agent learns **optimal actions** over time by interacting with the environment, storing experiences, and improving its decision-making process through deep learning.  

---

## **🚀 Features**  
✔ **Deep Q-Network (DQN) implementation** using TensorFlow  
✔ **Experience Replay** to improve learning stability  
✔ **Target Q-Network** for better convergence  
✔ **Epsilon-Greedy Exploration** to balance exploration & exploitation  
✔ **Trained model saving & evaluation**  
✔ **Runs on CPU, GPU, and Apple M1/M2 (TensorFlow-Metal)**  

---

## 🛠 **Installation & Setup**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/your-username/cartpole-dqn-ai-agent.git
cd cartpole-dqn-ai-agent
```

### **2️⃣ Install Dependencies**  
To install the required dependencies, run:  
```sh
pip install -r requirements.txt
<br>
If you are using Apple M1/M2, install TensorFlow with GPU support:

```sh
pip install tensorflow-macos tensorflow-metal

### **3️⃣ Run Training**  
To train the agent, run:  
```sh
python main.py
---

### 📈 **Training Results**
Below is the training progress:
```sh
Episode 100 | Avg Reward (last 100 eps): 17.76 | Epsilon: 0.606
Episode 200 | Avg Reward (last 100 eps): 43.13 | Epsilon: 0.367
Episode 300 | Avg Reward (last 100 eps): 110.69 | Epsilon: 0.222
Episode 354 | Avg Reward (last 100 eps): 201.86 | Epsilon: 0.170 ✅
Environment solved in 354 episodes!

Total Training Time: 599.87 s (10.00 min)

