# Starcraft 2 AI Bot

This project is an AI bot for **Starcraft 2** created to explore neural networks and AI decision-making in the context of a real-time strategy game. The AI is designed to handle both **macro** (economy management) and **micro** (unit control) strategies within a single neural network. Though it's still a work-in-progress, the bot performs decently and can make decisions faster than a human player.

Originally, the AI was designed to use three separate neural networks:
- **Macro** network: for economy management
- **Micro** network: for unit control and tactical decisions
- **Decision network**: to decide which of the above two networks to use at a given moment.

However, due to communication issues between these networks, the project was simplified to a single neural network that handles both macro and micro in one.

## Features

- **Single Neural Network**: A single neural network is used to handle both macro and micro decision-making.
- **Real-Time Decision Making**: The AI makes decisions extremely fast, resulting in quick responses during gameplay.
- **State-based Input**: The network takes in the current state of the game to decide actions, improving the AI's ability to respond to dynamic game situations.
- **Training Data**: The project has been trained a little, but the reward function still needs refining. It has been tested using various reward functions, such as winning the game and measuring combined economy and army.
- **Performance**: Despite training limitations, the AI can still make decisions faster than a human player going against it.

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/username/Starcraft2-AI.git
Install the necessary Python libraries:
  - pip install tensorflow sc2 numpy
  
  Starcraft 2 Installation:

  - Install Starcraft 2 on your machine. The sc2 Python API needs to interact with it.
    Manually change the path to the location where Starcraft 2 is installed on your computer in the project files.
    The project uses TensorFlow for the neural network and sc2 for the interaction with Starcraft 2.

Running the AI
To run the bot:

Run the neural network script (please note the typo in the file name is intentional, as part of the project):

python UnifiedNetowrk.py
This will start the bot, which will use its trained neural network to make decisions in the game.

Training the AI
  - The UnifiedNetowrk.py file contains the neural network and decision-making logic.
  - combinedmodel.keras contains the training data.
## Reward Function

The reward function is still not optimized. Previous attempts to use "just winning" or "economy/army combined measurement" 
did not yield optimal results. More refinement is required.

**Note**: Training the AI is not currently possible on the developer's machine due to issues with **Starcraft 2** not running properly. Once the issues are resolved, further training can be conducted.

## Technologies Used
- **Python**: The core language used for the project.
- **TensorFlow**: For building and training the neural network.
- **sc2 API**: Python API for interacting with **Starcraft 2**, controlling bots, and running games.
- **Numpy**: For numerical operations used in the neural network and game state processing.

### Libraries
- `tensorflow`
- `numpy`
- `sc2`
- `os`
- `time`
- `random`
- `collections` (deque)

## Future Plans

- **Improved Reward Function**: After taking AI classes at college, the plan is to refine the reward function for better performance.
- **More Training**: The goal is to continue training the neural network to improve its decision-making ability.
- **Multiple Neural Networks**: Eventually, I want to return to the original plan of having separate neural networks for **macro** and **micro** decision-making.
- **Further Game Strategy**: Additional features, such as better economy management, tactical decisions, and unit control, will be explored in the future.

## Acknowledgements

This project was created out of my love for **Starcraft 2** and my desire to learn AI and machine learning. 
I hope to revisit the project in the future after taking more AI courses and apply new concepts to improve the bot.
