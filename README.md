# Tennis AI (DQN)

This project implements the Deep Q-Network (DQN) algorithm to play the Atari 2600 game "Tennis." The implementation is based on the research presented in the following papers:

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2013). Playing Atari with deep reinforcement learning. _arXiv preprint arXiv:1312.5602_.
- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2017). Revisiting the Arcade Learning Environment: Evaluation protocols and open problems for general agents. _arXiv preprint arXiv:1709.06009_.

## Description

This repository contains the code to train and evaluate a DQN agent on the Atari 2600 Tennis game. The DQN algorithm uses deep learning to approximate the Q-value function, which is used to determine the best action to take in each state of the game. The project leverages the [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment for simulation and training.

## Installation and running

To run this project, you need to have Python 3.10 or higher installed. Follow these steps to set up the environment:

1. Clone the repository:

   ```sh
   git clone https://github.com/twet123/tennis-ai.git
   cd tennis-ai
   ```

2. Create and activate a virtual environment:

   ```sh
   python3 -m venv venv
   source venv/bin/activate   # Windows `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Train the model:

   You can set the values of hyperparameters in `config.py`

   ```sh
   python3 main.py
   ```

   Every 10000 frames the model is saved to `tennis-model.keras`. Also, the model is saved whenever you interrupt/stop the script (Ctrl + C).

5. Evaluating the model:

   ```sh
   python3 play.py
   ```
