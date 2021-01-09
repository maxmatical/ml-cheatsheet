## Reinforcement Learning

### Stable-baselines library for RL algorithms
https://stable-baselines3.readthedocs.io/en/master/index.html#

### Stable-baselines3-contrib library for QR-DQN (SOTA)
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

https://sb3-contrib.readthedocs.io/en/master/index.html

### RL Algorithm choices

1. Discrete Actions - Single Process
    - [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
    - [QR-DQN](https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html) might be better

2. Discrete/Continuous Actions - Multiprocess
    - [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
    - [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)

3. Continuous Actions - Single Process
    - [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
    - [TQC](https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html)

Note: For faster wall-time, PPO and A2C are fastest. DQN/QR-DQN are slower wall-time, but more sample efficient

### Creating custom gym environments:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html 