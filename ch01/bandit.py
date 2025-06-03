import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)  # 各マシンの確率

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    # 価値の推定関数
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        # epsilonの確率で探索
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        # それ以外は現在のQ値が最大のアクションを選択
        return np.argmax(self.Qs)


def main():
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        # 勝った割合
        rates.append(total_reward / (step + 1))

    print(f"Total reward: {total_reward}")
    print(f"Estimated rates: {agent.Qs}")
    print(f"Actual rates: {bandit.rates}")
    print(f"Estimated best action: {np.argmax(agent.Qs)}")
    print(f"Actual best action: {np.argmax(bandit.rates)}")

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()


if __name__ == '__main__':
    main()
