import numpy as np

# naive implementation
np.random.seed(0)
rewards = []

for n in range(1, 11):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)

print('---')

# incremental implementation
# 毎回listに追加したりsumを計算するのは非効率的なので、
# 漸化式を使い、逐次的に平均を更新する方法を使う
np.random.seed(0)
Q_n_1 = 0

for n in range(1, 11):
    reward = np.random.rand()
    Q = Q_n_1 + (reward - Q_n_1) / n  # 1/nは学習率と言える. 試行回数nが増えるとQは更新されなくなる
    Q_n_1 = Q  # 前の平均を保存
    print(Q)
