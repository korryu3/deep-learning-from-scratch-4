V = {'L1': 0.0, 'L2': 0.0}
new_V = V.copy()

cnt = 0
threshold = 0.0001
while True:
    new_V['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    new_V['L2'] = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])

    # 更新されたVと前のVの差の最大値
    # どちらもthreshold以下にする
    delta = abs(new_V['L1'] - V['L1'])
    delta = max(delta, abs(new_V['L2'] - V['L2']))
    V = new_V.copy()

    cnt += 1
    if delta < threshold:
        print(V)
        print(cnt)
        break
