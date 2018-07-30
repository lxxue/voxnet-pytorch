import numpy as np

acc = []
for i in range(1, 11):
    last_acc = -100000
    with open("log_{}/log.txt".format(i), 'r') as f:
        for line in f:
            if "Val Acc" in line:
                last_acc = float(line.split()[2])

    acc.append(last_acc)

print acc
print np.mean(acc)
print np.std(acc)
