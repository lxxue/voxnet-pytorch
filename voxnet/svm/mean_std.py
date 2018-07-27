import numpy as np

acc = []
with open("./log_acc.txt", 'r') as f:
    for line in f:
        a = float(line.strip('\n'))
        print a
        acc.append(a)

acc = np.array(acc)
print "-------"
print np.mean(acc)
print np.var(acc)
print np.std(acc)
