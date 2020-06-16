import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../output/mod2pix_out_200.csv', sep=',', header=None)

fig = plt.figure()

ax1 = plt.subplot2grid((2, 1), (0, 0))
ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

ax1.plot(df.index, df[4], label="D loss")
ax1.legend()

ax2.plot(df.index, df[5], label="G loss")
ax2.legend()

# plt.show()
plt.savefig('../output/loss_graph.png', dpi=300)
