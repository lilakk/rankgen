import matplotlib.pyplot as plt

swaps = [1, 2, 3, 4, 5]

# entity swap
rank = [0.69, 0.817, 0.839, 0.875, 0.886]
diff = [0.472, 0.913, 1.253, 1.572, 1.706]
ppl = [0.91, 0.95, 0.977, 0.99, 0.997]

fig, ax = plt.subplots()
ax.plot(swaps, rank, label = 'pct of times gold rank > perturbed rank')
ax.plot(swaps, diff, label = 'avg diff between gold rank and perturbed rank')
ax.plot(swaps, ppl, label = 'pct of times gpt2 gold ppl < perturbed ppl')
plt.title('Entity swapping')
ax.legend(loc = 'upper left')
ax.set(xlabel='# of swaps')
plt.show()

# sentence swap
rank = [0.772, 0.876, 0.902, 0.96, 0.962]
# diff = [2.009, 3.501, 4.546, 6.208, 6.858]
ppl = [0.873, 0.949, 0.98, 0.983, 0.981]

fig, ax = plt.subplots()
ax.plot(swaps, rank, label = 'pct of times gold rank > perturbed rank')
# ax.plot(swaps, diff, label = 'avg diff between gold rank and perturbed rank')
ax.plot(swaps, ppl, label = 'pct of times gpt2 gold ppl < perturbed ppl')
plt.title('Entity swapping')
ax.legend(loc = 'upper left')
ax.set(xlabel='# of swaps')
plt.show()