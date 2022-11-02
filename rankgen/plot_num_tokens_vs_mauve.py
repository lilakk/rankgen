import matplotlib.pyplot as plt

num_tokens = [8, 16, 32, 64, 128]

# on the 1000 shuffled instances
comet = [0.9728420395856243, 0.9624867551438443, 0.9473466261586049, 0.9502048476602636, 0.9321488701139485]
rankgen = [0.9474463563734028, 0.9307624371979657, 0.9325065714163447, 0.9429366837471573, 0.9165721148896158]

fig, ax = plt.subplots()
ax.plot(num_tokens, comet, label = 'COMET')
ax.plot(num_tokens, rankgen, label = 'RankGen')
plt.title('Impact of num_tokens on MAUVE score')
ax.legend(loc = 'upper left')
ax.set(xlabel='# tokens generated before reranking')
ax.set(ylabel='MAUVE')
plt.show()
