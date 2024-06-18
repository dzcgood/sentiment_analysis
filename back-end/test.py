import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
plt.style.use('fivethirtyeight')

data = {'happy': [817, 203], 'sad': [693, 402], 'fear': [164, 88], 'surprise': [199, 88], 'angry': [1212, 239],
        'neutral': [793, 102]}

x_labels = ['angry', 'happy', 'sad', 'neutral', 'fear', 'surprise']
correct = [data[label][0] for label in x_labels]
incorrect = [data[label][1] for label in x_labels]

x = np.arange(len(x_labels))
width = 0.8

fig, ax = plt.subplots()
rects1 = ax.bar(x, correct, width, color='#336699')
rects2 = ax.bar(x, incorrect, width, color='#FF6600', bottom=correct)

ax.set_ylabel('数量')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend((rects1[0], rects2[0]), ('正确', '错误'))

for index, rect in enumerate(rects1):
    height = rect.get_height() / 2
    ax.annotate('{}'.format(correct[index]),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='center')

for index, rect in enumerate(rects2):
    height = rect.get_height() / 2 + correct[index]
    ax.annotate('{}'.format(incorrect[index]),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='center')

plt.savefig('模型在各标签下的表现.svg', bbox_inches="tight")
plt.show()

