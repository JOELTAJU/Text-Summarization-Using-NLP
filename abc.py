import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['figure.figsize'] = [15, 15]
sns.set(font_scale=1, style='whitegrid')
plt.grid(True)
df_author = df.author.value_counts().head(50)
sns_author = sns.barplot(x=df_author.values, y=df_author.index)
sns_author.set_ylabel("Author", fontsize=12)
sns_author.set_xlabel("Count", fontsize=8)
sns_author.set_title("The Most Frequent Authors")
plt.yticks(fontsize=8)
plt.xticks(fontsize=8, rotation=0, ha='right')
sns.despite(left=True, bottom=True, axis='both')
sns.set_style('ticks')
plt.subplots_adjust(bottom=0.6, hspace=1, wspace=0.3)
plt.tight_layout()
plt.show()
