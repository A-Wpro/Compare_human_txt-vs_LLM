import plotly.express as px
import pandas as pd
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
path_txt = r"C:\Users\L1061703\workspace\txtGPT.txt"
with open(path_txt, 'r', encoding='utf-8') as file:
    text = file.read()

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing: Remove numbers, convert to lowercase, and remove stop words
text = re.sub(r'\d+', '', text)  # Remove numbers
text = text.lower()  # Convert to lowercase
words = nltk.word_tokenize(text)  # Tokenize words
filtered_words = [word for word in words if word not in stop_words and word.isalpha()]  # Remove stop words and non-alpha characters

# Calculate word frequency
word_freq = Counter(filtered_words)
# Create a DataFrame with words and their frequencies
df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])

# Sort DataFrame by frequency
df = df.sort_values(by='Frequency', ascending=False)

# Add a column for rank
df['Rank'] = df['Frequency'].rank(method='min', ascending=False)

# Plot Zipf's distribution using Plotly Express
fig = px.line(df, x='Rank', y='Frequency', log_x=True, log_y=True)
fig.update_traces(textposition='top center')
fig.update_layout(
    title="Zipf's Distribution Plot of input data",
    xaxis_title="Rank",
    yaxis_title="Frequency",
    showlegend=False,
)

fig.show()
