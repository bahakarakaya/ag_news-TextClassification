import google.generativeai as genai
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import os
from dotenv import load_dotenv


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ds = load_dataset("fancyzhx/ag_news")

random.seed(42)

inference_count = 100

article_indices = random.sample(range(len(ds['test'])), inference_count)
articles = [ds['test'][i] for i in article_indices]

PROMPT_TEMPLATE = r"""You are a media analyst evaluating the topic and focus of news articles for content classification.
Classify the following news article into one of the following categories: World, Sports, Businesses, Sci/Tech
First, write a concise one-sentence justification
Then, output ONLY the label on a new line.

Here are few examples, follow this exact format:
```
Text: Asteroid Toutatis Makes Closest Pass in 651 Years (Reuters) Reuters - An asteroid named for a Celtic god\of war will come as close to Earth this week as it has since\1353.
Justification: The article covers a scientific event involving an asteroid's close approach to Earth, which relates to space and technology.
Label: Sci/Tech

Text: NATO proclaims victory in Bosnia SARAJEVO, Bosnia-Herzegovina -- When NATO forces first came to Bosnia nearly a decade ago, they lived in heavily guarded compounds, patrolled the streets in tanks, and often wore full body armor.
Justification: The article discusses international military intervention and geopolitical developments involving NATO.
Label: World

Text: Google IPO: Type in 'confusing,' 'secrecy' I've submitted my bid to buy shares of Google Inc. in the computer search company's giant auction-style initial public offering. That could turn out to be the good news or the bad news.
Justification: The article focuses on Google's financial actions and the process of its public stock offering, which is a business topic.
Label: Businesses

Text: Hamilton Sets Early Pace as Woods Struggles KOHLER, Wis. (Reuters) - British Open champion Todd Hamilton made the first significant move in the U.S. PGA Championship final round Sunday as overnight pacesetter Vijay Singh prepared for an afternoon tee-off.
Justification: The article reports on the progress of professional golfers during a major sports tournament.
Label: Sports
```

Now classify this article:
Text: {text}
Label:"""

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name='gemma-3-12b-it',
    generation_config={
        'temperature': 0.0,
        'max_output_tokens': 100
    }
)

def classify_text(text):
    try:
        prompt = PROMPT_TEMPLATE.format(text=text)

        response = model.generate_content(contents=[prompt])
        print(response.text)

        usage_metadata = response.usage_metadata
        input_token = usage_metadata.prompt_token_count
        output_token = model.count_tokens(response.text).total_tokens

        predicted_label_str = response.text.split('Label:')[-1].strip()
        if predicted_label_str is None or predicted_label_str == '':
            raise Exception("Error: Empty response returned from model.")

        predicted_label_str

        return predicted_label_str, input_token, output_token

    except Exception as e:
        print(f"Error classifying text: {e}")
        return None, 0, 0


labels = {"World": 0, "Sports": 1, "Businesses": 2, "Sci/Tech": 3}
predicted_labels = []
actual_labels = []
request_count = 0
total_input_tokens = 0
total_output_tokens = 0

for article in articles:
    if request_count == 29:
        time.sleep(60)      # Pause for 60 seconds after every 29 requests to prevent rate limiting
        request_count = 0
    request_count += 1

    predicted_label, input_tokens, output_tokens = classify_text(article['text'])
    actual_label = article['label']

    predicted_labels.append(labels[predicted_label])
    actual_labels.append(actual_label)

    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    print(f"Article: {article['text']}")
    print(f"Classification: {predicted_label}")
    print(f"Actual label: {list(labels.keys())[list(labels.values()).index(actual_label)]}\n")

precision, recall, F1, _ = precision_recall_fscore_support(np.array(actual_labels), np.array(predicted_labels), average='macro')
conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=list(labels.values()), normalize='true')
conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(labels.keys()))

#Token usage
avg_input_tokens = total_input_tokens / inference_count
avg_output_tokens = total_output_tokens / inference_count
avg_total_tokens = (total_input_tokens + total_output_tokens) / inference_count

print("--TOKEN USAGE--")
print(f"Average input tokens used per inference: {avg_input_tokens}")
print(f"Average output tokens generated per inference: {avg_output_tokens}")
print(f"Average total tokens used per inference: {avg_total_tokens}\n")

print("--EVALUATION METRICS--")
print(f"Accuracy: {accuracy_score(actual_labels, predicted_labels)}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {F1}\n")

i = 1
while os.path.exists(f"stats/cm_{i}.png"):
    i += 1

conf_matrix_display.plot()
plt.savefig(f"stats/cm_{i}.png", dpi=300, bbox_inches='tight')
plt.show()


with open('stats/classification_report.txt', 'a') as f:
    f.write(f"-----------------------TEST_{i}-----------------------\n\n")

    f.write(f"Prompt: {PROMPT_TEMPLATE}\n\n")

    f.write("--TOKEN USAGE--\n")
    f.write(f"Average input tokens used per inference: {avg_input_tokens}\n")
    f.write(f"Average output tokens generated per inference: {avg_output_tokens}\n")
    f.write(f"Average total tokens used per inference: {avg_total_tokens}\n\n")

    f.write("--EVALUATION METRICS--\n")
    f.write(f"Accuracy: {accuracy_score(actual_labels, predicted_labels)}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {F1}\n\n")

    f.write("--CONFUSION MATRIX--\n" + str(conf_matrix) + "\n\n")