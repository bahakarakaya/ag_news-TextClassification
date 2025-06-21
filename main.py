import google.generativeai as genai
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import os
from dotenv import load_dotenv


"""TODO: 
[X] SEED ekle
[X] Train yerine test verileri kullan
[ ] Configuration kısmını ayrıca toparla bir yere, model adı vs gibi
[X] Confusion matrix ekle: sklearn.metrics.ConfusionMatrix ,sklearn.ConfusionMatrixDisplay
[X] Token verileri
[ ] Error handling ekle?
[ ] Raporlamayı dosyaya yazdır
"""

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ds = load_dataset("fancyzhx/ag_news")

random.seed(42)     # Her seferinde aynı verilerin seçildiğinen emin olmak için

inference_count = 10 # Toplam kaç tane makale üzerinde inference yapılacak

article_indices = random.sample(range(len(ds['test'])), inference_count)
articles = [ds['test'][i] for i in article_indices]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name='gemma-3-12b-it',
    generation_config={
        'temperature': 0.5,
        'max_output_tokens': 10
    }
)

def classify_text(text):
    try:
        prompt = f"""
        Classify the following text into one of these categories:
        - 0: World
        - 1: Sports
        - 2: Businesses
        - 3: Sci/Tech
    
        Text: "{text}"
    
        Please provide the category label number only.
        Label:"""

        response = model.generate_content(contents=[prompt])

        usage_metadata = response.usage_metadata
        input_token = usage_metadata.prompt_token_count
        output_token = usage_metadata.candidates_token_count    # 0 döndürüyor

        predicted_label_str = response.text.replace('\n', '').strip()
        if predicted_label_str is None or predicted_label_str == '':
            raise Exception("Error: Empty response returned from model.")

        pred_label = int(predicted_label_str)

        return pred_label, input_token, output_token

    except Exception as e:
        print(f"Error classifying text: {e}")
        return None, 0, 0


labels = {0: "World", 1: "Sports", 2: "Businesses", 3: "Sci/Tech"}
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

    predicted_labels.append(predicted_label)
    actual_labels.append(actual_label)

    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    print(f"Article: {article['text']}")
    print(f"Classification: {labels.get(predicted_label)}")
    print(f"Actual label: {labels.get(actual_label)}\n")

precision, recall, F1, _ = precision_recall_fscore_support(np.array(actual_labels), np.array(predicted_labels), average='macro')
conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=list(labels.keys()), normalize='true')     # satır bazında normalizasyon
conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(labels.values()))

#Token usage
avg_input_tokens = total_input_tokens / inference_count
avg_output_tokens = total_output_tokens / inference_count
avg_total_tokens = (total_input_tokens + total_output_tokens) / inference_count

conf_matrix_display.plot()
plt.show()

print("--TOKEN USAGE--")
print(f"Average input tokens used per inference: {avg_input_tokens}")
print(f"Average output tokens generated per inference: {avg_output_tokens}")
print(f"Average total tokens used per inference: {avg_total_tokens}\n")

print("--EVALUATION METRICS--")
print("Accuracy:", accuracy_score(actual_labels, predicted_labels))
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", F1)
