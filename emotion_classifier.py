# emotion_classifier.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Datu ielāde
print("Ielādējam datus...")
train_df = pd.read_csv('train_ekman.tsv', sep='\t', header=None, names=['text', 'emotion'])
dev_df = pd.read_csv('dev_ekman.tsv', sep='\t', header=None, names=['text', 'emotion'])
test_df = pd.read_csv('test_ekman.tsv', sep='\t', header=None, names=['text', 'emotion'])

print(f"Treniņa dati: {len(train_df)} ieraksti")
print(f"Validācijas dati: {len(dev_df)} ieraksti")
print(f"Testa dati: {len(test_df)} ieraksti")

# 1.1. Iztīrīt tukšos tekstus
print("\nTīrām tukšos tekstus...")
train_df = train_df.dropna(subset=['text'])  # Noņemt rindas ar tukšiem tekstiem
train_df['text'] = train_df['text'].fillna('')  # Aizpildīt atlikušos tukšumus ar tukšu virkni

dev_df = dev_df.dropna(subset=['text'])
dev_df['text'] = dev_df['text'].fillna('')

test_df = test_df.dropna(subset=['text'])
test_df['text'] = test_df['text'].fillna('')

print(f"Pēc tīrīšanas - Treniņa dati: {len(train_df)} ieraksti")
print(f"Pēc tīrīšanas - Validācijas dati: {len(dev_df)} ieraksti")
print(f"Pēc tīrīšanas - Testa dati: {len(test_df)} ieraksti")

# 2. Datu vektoruizācija
print("\nPārvēršam tekstus par vektoriem...")
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['emotion']

X_dev = vectorizer.transform(dev_df['text'])
y_dev = dev_df['emotion']

X_test = vectorizer.transform(test_df['text'])
y_test = test_df['emotion']

print(f"Vārdnīcas izmērs: {len(vectorizer.vocabulary_)} unikāli vārdi")

# 3. Modeļa apmācība
print("\nApmācam Naïve Bayes klasifikatoru...")
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
print("Modelis veiksmīgi apmācīts!")

# ... (tālākā koda daļa paliek nemainīga) ...

# 4. Validācija uz attīstības datiem
print("\nNovērtējam uz validācijas datiem...")
y_dev_pred = classifier.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"Precizitāte uz validācijas datiem: {dev_accuracy:.2%}")

# 5. Testēšana uz testa datiem
print("\nNovērtējam uz testa datiem...")
y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Precizitāte uz testa datiem: {test_accuracy:.2%}")

# 6. Detalizēta veiktspējas analīze
print("\nDetalizēta klasifikācijas atskaite (testa dati):")
print(classification_report(y_test, y_test_pred))

# 7. Emociju prognožu vizualizācija
print("\nSaglabājam rezultātu vizualizāciju...")
emotions = sorted(y_test.unique())
prediction_counts = pd.Series(y_test_pred).value_counts().reindex(emotions).fillna(0)

plt.figure(figsize=(12, 6))
bars = plt.bar(prediction_counts.index, prediction_counts.values, color='skyblue')
plt.title('Emociju sadalījums testa datu prognozēs')
plt.xlabel('Emocijas')
plt.ylabel('Prognožu skaits')
plt.xticks(rotation=45)

# Pievienot skaitļus virs kolonnām
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('emotion_predictions.png')
print("Grafiks saglabāts kā emotion_predictions.png")

# 8. Svarīgāko vārdu analīze
print("\nAnalizējam svarīgākos vārdus katrai emocijai...")

def get_top_words(classifier, vectorizer, class_label, n=10):
    feature_names = vectorizer.get_feature_names_out()
    class_index = list(classifier.classes_).index(class_label)
    class_probabilities = classifier.feature_log_prob_[class_index]
    top_indices = class_probabilities.argsort()[-n:][::-1]
    return [(feature_names[i], class_probabilities[i]) for i in top_indices]

print("\nTop 10 vārdi katrai emocijai:")
for emotion in emotions:
    top_words = get_top_words(classifier, vectorizer, emotion)
    print(f"\n{emotion.upper()}:")
    for word, score in top_words:
        print(f"  {word}: {score:.4f}")

# 9. Saglabā rezultātus
results_df = test_df.copy()
results_df['predicted_emotion'] = y_test_pred
results_df['is_correct'] = results_df['emotion'] == results_df['predicted_emotion']
results_df.to_csv('classification_results.csv', index=False)
print("\nPilnie klasifikācijas rezultāti saglabāti failā: classification_results.csv")

# 10. Neveiksmīgo prognožu analīze
incorrect = results_df[~results_df['is_correct']]
print(f"\nKopējais pareizo prognožu procents: {test_accuracy:.2%}")
print(f"Neveiksmīgo prognožu skaits: {len(incorrect)}")
print("\nParaugs no 5 neveiksmīgām prognozēm:")
print(incorrect[['text', 'emotion', 'predicted_emotion']].head(5))

print("\nViss pabeigts! Klasifikators veiksmīgi izveidots un novērtēts.")