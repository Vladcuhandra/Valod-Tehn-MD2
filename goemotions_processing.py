import json
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests 

# Lejupielādējam nepieciešamos NLTK resursus
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ========================================
# 0. Datu failu lejupielāde
# ========================================

def download_file(url, filename):
    """Lejupielādē failu, ja tas neeksistē"""
    if os.path.exists(filename):
        print(f"Fails {filename} jau eksistē.")
        return
    
    # Izveidojam direktoriju, ja tāda nav
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        print(f"Lejupielādē {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Lejupielādēts: {filename}")
    except Exception as e:
        print(f"Kļūda lejupielādējot {filename}: {e}")
        raise

# Lejupielādējam nepieciešamos failus
base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"

files_to_download = {
    'data/train.tsv': base_url + 'train.tsv',
    'data/dev.tsv': base_url + 'dev.tsv',
    'data/test.tsv': base_url + 'test.tsv',
    'data/ekman_mapping.json': base_url + 'ekman_mapping.json',
    'data/emotions.txt': base_url + 'emotions.txt'
}

for local_path, url in files_to_download.items():
    download_file(url, local_path)

# ========================================
# 1. Datu ielāde un kartēšanas sagatavošana
# ========================================

# Ielādējam emociju nosaukumus
with open('data/emotions.txt', 'r') as f:
    emotions = [line.strip() for line in f]

# Ielādēja Ekman kartēšanu
with open('data/ekman_mapping.json') as f:
    ekman_mapping = json.load(f)

# Izveidojam apgriezto kartēšanu: emocija -> Ekman kategorija
reverse_ekman_mapping = {}
for ekman_category, emotion_list in ekman_mapping.items():
    for emotion in emotion_list:
        reverse_ekman_mapping[emotion] = ekman_category

# Izveidojam kartēšanu no emocijas ID uz Ekman kategoriju
id_to_ekman = {}
for idx, emotion_name in enumerate(emotions):
    if emotion_name in reverse_ekman_mapping:
        id_to_ekman[idx] = reverse_ekman_mapping[emotion_name]
    elif emotion_name == "neutral":
        id_to_ekman[idx] = "neutral"
    else:
        id_to_ekman[idx] = "neutral"

print("Kartēšanas piemērs:")
for i in range(5):
    print(f"ID {i} ({emotions[i]}) -> {id_to_ekman[i]}")

# ========================================
# 2. Teksta priekšapstrādes funkcijas
# ========================================

def clean_text(text):
    """Attīra tekstu no nevajadzīgām rakstzīmēm un trokšņiem"""
    if not isinstance(text, str):
        return ""
    
    # Noņemam URL, lietotājvārdus, hashtagus
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Noņemam skaitļus
    text = re.sub(r'\d+', '', text)
    
    # Aizstājam speciālās rakstzīmes ar atstarpēm
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Aizstājam vairākus atstarpju rakstzīmes ar vienu atstarpi
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip().lower()

def lemmatize_tokens(tokens):
    """Lemmatizē tokenus (normalizē vārdus uz pamatformām)"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    """Pilna teksta priekšapstrādes pipeline"""
    # 1. Attīrīšana no trokšņiem
    text = clean_text(text)
    
    # 2. Tokenizācija (sadalīšana vārdos)
    tokens = word_tokenize(text)
    
    # 3. Lemmatizācija
    tokens = lemmatize_tokens(tokens)
    
    return " ".join(tokens)

# ========================================
# 3. Datu apstrādes funkcijas
# ========================================

def process_emotion_file(input_path, output_path):
    """
    Apstrādā vienu emociju datu failu:
    - Filtrē ierakstus ar vienu emociju
    - Pārveido emocijas uz Ekman sistēmu
    - Pievieno priekšapstrādātu tekstu
    """
    print(f"\nApstrādājam: {input_path}")
    
    # Ielādējam datus (tikai pirmās divas kolonnas)
    df = pd.read_csv(input_path, sep='\t', header=None, usecols=[0, 1])
    df.columns = ['text', 'emotion_ids']
    print(f"  Ielādēti {len(df)} ieraksti")
    
    # Konvertējam emociju ID (no "0,1" uz [0,1])
    df['emotion_ids'] = df['emotion_ids'].apply(
        lambda x: [int(i) for i in str(x).split(',')] if pd.notnull(x) else [])
    
    # Filtrējam ierakstus ar tieši vienu emociju
    single_emotion_df = df[df['emotion_ids'].apply(len) == 1]
    print(f"  Ieraksti ar vienu emociju: {len(single_emotion_df)}")
    
    # Pārveidojam uz Ekman kategorijām
    single_emotion_df['ekman_emotion'] = single_emotion_df['emotion_ids'].apply(
        lambda ids: id_to_ekman.get(ids[0], "neutral") if ids else "neutral")
    
    # Pievienojam priekšapstrādātu tekstu
    print("  Veicam teksta priekšapstrādi...")
    single_emotion_df['processed_text'] = single_emotion_df['text'].apply(preprocess_text)
    
    # Saglabājam jauno failu
    single_emotion_df[['processed_text', 'ekman_emotion']].to_csv(
        output_path, sep='\t', index=False, header=False
    )
    print(f"  Saglabāts {output_path} ar {len(single_emotion_df)} ierakstiem")
    
    return single_emotion_df

# Apstrādājam visus datu kopas failus
train_df = process_emotion_file('data/train.tsv', 'train_ekman.tsv')
dev_df = process_emotion_file('data/dev.tsv', 'dev_ekman.tsv')
test_df = process_emotion_file('data/test.tsv', 'test_ekman.tsv')

# ========================================
# 4. Tokenizācija un biežumsaraksts
# ========================================

print("\nVeidojam tokenu biežumsarakstu...")

# Apvienojam visus apstrādātos tekstus
all_text = pd.concat([
    train_df['processed_text'], 
    dev_df['processed_text'], 
    test_df['processed_text']
]).tolist()

# Tokenizējam un saskaitīt tokenus
all_tokens = []
for text in all_text:
    if text:  # Izvairīties no tukšiem tekstiem
        tokens = word_tokenize(text)
        all_tokens.extend(tokens)

# Izveidojam biežuma vārdnīcu
token_freq = Counter(all_tokens)
print("\nBiežākās 10 tekstvienības pirms apcirpšanas:")
print(token_freq.most_common(10))

# ========================================
# 5. Datu analīze
# ========================================

def plot_emotion_distribution(df, title):
    """Vizualizē emociju sadalījumu un saglabā grafiku"""
    counts = df['ekman_emotion'].value_counts()
    print(f"\n{title}:")
    print(counts)
    
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel('Emocijas')
    plt.ylabel('Skaits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Saglabāts grafiks: {title.replace(' ', '_')}.png")

# Vizualizējam emociju sadalījumu
plot_emotion_distribution(train_df, 'Emociju sadalījums treniņa datos')
plot_emotion_distribution(dev_df, 'Emociju sadalījums validācijas datos')
plot_emotion_distribution(test_df, 'Emociju sadalījums testa datos')

# Aprēķinām tokenu statistiku
print(f"\nKopējais tokenu skaits: {len(all_tokens)}")
print(f"Unikālo tokenu skaits: {len(token_freq)}")
avg_length = np.mean([len(word_tokenize(t)) for t in all_text if t])
print(f"Vidējais teikuma garums: {avg_length:.2f} tokeni")

# ========================================
# 6. Leksikona apcirpšana
# ========================================

def filter_infrequent_tokens(tokens, min_freq=5):
    """Noņem retos tokenus (ar biežumu < min_freq)"""
    freq = Counter(tokens)
    return [t for t in tokens if freq[t] >= min_freq]

print("\nVeicam leksikona apcirpšanu...")
filtered_tokens = filter_infrequent_tokens(all_tokens, min_freq=5)
filtered_freq = Counter(filtered_tokens)

print(f"Unikālo tokenu skaits pēc apcirpšanas: {len(filtered_freq)}")
print("Biežākās 10 tekstvienības pēc apcirpšanas:")
print(filtered_freq.most_common(10))

# Saglabājam apgriezto vārdnīcu
with open('filtered_vocab.txt', 'w') as f:
    for word, count in filtered_freq.most_common():
        f.write(f"{word}\t{count}\n")

# ========================================
# 7. Rezultātu paraugi
# ========================================

print("\nRezultātu paraugi:")
print("Train paraugs:")
print(train_df[['text', 'processed_text', 'ekman_emotion']].head(3))
print("\nBiežākie tokeni pēc apstrādes:")
print(filtered_freq.most_common(10))

print("\nApstrāde pabeigta veiksmīgi!")
print("Izveidotie faili:")
print("- train_ekman.tsv: Treniņa dati ar Ekman emocijām un apstrādātiem tekstiem")
print("- dev_ekman.tsv: Validācijas dati")
print("- test_ekman.tsv: Testa dati")
print("- filtered_vocab.txt: Apgrieztā vārdnīca ar biežumiem")
print("- Emociju sadalījuma grafiki")