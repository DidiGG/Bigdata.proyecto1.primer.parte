import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
datos_videos = r'C:\Users\didie\OneDrive\Documents\GitHub\Bigdata.proyecto1.primer.parte\data\MXvideos.csv'

df = pd.read_csv(datos_videos, encoding="latin-1")
print(df.head())



# Seleccionar la columna de títulos
titles = df['title'].dropna().tolist()  

# Aplicar TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')  # Se eliminan stopwords en inglés
tfidf_matrix = vectorizer.fit_transform(titles)  # Convertir títulos en vectores TF-IDF

# Elegir un título y calcular similitudes
titulo_consulta = "El viaje de chihiro" 
query_vector = vectorizer.transform([titulo_consulta])  # 
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()  

# Obtener los 10 títulos más similares
top_indices = similarities.argsort()[-11:-1][::-1] 
top_titles = [(titles[i], similarities[i]) for i in top_indices]

# Imprimir resultados
print("\nTop 10 títulos más similares a:", titulo_consulta)
for i, (title, score) in enumerate(top_titles):
    print(f"{i+1}. {title} (Similitud: {score:.4f})")
