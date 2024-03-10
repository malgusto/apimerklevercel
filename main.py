#Importaciones necesarias
import joblib 
import pandas as pd
from sklearn.cluster import KMeans
import spacy 
from goose3 import Goose
from sklearn.cluster import KMeans
import uvicorn
from fastapi import FastAPI,Query
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import requests
from fastapi.staticfiles import StaticFiles
# import selenium

# from capturescreenshot import *

# Cargando K-Means
vectorizadorKmeans=joblib.load("./modelos/k-means/vectorizer_kmeans")
modeloKmeans=joblib.load('./modelos/k-means/modelo_kmeans')
categoriasKmeans=joblib.load('./modelos/k-means/categorias-k-means')

# Cargando LDA-Gensim
modeloldag = joblib.load('./modelos/lda-gensim/Mejor_modelo_LDA_GENSIM')
categoriasldag = joblib.load('./modelos/lda-gensim/Categorias_mejor_modelo_LDA_GENSIM')
diccionariolda = joblib.load('./modelos/lda-gensim/Diccionario_LDA_GENSIM')

# Cargando LDA-SKL
vectorizerldask=joblib.load('./modelos/lda-sk/vectorizerldask')
modeloldask = joblib.load('./modelos/lda-sk/Mejor_modelo_LDA_SKLEARN')
categoriasldask = joblib.load('./modelos/lda-sk/Categorias_mejor_modelo_LDA_SKLEARN')

#Cargando LSA-Gensim
modelolsag = joblib.load('./modelos/lsa-gensim/Mejor_modelo_LSA_GENSIM')
categoriaslsag = joblib.load('./modelos/lsa-gensim/Categorias_mejor_modelo_LSA_GENSIM')
diccionariolsag = joblib.load('./modelos/lsa-gensim/Diccionario_LSA_GENSIM')


# app.mount("/capturas", StaticFiles(directory="capturas"), name="capturas")


# Función que predice la noticia
def predecir_articulo(noticia):

  nlp = spacy.load("es_core_news_sm")
  
  def limpiar(doc):
      doc_procesado = []
      for token in nlp(doc.lower()):
          # Filtrar los tokens
          if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_space and not token.is_punct and not token.is_stop and not token.like_num and not token.like_email:
              # Si es un sustantivo, verbo, adjetivo, adverbio o conjunción y no signos de puntuación, ni es un número o email, añadir el lexema de la palabra al documento procesado
              doc_procesado.append(token.lemma_)
      return doc_procesado
    
  g=Goose()
  articulo=g.extract(url=noticia)
  titulo_noticia = articulo.title
  #Extraemos el contenido de la pagina
  nueva_noticia_limpiada = articulo.cleaned_text
  limpiada= limpiar(nueva_noticia_limpiada.lower())

  #Datos a sacar sobre el articulo
  numparrafos = len(articulo.cleaned_text.split("\n"))
  numeropalabras = len(articulo.cleaned_text.split())
  numerofrases = len(articulo.cleaned_text.split("."))
  
  sobre_texto=f"El artículo consta de un total de {numparrafos} párrafos, {numeropalabras} palabras y {numerofrases} frases."  
  

  #############
  # K-Means
  #############    
  # Vectorizar nueva noticia
  limpiada_kmeans=" ".join(limpiada)
  nuevas_noticias_vectorizadas=vectorizadorKmeans.transform([limpiada_kmeans])
  # Predecir
  cluster_predicho = modeloKmeans.predict(nuevas_noticias_vectorizadas)
  etikmeans=categoriasKmeans[cluster_predicho[0]]
  
  #############
  # LDA-Gensim
  #############    
  corpus = diccionariolda.doc2bow(limpiada)
  # Categoriza la nueva noticia usando el modelo LDA
  categorias_noticia = modeloldag.get_document_topics(corpus)
  tema_principal = max(categorias_noticia, key=lambda x: x[1])[0]
  if tema_principal > 0:
     etildag = categoriasldag[tema_principal - 1]
  else:
     etildag = categoriasldag[tema_principal]

  
  #############
  # LDA-SKL
  #############    
  noticia_procesadaldask = " ".join(limpiada)
  noticia_vectorizada = vectorizerldask.transform([noticia_procesadaldask])

  # Transformación del artículo con el modelo LDA
  distribucion_topicos = modeloldask.transform(noticia_vectorizada)

  # Identificación del tópico dominante
  topico_dominante = distribucion_topicos.argmax(axis=1)[0]
  etildask=categoriasldask[topico_dominante]
  
  #############
  # LSA-Gensim
  #############
  # Convierte el texto preprocesado en el formato del corpus
  corpus = diccionariolsag.doc2bow(limpiada)

  # Categoriza la nueva noticia usando el modelo LDA
  lsa_vector = modelolsag[corpus]
  ''' Enlazarlo con las categorías '''
  topico_dominante = max(lsa_vector, key=lambda x: x[1])[0]

  etilsag=categoriaslsag[topico_dominante]
  
  
    
  # url_screenshot="https://apimerklevercel-production.up.railway.app/"+captscreehshot(noticia)
  url_screenshot="1"
    

  datos_web=[titulo_noticia,etikmeans,etildag,etildask,etilsag,url_screenshot,sobre_texto]
  return datos_web

# Creamos la API
app = FastAPI()

@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.get('/')
async def index():
  return {"Hola":"Empezamos"}

@app.get('/predict/')
async def predict(url: str = Query(..., min_length=30, max_length=500)):
    result = predecir_articulo(url)
    return {"url": url,"titulo":result[0], "etikmeans": result[1].upper(),"etildag":  result[2].upper(),"etildask":  result[3].upper(),"etilsag":  result[4].upper(),"screenshot": "1","sobretexto": result[6]}
    # return {"url": url,"titulo":result[0], "etikmeans": result[1].upper(),"etildag":  result[2].upper(),"etildask":  result[3].upper(),"etilsag":  result[4].upper(),"screenshot": result[5],"sobretexto": result[6]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
