#Importaciones necesarias
import pandas as pd
from sklearn.cluster import KMeans
import spacy
from goose3 import Goose
from sklearn.cluster import KMeans
import uvicorn
from fastapi import FastAPI,Query
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Función que predice la noticia
def predecir_articulo(noticia):

  g=Goose()
  articulo=g.extract(url=noticia)
  titulo_noticia = articulo.title
  #Extraemos el contenido de la pagina
  nueva_noticia_limpiada = articulo.cleaned_text
  limpiada= nueva_noticia_limpiada

  etikmeans="Km"
  etildag="lda"
  etildask="lda111
  etilsag="ssdd"
  url_screenshot=""
    

  datos_web=[titulo_noticia,etikmeans,etildag,etildask,etilsag,url_screenshot,sobre_texto]
  return datos_web

# Creamos la API
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def index():
  return {"Hola":"Empezamos"}

@app.get('/predict/')
async def predict(url: str = Query(..., min_length=30, max_length=500)):
    result = predecir_articulo(url)
    return {"url": url,"titulo":result[0], "etikmeans": result[1].upper(),"etildag":  result[2].upper(),"etildask":  result[3].upper(),"etilsag":  result[4].upper(),"screenshot": result[5],"sobretexto": result[6]}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
