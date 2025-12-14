import pickle
from pathlib import Path
from typing import cast

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

__version__ = '0.1.0'
# __file__ Ubicacion del archivo actual
# BASE_DIR Carpeta padre del archivo
# Esto se hace porque las rutas del docker container pueden variar
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f'{BASE_DIR}/sentiment-analysis-pipeline-{__version__}.pkl', "rb") as f:
    # Use a cast so static analyzers know the expected type of `pipeline`.
    pipeline = cast(Pipeline, pickle.load(f))

def predict_sentiment(text: str) -> tuple[str, float]:
    """Predict the sentiment of a given text.

    Args:
        text (str): The input text to analyze.

    Returns:
        tuple[str, float]: A tuple containing the predicted sentiment label and the probability of the text being positive or negative.
    """
    
    text_prep = cast(Pipeline, pipeline.named_steps['text_prep'])
    clean_text = text_prep.transform([text])

    model = cast(LogisticRegression, pipeline.named_steps['model'])
    prediction = model.predict(clean_text)
    proba = model.predict_proba(clean_text)
    
    if prediction[0] == 0:
        return "Negative", proba[0][0]
    else:
        return "Positive", proba[0][1]
    
print(predict_sentiment("Este produto é ótimo! Adorei a qualidade e o preço foi justo."))  # Expected: ("Positive", probability)

# Me salio el mismo error de que no se encuentra el modulo utils (elimine los imports que habian al inicio, cuando los corria en local no habia problema)
# Quizas haya que restaurar los imports o que el docker tenga un archivo launch.json
# Igual gpt me dio una solucion
# Descansa!!! Te lo ganaste :)))))))