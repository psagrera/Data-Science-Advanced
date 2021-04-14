---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: mioti_nlp
    language: python
    name: mioti_nlp
---

<img src="mioti.png" style="height: 100px">
<center style="color:#888">Módulo Advanced Data Science<br/>Natural Language Processing</center>

# S1. Challenge. Clasificación multiclase


En este **challenge** vamos a aprender a predecir etiquetas de *posts* de [StackOverflow](https://stackoverflow.com). Técnicamente, es una tarea de clasificación multiclase. Nótese que el lenguaje en el que están escritas las entradas es el **INGLÉS**, con lo que algunos de los pasos serás específicos para dicho idioma.

## Librerías

Haremos uso de las siguientes librerías
- [Numpy](http://www.numpy.org) 
- [Pandas](https://pandas.pydata.org) 
- [scikit-learn](http://scikit-learn.org/stable/index.html)
- [NLTK](http://www.nltk.org) — librería básica para trabajar con texto en Python

aunque si quieres pudes usar spaCy para algunas tareas.


##  Preprocesado


Una de las primeras técnicas que vamos a utilizar para preprocesar textos es la eliminación de las conocidas como **stop words**, es decir, palabras que no aportan mucho significado, pero que son necesarias para que el texto sea legible y siga las normas. Para ello, lo primero es conseguir una lista con las *stop words* del lenguaje requerido.

Una opción para conseguir esta lista de palabras, es usar la librería `nltk`.

```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
```

En el *challenge* tenemos un dataset con títulos de entradas de StackOverflow, debidamente etiquetado (con 100 etiquetas distintas).

```python
from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```

```python
def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data
```

```python
train = read_data('data/train.tsv')
train, validation = train_test_split(train, test_size = .15, random_state = 0)
test = read_data('data/test.tsv')
```

```python
train.head()
```

Como vemos, la columna *title* contiene los títulos de las entradas, y la columna *tags* una lista con las etiquetas de cada entrada, que puede ser un número arbitrario.


Para seguir los convenios, inicializamos `X_train`, `X_val`, `X_test`, `y_train`, `y_val`.

```python
X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values
```

La principal dificultad de trabajar con lenguaje natural es que no está estructurado. Si cojemos el texto y creamos tokens simplemente separando por los espacios, tendremos *tokens* como '3.5?', 'do.', etc. Para evitar esos problemas, es útil preprocesar el texto.

### **Tarea 1 (Preprocesado):**

Implementa la función `text_tokenizer()` y `text_prepare()` siguiendo las instrucciones.

```python
from string import ascii_lowercase

REPLACE_BY_SPACE = '[/(){}\[\]\|@,;]'
GOOD_CHARS = ascii_lowercase+''.join([str(n) for n in range(10)])+' #+_'
STOPWORDS = set(stopwords.words('english'))


def text_tokenizer(text):
    """
    Transforma un texto (str) en una lista de palabras/tokens (list).
    Es importante usar esta función siempre para ser consistentes.
    """
    ## ESCRIBE AQUÍ TU CÓDIGO

    ##


def text_prepare(text):
    """
    Preprocesa el texto inicial:
    1. eliminando espacios al inicio y final, y convirtiéndolo a minúsculas
    2. cambia los caracteres de REPLACE_BY_SPACE por espacios
    3. elimina los caracteres que no estén en GOOD_CHARS
    4. elimina los tokens que sean STOPWORDS
    5. une los tokens de nuevo en una sóla string
    
    text: str
    return: str
    """
    ## ESCRIBE AQUÍ TU CÓDIGO
   
    
    ##
    return text_clean
```

```python
def test_text_prepare():
    examples = ["   SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Respuesta incorrecta para: '%s'" % text_prepare(ex)
    return '¡Tests correctos!'

print(test_text_prepare())
```

Ahora preprocesamos los textos de todos los conjuntos:

```python
X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]
```

```python
X_train[:3]
```

### **Tarea 2 (Cuentas de palabras y etiquetas):** 

Cuénta cuantas veces aparece cada token (palabra) y cada etiqueta en el corpus de entrenamiento. Es decir, crea un diccionario con las cuentas totales de palabras y etiquetas.
 
El resultado deben ser dos diccionarios *tags_counts* y *words_counts* del tipo `{'palabra_o_etiqueta': cuentas}`.

```python
######################################
##ESCRIBE AQUÍ TU CÓDIGO
######################################

# Diccionario con todas las etiquetas del corpus de entrenamiento con sus cuentas
tags_counts = 
# Diccionario con todas las palabras del corpus de entrenamiento con sus cuentas
words_counts = 

##
```

Exploramos las más comunes:

```python
most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]
print(most_common_tags)
print(most_common_words)
```

<!-- #region -->
### Transformando el texto a vectores

Vamos a construir los vectores asociados a cada frase en dos representaciones distintas.


#### Bag of words

Recuerda que para crear la representación de *bag of words*, convertimos cada frase en un vector que cuenta el número de ocurrencias de cada token. Se siguien los pasos:
1. Encuentra los **N** tokens mas comunes del corpus de entrenamiento y se les asigna un índice, este es nuestro **vocabulario**. Creamos un diccionario para convertir de tokens a índices y viceversa.
2. Para cada frase en el corpus, creamos un vector de dimensión **N** y lo inicializamos con ceros.
3. Iteramos sobre los tokens de cada frase, y si el token está en el diccionario, incrementamos en 1 el índice correspondiente del vector.
   
**Tarea 3 (BagOfWords):** 

Contruye la función que transforma un texto en su representación *bag of words*.

Implementa la codificación de *bag of words* en la función `my_bag_of_words()` con un tamaño de diccionario de **N=5000**. Para definir el diccionario, sólo podemos usar el conjunto de entrenamiento, sino tendríamos un *data leaking*.

Primero, contruimos el vocabulario y los diccionarios correspondientes, así como un `set` con las palabras del diccionario.
<!-- #endregion -->

```python
DICT_SIZE = 5000

## ESCRIBE AQUÍ TU CÓDIGO
INDEX_TO_WORDS = 
WORDS_TO_INDEX =
##

ALL_WORDS = WORDS_TO_INDEX.keys()
assert len(ALL_WORDS)==DICT_SIZE
```

```python
def my_bag_of_words(text, words_to_index, dict_size):
    """
    text: str
    words_to_index: dict, diccionario con los índices del vocabulario
    dict_size: int, tamaño del diccionario
    
    return
    result_vector: numpy.array, vector con la representación bag-of-words de `text`
    """
    result_vector = np.zeros(dict_size)

    ### ESCRIBE AQUÍ TU CÓDIGO

    
    ###
    
    return result_vector
```

```python
def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Respuesta incorrecta: '%s'" % ex
    return '!Tests correctos¡'

print(test_my_bag_of_words())
```

Ahora aplicamos la función anterior a todos los datos.

La representación *bag of words* devuelve vectores __*sparse*__ (la mayoría de sus entradas son ceros), con lo que conviene usar estructuras de datos especiales para datos *sparse* para ser eficientes.

Hay muchos [tipos de representación sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html), y `sklearn` sólo trabaja con la representación [csr matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix), que es la que usamos.

```python
from scipy import sparse as sp_sparse
```

```python
X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)
```

#### tf-idf

En vez de hacerlo desde cero, podemos usar la clase [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) de `sklearn`. Como siempre, para entrenar el codificador sólo se puede usar el set de entrenamiento.

Investiga los argumentos que definen `TfidfVectorizer`. Puedes filtrar las palabras muy raras y también las demasiado frecuentes. También permite utilizar combinaciones de palabras como tokens, es decir, n-gramas. Por último, el tokenizador por defecto separa palabras como 'c++' o 'c#' en varios tokens, pero esto no nos interesa, con lo que vamos a indicar que sólo separe por espacios con el parámetro `token_pattern`.

Puedes usar:
* `min_df=5`
* `max_df=0.9`
* `ngram_range=(1,2)` 
* `token_pattern='(\S+)'`

**Tarea 4 (tf-idf):** 

Contruye la función `tfidf_features()` que transforma el corpus en su representación *tf-idf*.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
    """
    X_train, X_val, X_test — samples        
    return TF-IDF vectorized representation of each sample and vocabulary
    """
    ## ESCRIBE AQUÍ TU CÓDIGO
    
    # Create TF-IDF vectorizer with a proper parameters choice
    tfidf_vectorizer = 
    
    # Ajusta tfidf_vectorizer al set de entrenamiento
    # Transforma los sets de train, test, and val
    X_train_tfidf =  
    X_val_tfidf = 
    X_test_tfidf =
    
    ##
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer.vocabulary_


```

```python
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
```

```python
assert 'c++' in tfidf_vocab.keys()
assert 'c#' in tfidf_vocab.keys()
```

### Clasificador multi-clase con sklearn

El resultado de nuestro clasificador puede consistir en varias etiquetas. Lo primero que tenemos que hacer, es convertir los `y` en números, convirtiendo cada uno en un vector de 0's y 1's indicando la presencia de cada una de las etiquetas.

Esto se puede hacer automáticamente con [MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) de `sklear`.

```python
from sklearn.preprocessing import MultiLabelBinarizer
```

```python
mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)
```

```python
print(y_train[0:5])
```

### **Tarea 5 (Entrenamiento):** 

Implementa la función `train_classifier()` que entrena un clasificador dados los datos de entrenamiento. 

Como ya sabes, una clasificación multi-clase con $L$ etiquetas, se puede estudiar como $L$ clasificadores binarios. Esto se puede hacer en el formato *Uno contra todos*, que está implementado en [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html).

Como clasificador base se puede usar [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Es uno de los métodos más simples, pero generalmente funciona bien en tareas de clasificación de texto.

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
```

```python
def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    ######################################
    ### ESCRIBE AQUÍ TU CÓDIGO 
    ###################################### 

    
    ### 
    return model
```

Entrena un modelo para cada una de las features que hemos construido: *bag-of-words* y *tf-idf*. Y luego calcula las predicciones sobre el conjunto de validación, vamos a necesitar las predicciones (labels) y las probabilidades (scores) para poder calcular métricas como la ROC curve.

```python
classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)
```

```python
y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
```

Veamos algún ejemplo:

```python
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))
```

### Evaluación

Para evaluar el modelo de clasificación multi-clase, usaremos las siguientes métricas:

 - [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
 - [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
 - [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
 - [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) 

Estudia el significado de cada de las métricas, teniendo en cuenta que estamos ante un problema multi-clase y no binario. Lee sobre micro/macro/weighted averaging en la documentación de `sklearn`.

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
```

### **Tarea 6 (Evaluación):** 

Implementa la función `print_evaluation_scores()` que calcula e imprime las siguientes métricas:
 - *accuracy*
 - *F1-score macro/micro/weighted*
 - *Precision macro/micro/weighted*
 
Utiliza para ello las implementaciones de estas métricas de `sklearn`.

```python
def print_evaluation_scores(y_val, predicted):
    ######################################
    ### ESCRIBE AQUÍ TU CÓDIGO 
    ######################################
    
```

```python
print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
```

Es interesante mostrar una generalización de la [ROC curve](http://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc) para el caso multi-clase. Utiliza la función `roc_auc()` para ello. Los parámetros de entrada son:
 - y_test : etiquetas correctas (labels)
 - y_score: probabilidades, (score decision function)
 - n_classes: número de clases

```python
from metrics import roc_auc
%matplotlib inline
```

```python
n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)
```

```python
n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)
```

### Extra : Hyper parameter tuning


Una vez hemos entrenado un modelo y lo hemos evaluado, podemos proceder a hacer ajuste de hiperparámetros, para ello usaremos como métrica de validación *F1-score weighted*, **sobre los datos de validación**.

Pasos:
* Compara la calidad de bag of words y TF-IDF y elige uno.
* Investiga cambiando los parámetros de la regularización *L1* y *L2* de la Logistic Regression (e.g. C con valores de 0.1, 1, 10, 100). 

Puedes elegir también otro clasificador base, como Random Forest. O modificar el preprocessing.

Para finalizar, evalua el mejor modelo (sobre la métrica de validación) en el **set de test** para estimar sus métricas.

```python
######################################
### ESCRIBE AQUÍ TU CÓDIGO 
######################################
```

### Extra: Interpretabilidad del modelo


En la práctica es muy importante explorar las features (en este caso las palabras o n-gramas) que tienen pesos más altos en el modelo de regresión logística (en el caso de árboles de decisión también hay feature importance).

Implementa la función `print_words_for_tag()` para encontrarlas. Investiga la documentación de [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) y [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) para saber como acceder a los coeficientes de la regresión.

```python
def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))
    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator. 
    
    ######################################
    ### ESCRIBE AQUÍ TU CÓDIGO 
    ######################################
    
    
    ###
    print('Top palabras positivas:\t{}'.format(', '.join(top_positive_words)))
    print('Top palabras negativas:\t{}\n'.format(', '.join(top_negative_words)))
```

```python
print_words_for_tag(classifier_tfidf, 'python', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
```
