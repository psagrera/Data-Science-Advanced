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

# S1. Herramientas clásicas de NLP

```python
import sys
!{sys.executable} -m pip install -r ../requirements.txt
```

```python
import nltk
import spacy
import sklearn
```

<!-- #region -->
En este notebook vamos a repasar herramientas clásicas (sin *deep learning*) que pueden sernos útiles para trabajar con texto, y que nos van a permitir dar solución a muchas de las peculiaridades del lenguaje natural.


## Simplificación de los datos

Uno de los principales problemas de las representaciones clásicas de texto es la dimensionalidad, ya que nos basamos en un diccionario, que fácilmente puede tener $10-100$ K tokens. Así, surge un problema técnico que nos lleva a intentar reducir la dimensionalidad de los vectores utilizados. 

Podemos ver un texto como una unidad de comunicación humana bastante compleja. Un documento, hasta el más pequeño, está hecho no sólo de palabras, sino de un sinnúmero de relaciones semánticas que solo pueden ser descodificadas por quienes dominan ciertos códigos. En fin, que son un desastre y un dolor de cabeza para quienes están acostumbrados a la información estructurada (e.g., en tablas o esquemas).

Extraer automáticamente información relevante de los textos es una tarea trivial para un ser humano, pero un verdadero reto para una máquina. Muchas veces no nos interesa conocer todos los significados de un texto, sino solamente algunos pertinentes para realizar una tarea. Aunque las computadoras (aún) no entienden el lenguaje natural, son muy competentes leyendo superficialmente grandes cantidades de texto en segundos.

Una buena técnica para obtener la información relevante de un texto consiste en eliminar los elementos que puedan ser irrelevantes, y resaltar más lo que de verdad contengan información. Las siguientes herramientas tienen el mismo objetivo, y es el de atacar el problema de la dimensionalidad, intentando simplificar a las máquinas el texto libre que reciben.

### tokenización

Vamos a eliminar las palabras que tienen poco interés para nosotros. El primer paso es delimitar las palabras del texto, y convertir esas palabras en elementos de una lista, los **tokens**. Este procedimiento es conocido como tokenización. Este proceso también está repleto de ambigüedades:
* hispano-romano, astur-leonés
* aren't, o'neill
* whitespace, white space, 


Hay diversas librerías que nos facilitan la tokenización, y este proceso se puede hacer usando reglas, o mediante modelos estadísticos. Dos librerías que facilitan esta tarea son `spacy` y `nltk`. O se puede hacer directamente en Python si la regla es sencilla, incluso usando expresiones regulares.
<!-- #endregion -->

```python
texto = '''El producto cuesta 3€. Este procedimiento es conocido como tokenización. Vamos a implementar esto en Python. Lo ha hecho la empresa TEST S.A. '''
```

```python
texto.split()
```

La librería `nltk` tiene un amplio rango de [tokenizers](https://www.nltk.org/api/nltk.tokenize.html), tanto para palabras, como para frases y sílabas. Incluso hay uno específico para tweets `nltk.tokenize.TweetTokenizer()`.

```python
tokenizer = nltk.tokenize.ToktokTokenizer()
tokenizer.tokenize(texto)
```

```python
tokenizer = nltk.tokenize.NLTKWordTokenizer()
tokenizer.tokenize(texto)
```

El recomendado por `nltk`, y que permite elegir el idioma, es `nltk.tokenize.word_tokenize()`:

```python
try:
    nltk.tokenize.word_tokenize('test', language='spanish')
except LookupError:
    # descarga los archivos necesarios
    nltk.download('punkt')
```

```python
tokenize = lambda x: nltk.tokenize.word_tokenize(x, language='spanish')
tokenize(texto)
```

En `spacy`, además de la librería, debes descargar el modelo de la lengua que vas a utilizar. 

Para descargar el modelo del español, por ejemplo, escribe en tu terminal lo siguiente: 

`python -m spacy download es`


```python
try:
     # Crea un objeto de spacy tipo nlp, que tokeniza directamente
    nlp = spacy.load('es_core_news_sm')
except:
    import sys
    !{sys.executable} -m spacy download es_core_news_sm
    nlp = spacy.load('es_core_news_sm')
```

Si aplicamos el modelo de lenguaje `nlp` al texto, directamente lo tokeniza, pero además aplica PoS tagging y NER por defecto:

```python
doc = nlp(texto) 
tokens = [t.text for t in doc] # Crea una lista con las palabras del texto
tokens
```

Si lo **único que queremos es tokenizar, es mucho más eficiente** decirle a spacy que sólo haga eso, o crear un tokenizador a partir de `nlp`. Se puede hacer de muchas formas:

```python
tokenizer = nlp.tokenizer
tokens = [t.text for t in tokenizer(texto)] # Crea una lista con las palabras del texto
print(tokens)
```

```python
with nlp.disable_pipes(nlp.pipe_names): # desactivamos todo
    doc = nlp(texto)
print([tok.text for tok in doc])
```

```python
doc = list(nlp.pipe([texto],disable=nlp.pipe_names))[0]
print([tok.text for tok in doc])
```

<!-- #region -->

### Stop words

Son palabras excesivamente comunes que no añaden significado a un texto, debido al gran tamaño de los vocabularios y por ende, de los vectores que codificaban texto (*sparsity*), se eliminaban directamente. La eliminación de las conocidas como **stop words** no es obligatoria, y su utilización ha disminuido recientemente. Queda muy bien explicado en el siguiente fragmento de [Intro to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html):


*Some extremely common words which would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called stop words.*

*The general trend in IR systems over time has been from standard use of quite large stop lists (200-300 terms) to very small stop lists (7-12 terms) to no stop list whatsoever. Web search engines generally do not use stop lists.*

Las listas de *stop words* no son únicas, y diferentes librerías tienen listas distintas, como veremos a continuación. Además, lo que es y no es una stop word **depende del problema**, ya que una misma palabra puede tener importancia en un contexto, pero no en otro.

Además de las stop words, también es común eliminar la **puntuación** y/o **símbolos** (@,$,€,#) que creemos que no nos van a aportar significado.
<!-- #endregion -->

`sklearn` también tiene su lista de stop words en inglés:

```python
sk_stopwords = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
print(sorted(list(sk_stopwords))[:20])
```

Vamos ahora a las librerías especializadas en NLP, como `nltk`:

```python
from nltk.corpus import stopwords as nltk_stopwords
print(sorted(nltk_stopwords.words('english'))[:20])
```

`nltk` tiene listas de stop words en varios idiomas:

```python
stopwords_sp_nltk = sorted(nltk_stopwords.words('spanish'))
print(stopwords_sp_nltk[:20])
```

Podemos hacer lo mismo con `spacy`:

```python
print(sorted(spacy.lang.es.stop_words.STOP_WORDS)[:20])
```

**Ejercicio:** 

Comprueba si palabras como `['nada','no','mal','bien','bueno','ninguno']` son *stop words*. ¿Tiene esto sentido?

```python


















```

```python
stopwords_sp = sorted(spacy.lang.es.stop_words.STOP_WORDS)
for word in ['nada','no','mal','bien','bueno','ninguno']:
    for lib, sw in zip(['nltk','spacy'],[stopwords_sp_nltk, stopwords_sp]):
        print(f"'{word}' is stop word in {lib}: {word in sw}")
```

**Ejercicio:** Calcula los tokens que estén en la lista de **stop words** de `spacy`, pero no en la de `nltk`. También la longitud de cada conjunto. Por ejemplo, para el idioma español.

```python

```

**Ejercicio:** ¿En qué tareas la eliminación de *stop words* no es viable?

```python



















```

Algunas tareas donde no es aplicable porque se pierde información esencial serían:

* Traducción automática
* Modelado del lenguage (Language Modeling)
* Resumen automático de textos
* Sistemas conversacionales o de Pregunta Respuesta


### Stemming y Lemmatization

<!-- #region -->
En general, podemos observar que hay palabras diferentes que representan significados parecidos. En español, por ejemplo, sabemos que *canto, cantas, canta, cantamos, cantáis, cantan* son distintas formas (conjugaciones) de un mismo verbo (*cantar*). Y que *niña, niño, niñita, niños, niñotes, y otras más*, son distintas formas del vocablo *niño*. Así que sería genial poder obviar las diferencias y juntar todas estas variantes en un mismo token.
Stemming y Lemmatization son dos transformaciones parecidas, que nos devuelven el lema o raiz de la palabra.

* **Lematización**: relaciona una palabra flexionada o derivada con su forma **canónica o lema**. Y un lema no es otra cosa que la **forma que tienen las palabras cuando las buscas en el diccionario**. La lematización tiene dos costes. Primero, es un proceso que consume recursos (sobre todo tiempo). Segundo, suele ser probabilística, así que en algunos casos obtendremos resultados inesperados.


* **Radicalización** o **stemming**: procedimiento de convertir palabras en raíces. Estas raíces son la parte invariable de palabras relacionadas sobre todo por su forma. De cierta manera se parece a la lematización, pero los resultados (las raíces) no tienen por qué ser palabras de un idioma. Por ejemplo, el algoritmo de stemming puede decidir que la raíz de *amamos* no es *am-* sino *amam-*. El stemming es mucho más rápido, ya que es un procedimiento heurístico que recorta el final de las palabras.

"Stemming is the poor-man’s lemmatization." (Noah Smith, 2011)

**Es importante notar que ambos procesos dependen altamente del idioma.**
<!-- #endregion -->

Para lematizar es importante la etiqueta *Part of Speech* (POS), es decir, si la palabra es un sustantivo, verbo, etc. En el caso de `nltk`, hay que indicar en cada caso esta etiqueta:

```python
# descargamos el lemmatizer wordnet en caso de que sea necesario
try:
    wnl = nltk.stem.WordNetLemmatizer()
    wnl.lemmatize("test", pos='n')
except LookupError:
    nltk.download('wordnet')
```

```python
word_list = ['feet', 'foot', 'foots', 'footing']
```

```python
wnl = nltk.stem.WordNetLemmatizer()
[wnl.lemmatize(word, pos='n') for word in word_list]
```

Si queremos un lematizador en español, tenemos que ir a `spacy`, que además de manera automática aplica un **PoS-tagger** y hace muy sencillo obtener el *lema*:

```python
nlp = spacy.load('es_core_news_sm')
```

```python
doc = nlp(texto)
lemmas = [tok.lemma_.lower() for tok in doc]
for tok in doc:
    print(tok.text, '->', tok.lemma_.lower())
```

<!-- #region -->
Aunque te parezca sorprendente, `spacy` no contiene ninguna función para stemming, ya que se basa únicamente en la lematización. Por lo tanto, en esta sección, utilizaremos únicamente `NLTK`.


Hay dos tipos de stemmers en NLTK: [Porter Stemmer](https://tartarus.org/martin/PorterStemmer/) and [Snowball stemmers](https://tartarus.org/martin/PorterStemmer/). Cada uno de ellos se ha implementado siguiendo algoritmos distintos. Snowball stemmer es una versión ligeramente mejorada del Porter stemmer y generalmente se prefiere sobre este último, además, **sólo Snowball stemmer funciona para idiomas distintos al inglés**.
<!-- #endregion -->

```python
sp_snowball = nltk.SnowballStemmer('spanish')
tokens = texto.split()  # crear una lista de tokens
stems = [sp_snowball.stem(token) for token in tokens]
for w, s in zip(tokens, stems):
    print(w, '->', s)
```

**Ejercio:** Prueba a lematizar y radicalizar las siguientes palabras:

* Inglés: fly, flies, flying
* Español: universo, universidad, universal

```python

```

### spaCy vs NLTK

La librería spaCy es una de las más populares en NLP junto con NLTK. La diferencia básica entre las dos es el hecho de que **NLTK contiene una amplia variedad de algoritmos** para resolver un problema, mientras que **spaCy contiene solo uno, el mejor**.

NLTK se lanzó en 2001, mientras que spaCy es relativamente nuevo y se desarrolló en 2015. En este curso sobre NLP, trataremos principalmente de spaCy, debido a que en general es **state of the art**. Sin embargo, también usaremos NLTK cuando sea más fácil realizar una tarea que con spaCy.

**Si puedes hacerlo en spaCy, mejor usar spacy.** Además, la documentación de spaCy es mucho más intuitiva.


### ¿Cuándo usar estos métodos?

<!-- #region -->
Estas técnicas fueron consideradas técnicas estándar durante mucho tiempo, pero a menudo pueden **dañar** el rendimiento **si se utiliza deep learning**. El stemming, la lematización y la eliminación *stop words* implican una pérdida de información en muchos casos.

Sin embargo, aún pueden ser útiles cuando se trabaja con modelos más simples. A grandes rasgos, estamos simplificando los datos, lo que es una técnica de **regularización**, es decir, evitamos el **overfitting** a features que sólo nos generan ruido. 

Como siempre en el Machine Learning, no es fácil situar la línea entre el ruido y la señal. Sólo la experiencia, y el sobre todo tiempo para validar entre distintas opciones deben ser la guía que nos indique qué usar en el modelo final.


<img src="resources/skomoroch.png" alt="" style="width: 65%"/>
<!-- #endregion -->

<!-- #region -->
## Vectorizando texto

Hasta ahora hemos estudiado formas de reducir la complejidad del texto, todas ellas ayudándonos a reducir el tamaño del vocabulario (conjunto de palabras/tokens únicos) de nuestros modelos. Los algoritmos de **Machine Learning** trabajan con datos numéricos, no pueden usar el texto directamente. Hay muchas formas de convertir el texto en vectores numéricos (**feature engineering**), y vamos a explorar las dos formas más básicas.


### Bag of words

Para crear la representación de *bag of words*, cada token es una dimensión en un espacio de dimensión el tamaño del vocabulario $V$. Para construir el vector de una frase, será la suma de los vectores de cada token, es decir, convertimos cada frase en un vector que cuenta el número de ocurrencias de cada token. Hacemos la distinción entre palabra y token porque **se pueden usar como tokens n-gramas de palabras**, lo que aumenta la complejidad del modelo, pero permite captar correlaciones en el orden de las palabras.


Se siguien los pasos:
1. Encuentra los **V** tokens mas comunes del corpus de entrenamiento y se les asigna un índice, este es nuestro **vocabulario**. Creamos un diccionario para convertir de tokens a índices y viceversa.
2. Para cada frase en el corpus, creamos un vector de dimensión **V** y lo inicializamos con ceros.
3. Iteramos sobre los tokens de cada frase, y si el token está en el diccionario, incrementamos en 1 el índice correspondiente del vector.

Veamos el siguiente ejemplo con **V=4**, cuyo vocabulario es:

    ['hi', 'you', 'me', 'are']

Les asignamos un índice: 

    {'hi': 0, 'you': 1, 'me': 2, 'are': 3}

Y si tenemos la siguiente frase:

    'hi, how are you? are you ok?'

que hemos preprocesado a la siguiente lista de tokens:

    ['hi', 'how', 'are', 'you', 'are', 'you', 'ok']

Inicializamos el vector:

    [0, 0, 0, 0]
    
Iteramos sobre las palabras, teniendo en cuenta sólo aquellas en nuestro vocabulario:

    'hi':  [1, 0, 0, 0]
    'how': [1, 0, 0, 0] # word 'how' is not in our dictionary
    'are': [1, 0, 0, 1]
    'you': [1, 1, 0, 1]
    'are': [1, 1, 0, 2]
    'you': [1, 2, 0, 2]
    'ok':  [1, 2, 0, 2] # word 'ok' is not in our dictionary

El vector resultante es:

    [1, 2, 0, 2]
    
En el [s1_challenge](./s1_challenge.ipynb) construiremos este vectorizador a mano, pero viene implementado en sklearn en la función [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).
<!-- #endregion -->

**Ejercicio:** Usando el vocabulario del ejemplo anterior, cuál sería el vector *bag of words* de la frase con tokens

`['hi', 'i', 'am', 'david,', 'and', 'you?', 'are', 'you', 'ok?']`


**Ejercicio:** La variable `quijote` contiene el texto completo del Quijote:

* ¿De qué tamaño serán los vectores *bag of words* sin utilizar ninguna de las técnicas de reducción anteriores? Tokeniza con `str.split()` y con el tokenizador de una de las librerías.
* ¿Por qué si tokenizamos con la librería cambia tanto el tamaño del vocabulario?
* ¿Y si sólo trabajas con minúsculas?

```python
with open('./resources/quijote_largo.txt', 'r', encoding='UTF-8') as f:
    quijote = f.read()
```

```python
## ESCRIBE AQUÍ LA SOLUCIÓN

```

```python
# tamaño del vocabulario

len(set(quijote.split()))
```

```python
# esto con minusculas
len(set(quijote.lower().split()))
```

```python
quijote.split()[:200]
```

```python
tokenize(quijote)[:200]
```

```python
tokenize = lambda x: nltk.tokenize.word_tokenize(x, language='spanish')
len(set(tokenize(quijote)))
```

```python
len(set(tokenize(quijote.lower())))
```

```python





















```

```python
print(f'Python split, V: {len(set(quijote.split()))}')
```

```python
print(
    f'NLTK word_tokenize, V: {len(set(nltk.tokenize.word_tokenize(quijote)))}')
```

```python
# Crea un Tokenizer de la nada, sólo teniendo en cuenta el vocabulario
tokenizer_vocab = spacy.tokenizer.Tokenizer(nlp.vocab)
print(
    f'spacy tokenizer, V: { len({token.text for token in tokenizer_vocab(quijote)})} '
)
```

```python
# Crea un Tokenizer con las opciones por defecto de un idioma
# en este caso incluye el vocabulario, puntuación y excepciones
tokenizer_complete = nlp.tokenizer
print(
    f'spacy tokenizer, V: { len({token.text for token in tokenizer_complete(quijote)})} '
)
```

```python
print(f'Python split, V: {len(set(quijote.lower().split()))}')
print(
    f'NLTK word_tokenize, V: {len(set(nltk.tokenize.word_tokenize(quijote.lower())))}'
)
print(
    f'spacy tokenizer, V: {len({token.text for token in tokenizer_complete(quijote.lower())})} '
)
```

### tf-idf (term frequency - inverse document frequency)

Una extensión sobre *bag of words* es la representación **tf-idf**. [Link](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). En este caso, también se cuenta la ocurrencia de cada token, pero se pondera para penalizar palabras que aparecen en casi todos los textos (como las *stop words*), por lo que no ayudan a diferenciarlos. Es muy popular para tareas de **information retrieval**.

Si utilizamos una representación *bag of words* para crear un modelo, esos términos muy frecuentes pueden ensombrecer otros tokens cuyas frecuencias son mucho menores, pero que contienen más información. Una posible solución a este problema es darle un peso distinto a cada una de las dimensiones del vector. El esquema de ponderado más común es:

* term frequency: $$
\operatorname{tf}(t, d) = \frac{|\{t \in d\}| }{|d|} \;,
$$
que es el porcentaje de veces que aparece el token $t$ en el documento $d$. Da la importancia que tiene un término en un documento concreto.
* inverse document frequency: 
$$
\operatorname{idf}(t, D)=\log \frac{|D|}{1+|\{d \in D: t \in d\}|} \;,
$$
donde $|D|$ es el número de documentos, y $|\{d \in D: t \in d\}|$ es el número de documentos en los que aparece el término $t$, el factor 1 se añade para que no se divida entre 0. Quiza importancia a términos que están presentes en muchos documentos.

Luego, cada término es ponderado por el siguiente factor:
$$
\operatorname{tfidf}(t, d, D) = \operatorname{tf}(t, d)\times \operatorname{idf}(t, D) \; .
$$

En el caso de `sklearn`, por defecto normaliza los vectores resultante según la norma euclídea o L2 ([Más información](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)). La función que lo implementa es [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).


**Ejercicio:**
* ¿Qué distancia hay entre dos tokens en una representación *bag of words*? ¿Cambia esta distancia si los tokens tienen un significado más o menos cercano? ¿Cambia esto en la representación *tf-idf*?
* ¿En una tarea de clasificación, si un término aparece en todos los casos de una clase, cómo afecta al su valor de $\operatorname{tfidf}(t, d, D)$? ¿Es este término una feature importante para clasificar?
* Una vez contruido un vocabulario, ¿qué pasará si en el conjunto de validación, test o en producción aparecen palabras nuevas?
* ¿Cómo actúan las herramientas aprendidas ante las erratas? Analiza el caso de que existan erratas tanto durante el entrenamiento como en producción? Por ejemplo: 'ayuntamiento' vs 'ayuntameinto'.
* En el lenguaje, el orden de los factores altera el producto. 'Juan mordió a el perro' vs 'El perro mordió a Juan'. 'Si no viene a comer lo dejamos', 'No viene a comer si lo dejamos'. ¿Cómo actúa *bag of words* y *tf-idf* ante este fenómeno?

```python
# Ejemplos de tokens en representación BoW

t1 = [1,0,0,0]

t2 = [0,1,0,0]

t3 = [0,0,1,0]
```
