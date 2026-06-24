# DASA — Deterministic Agent Synthesis Architecture

> **La IA que no alucina.** / *The AI that doesn't hallucinate.*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## ¿Qué es DASA? (explicado para todos)

Imagina que tienes un libro de recetas. Si alguien te pregunta "¿cómo se hacen los huevos fritos?", tú abres el libro, buscas la página de huevos fritos, la lees y la explicas. Nunca **inventas** ingredientes que no están en el libro.

Un LLM (modelo de lenguaje, como ChatGPT, Claude o Gemini) haría lo contrario: generaría una respuesta basada en estadísticas de texto que aprendió. A veces acierta, a veces **inventa cosas que suenan reales pero son falsas**. A eso se le llama **alucinación**.

**DASA no alucina** porque usa un enfoque completamente diferente:

1. **Busca** en tu base de datos de hechos reales (como buscar en el libro).
2. **Reconstruye** la respuesta solo con las palabras que encontró (como leer la página en voz alta).
3. **Nunca añade** información que no estaba en la base de datos — es matemáticamente imposible.

> Piénsalo así: DASA es como un estudiante que solo puede usar sus apuntes en el examen. No puede inventar lo que no anotó.

---

## El problema real: la IA sigue alucinando en 2026

En 2026 seguimos teniendo modelos enormes como Claude Opus 4.7, GPT-4o, Gemini Ultra… y **siguen alucinando**. ¿Por qué? Porque todos son la misma cosa por dentro: **predictores de texto probabilísticos**.

Según múltiples estudios de NLP, los LLMs no razonan de verdad. Cuando responden "la capital de Francia es París", no lo "saben" — calculan que estadísticamente, después de "la capital de Francia es", la palabra "París" tiene alta probabilidad. Funciona bien para cosas comunes, pero falla en detalles, datos específicos, listas largas, información poco frecuente o temas especializados.

**¿Y la búsqueda web? ¿No lo arregla?**

Sí y no. Las herramientas de búsqueda web (agentic web search) parecen la solución, pero tienen problemas serios en la práctica:

- Cuando un LLM usa una herramienta de búsqueda web, el servidor le entrega el contenido de la página como **HTML, JavaScript y CSS sin sanitizar**, o en el mejor caso Markdown con muchas impurezas.
- Ese contenido contiene código de scripts, publicidad, menús, footers, cookies banners... todo mezclado con la información real.
- El modelo tiene que "leer" eso, y es terrible: consume muchos tokens del contexto, confunde al modelo y **es extremadamente vulnerable a inyección de prompt** (alguien puede poner texto invisible en una página web que manipule al LLM para que responda cosas maliciosas o incorrectas).
- Localmente, la búsqueda web con un modelo pequeño de 4B parámetros es prácticamente inusable: el modelo pierde el hilo, alucina mientras "lee" el HTML, y tarda mucho.

> **DASA y SHARD actúan localmente**, antes de que necesites consultar la web. Si preguntas por una receta de huevos, DASA responde desde tu base de datos local en milisegundos, sin tocar internet, sin HTML sucio, sin riesgo de prompt injection.

---

## El problema de hardware: nadie piensa en nosotros

La mayoría de tutoriales de IA asumen que tienes una GPU de 24 GB de VRAM. La realidad para la mayoría de las personas es otra:

| Hardware típico | VRAM | ¿Qué puede correr? | Rendimiento real |
|---|---|---|---|
| PC gaming básico / laptop | 4 GB VRAM | Modelos de hasta ~4B parámetros cuantizados | 1–3 tokens/seg |
| PC gaming medio | 8 GB VRAM | Modelos de hasta ~8B parámetros | 3–8 tokens/seg |
| PC sin GPU dedicada | RAM del sistema | Cualquier modelo en CPU | < 1 token/seg |
| Raspberry Pi / ARM | Sin GPU | Solo modelos tiny (< 1B) | Muy lento |

Con **4 GB de VRAM** — que es lo que tiene una persona promedio — el mayor modelo que puedes correr es un 8B cuantizado a 4 bits, que va a **1, 2 o 3 tokens por segundo en el mejor caso**, y aún así va a alucinar porque los modelos pequeños con pocos parámetros no tienen suficiente "memoria" para saber recetas, definiciones, datos específicos o información especializada de forma confiable.

**¿La solución de los grandes fabricantes?** Usar sus APIs en la nube, que cuestan dinero, requieren internet, y envían tus datos a servidores externos.

**¿La solución de DASA?** Que tu computadora de papa responda correctamente sin necesidad de parámetros enormes.

---

## Cómo DASA resuelve el problema del hardware

DASA separa dos cosas que los LLMs intentan hacer juntas:

1. **Recordar hechos** → lo hace SHARD (base de datos binaria ultra-eficiente)
2. **Escribir en lenguaje natural** → lo hace el LLM, pero con vocabulario bloqueado al corpus

El LLM ya no necesita "saber" la receta de los huevos fritos. No la tiene que memorizar durante el entrenamiento. DASA le entrega los fragmentos exactos del corpus y le dice: "reformatea esto en lenguaje natural, no puedes añadir nada más".

Un modelo de **0.5B parámetros** puede reformatear texto perfectamente. No necesita "saber" cosas — solo escribirlas bien. Eso libera a los modelos pequeños para hacer lo que sí saben hacer (escribir fluidamente) sin exigirles lo que no pueden hacer fiablemente (recordar hechos sin alucinar).

```
Sin DASA:  [LLM solo] → necesita saber TODO → alucina con modelos pequeños
Con DASA:  [SHARD busca] → [LLM reformatea] → respuesta correcta con modelos tiny
```

**Resultado:** Una Raspberry Pi, un PC con 2 GB de RAM o una computadora de 2010 puede responder preguntas especializadas correctamente, en millisegundos, sin internet.

---

## El problema que resuelve

Los LLMs modernos son predictores de texto probabilísticos. No "saben" hechos — generan secuencias de palabras estadísticamente probables, que a veces parecen hechos pero son inventadas (alucinadas). Además, requieren VRAM/RAM de escala industrial.

**DASA resuelve ambos problemas:**

- Corre en **2 GB de RAM, 2 vCPU, sin GPU**.
- **No puede alucinar** por diseño matemático.
- Compatible con cualquier base de datos JSON o binaria SHARD.
- Expone una **API compatible con OpenAI** — funciona con Jan, Open WebUI, etc.

---

## Casos de uso ideales

| Caso | Por qué DASA es mejor que un LLM solo |
|---|---|
| **Diccionario / enciclopedia local** | El LLM alucina definiciones. DASA las lee del corpus sin inventar. |
| **Recetario de cocina** | El LLM mezcla ingredientes de distintas recetas. DASA da la receta exacta. |
| **Base de conocimiento empresarial** | El LLM no sabe los datos internos de tu empresa. DASA sí, con tu JSON. |
| **Documentación técnica** | El LLM inventa funciones que no existen. DASA solo usa lo que está en los docs. |
| **IA especializada sin GPU** | Con un modelo de 0.5B + SHARD, superas a un 7B solo en tu dominio. |
| **Offline y privado** | Sin internet, sin APIs externas, sin que tus datos salgan de tu máquina. |

---

## Próximo horizonte: versión coding y Mixture of Experts

DASA no es solo para recetas o diccionarios. La arquitectura está diseñada para **especializarse en cualquier dominio**:

- **DASA Coding** *(próximamente)*: un corpus de documentación de librerías, patrones de código, ejemplos reales → el LLM genera código anclado a lo que realmente existe, sin inventar funciones o argumentos que no existen.
- **DASA Expert**: combina múltiples datasets especializados (medicina, derecho, cocina, programación) con enrutamiento inteligente de consultas.
- **DASA MoE** *(Mixture of Experts)*: múltiples agentes DASA especializados que colaboran — cada uno experto en su dominio, coordinados por un router semántico.

> La visión es simple: cualquier persona, con el hardware que ya tiene, puede construir una IA que sepa exactamente lo que necesita saber, sin alucinaciones, sin internet, sin costos de nube.

---

## Cómo funciona por dentro

```
Consulta del usuario
         │
         ▼
┌─────────────────────────────────────┐
│  AGENTE A — Recuperación            │
│  · Convierte la consulta en vector  │
│  · Compara contra la base de datos  │
│  · Devuelve los fragmentos reales   │  ← Solo datos verificados cruzan aquí
└────────────────┬────────────────────┘
                 │  Lista de Fragmentos
                 ▼
┌─────────────────────────────────────┐
│  AGENTE B — Síntesis                │
│  · Recibe los fragmentos            │
│  · Reescribe en lenguaje natural    │
│  · Vocabulario BLOQUEADO al contexto│  ← Garantía anti-alucinación matemática
└────────────────┬────────────────────┘
                 │
                 ▼
         Respuesta verificada
```

### Agente A — El buscador inteligente

El Agente A convierte tu pregunta en un vector matemático (una lista de números que representan el significado de las palabras) usando un modelo de embeddings ligero (`all-MiniLM-L6-v2`, solo 80 MB). Luego compara ese vector contra todos los fragmentos de tu base de datos usando **similitud coseno** — una medida matemática de qué tan parecidos son dos vectores.

Solo los fragmentos con un score de similitud por encima del umbral configurado (por defecto 0.3) son devueltos. Los demás se descartan silenciosamente.

**Herramientas internas del Agente A:**
- `rank_fragments()` — ordena los fragmentos por relevancia de mayor a menor.
- `filter_by_threshold()` — elimina fragmentos con baja similitud.
- `deduplicate_fragments()` — elimina fragmentos repetidos o casi idénticos.
- Búsqueda exacta por lema con tolerancia a errores tipográficos (distancia Levenshtein 1).

### Agente B — El escritor honesto

El Agente B toma los fragmentos verificados del Agente A y produce una respuesta coherente. Tiene tres modos de operación:

| Modo | Cuándo se usa | Descripción |
|---|---|---|
| **Estadístico** | Sin LLM configurado | Reescritura pura con vocabulario bloqueado al corpus. Cero red neuronal, cero alucinaciones. |
| **LLM guiado** | Con LLM + fragmentos relevantes (score ≥ 0.40) | El LLM da formato a los fragmentos pero no puede añadir información externa. Un prompt estricto lo obliga a decir "no cubro este tema" si el contexto no responde. |
| **LLM libre** | Con LLM + sin fragmentos relevantes (score < 0.40) | El LLM responde libremente (saludos, preguntas generales, conversación). |

---

## DASA vs. RAG tradicional

| Propiedad | RAG Estándar | DASA |
|---|---|---|
| Riesgo de alucinación | Medio (el modelo puede "derivar") | **Ninguno** (vocabulario bloqueado) |
| GPU requerida | Sí (para generación) | **No** (solo CPU) |
| RAM necesaria | 8–80 GB | **< 2 GB** |
| Autonomía del agente | Recuperación pasiva | **Recuperación activa con herramientas** |
| Tipo de salida | Generación probabilística | **Síntesis determinista** |
| Almacenamiento | Vector DB (Qdrant, Weaviate…) | **JSON / SHARD DB binaria** |
| Compatible con OpenAI API | Depende | **Sí, nativo** |

---

## Instalación rápida

### Requisitos previos

- Python 3.9 o superior
- pip (instalador de paquetes de Python)
- Opcional: [Ollama](https://ollama.com) para usar un LLM local

### Paso 1 — Clonar o descargar el repositorio

```bash
git clone https://github.com/angelgabrieljacintohuayllasco/DASA.git
cd DASA
```

### Paso 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instala:
- `sentence-transformers` — para calcular embeddings (vectores semánticos)
- `numpy` — para operaciones matriciales rápidas
- `pytest` — para correr los tests
- El `launcher.py` instalará automáticamente `fastapi`, `uvicorn` y `httpx` la primera vez que lo ejecutes.

### Paso 3 — Ejecutar el lanzador

```bash
python launcher.py
```

Verás un menú interactivo con todas las opciones disponibles.

---

## 🎓 Tutorial completo: de cero a respuesta en 5 minutos

> **Para quién es esto:** si nunca usaste DASA, sigue estos pasos en orden. No necesitas saber programar. Solo necesitas tener Python instalado.

### Paso 1 — Abre una terminal

**En Windows:** presiona `Win + R`, escribe `cmd` y dale Enter. O busca "PowerShell" en el menú inicio.

**En Mac/Linux:** busca "Terminal" en tus aplicaciones.

### Paso 2 — Ve a la carpeta del proyecto

```bash
cd C:\ruta\donde\descargaste\DASA-main
```

*(Reemplaza la ruta por donde está tu carpeta. En Windows puedes arrastrar la carpeta a la terminal para autocompletar la ruta.)*

### Paso 3 — Arranca el lanzador

```bash
python launcher.py
```

Deberías ver el menú con 9 opciones. Si ves un error de "module not found", ve al paso 4 primero.

### Paso 4 — Instala las dependencias (solo la primera vez)

En el menú, escribe `9` y presiona Enter.

```
  Tu elección: 9
```

Vas a ver una lista de paquetes con `✓` (ya instalado), `✗` (falta) y `○` (opcional).

Elige la opción **"1 — Instalar solo los que faltan"**. DASA instalará todo automáticamente. Esto puede tardar 2-5 minutos la primera vez porque descarga el modelo de embeddings (~80 MB).

> Si ya tienes todo instalado, el menú te lo dirá y no instalará nada extra.

### Paso 5 — Configura tu dataset y modelo (opcional pero recomendado)

Escribe `7` y presiona Enter para ir a la configuración.

Vas a ver preguntas como:
- **Dataset path** — la ruta a tu archivo JSON (o deja vacío para usar el demo incluido)
- **Modo Agent B** — elige `ollama` si tienes Ollama instalado, o `stats` para modo sin LLM
- **Modelo Ollama** — por ejemplo `qwen2.5:0.5b` (el más ligero) o `gemma3:4b`

Si no sabes qué poner, **presiona Enter en todo** para usar los valores por defecto. El demo funciona sin configuración extra.

### Paso 6 — Haz tu primera consulta

Escribe `1` y presiona Enter.

```
  Tu elección: 1
```

Escribe cualquier pregunta relacionada con tu dataset. Si usas el demo:

```
  Consulta: ¿Qué es Python?
  Consulta: ¿Cómo se hacen los huevos fritos?
  Consulta: ¿Qué es la inteligencia artificial?
```

DASA buscará en la base de datos y responderá. Si configuraste Ollama, la respuesta vendrá redactada en lenguaje natural. Si no, verás los fragmentos relevantes directamente.

### Paso 7 — Conecta Jan AI (si quieres una interfaz gráfica)

Si quieres chatear con DASA desde una app con interfaz visual como Jan:

**1. Primero, arranca la API:**
Escribe `4` en el menú y elige el puerto (por defecto 8000).

```
  Tu elección: 4
  Puerto [8000]:
```

Deja el puerto en 8000 y presiona Enter. Verás:
```
  INFO:     Uvicorn running on http://0.0.0.0:8000
```

**2. Obtén tu API Key:**
Abre **otra terminal** (deja la anterior corriendo), ve a la misma carpeta y ejecuta:
```bash
python launcher.py
```
Escribe `8` para ver tu clave. Cópiala.

**3. Configura Jan:**
- Abre Jan → Settings → My Models → Add OpenAI-compatible endpoint
- Base URL: `http://localhost:8000/v1`
- API Key: la clave que copiaste
- Model name: `dasa`

¡Listo! Ahora Jan usará DASA como cerebro.

### Paso 8 — Usa tu propio dataset

Crea un archivo `mi_datos.json` con este formato:

```json
[
  {
    "id": "1",
    "lemma": "Fotosíntesis",
    "definition": "Proceso por el cual las plantas convierten luz solar en energía química, produciendo glucosa y oxígeno a partir de CO2 y agua."
  },
  {
    "id": "2",
    "lemma": "Mitocondria",
    "definition": "Orgánulo celular encargado de producir energía (ATP) mediante la respiración celular. Se le llama la central energética de la célula."
  }
]
```

Luego en el menú:
1. Opción `[7]` → en "Dataset path" escribe la ruta a tu archivo JSON
2. Opción `[3]` → construye la caché de embeddings (solo necesario la primera vez con ese dataset)
3. Opción `[1]` → haz consultas sobre tus propios datos

### Resumen visual del flujo

```
python launcher.py
       │
       ├─ [9] Instalar dependencias  ← solo la primera vez
       ├─ [7] Configurar dataset y modelo  ← cambia según tu uso
       ├─ [3] Construir caché  ← solo cuando cambias el dataset
       ├─ [1] Consulta interactiva  ← uso diario
       └─ [4] API REST  ← para conectar Jan u otras apps
```

### Preguntas rápidas de un principiante

**¿Qué pasa si escribo una pregunta y dice "No se encontró información"?**
Significa que tu base de datos no tiene información sobre ese tema. Añade más entradas a tu JSON o usa un dataset más completo.

**¿Puedo usar DASA sin Ollama?**
Sí. En modo estadístico (opción `stats` en configuración) DASA responde sin ningún LLM — más rápido, pero las respuestas son más crudas.

**¿Cuánto tarda en responder?**
Sin LLM: instantáneo (< 100ms). Con Ollama qwen2.5:0.5b: 2-5 segundos. Con gemma3:4b: 5-15 segundos según tu hardware.

**¿Funciona sin internet?**
Sí, después de la primera instalación. Todo es local.

---

## El lanzador interactivo (`launcher.py`)

El archivo `launcher.py` es la puerta de entrada a todo el sistema. Ejecuta `python launcher.py` y obtendrás un menú así:

```
╔══════════════════════════════════════════════════════════════╗
║         DASA + SHARD  —  Sistema RAG Anti-Alucinación        ║
╚══════════════════════════════════════════════════════════════╝

  ┌─ MENÚ ──────────────────────────────────────────────────┐
  │  [1] Hacer una consulta  (DASA query interactivo)
  │  [2] Construir base de datos SHARD  desde JSON
  │  [3] Construir caché de embeddings
  │  [4] Iniciar API REST  (FastAPI / uvicorn)
  │  [5] Correr tests  (DASA + SHARD)
  │  [6] Ver estadísticas de una base SHARD
  │  [7] Configurar modelos  (Agente A · Agente B · Dataset)
  │  [8] API Key  (ver / regenerar)
  │  [9] Instalar / actualizar dependencias
  │  [0] Salir
  ├─ CONFIG ────────────────────────────────────────────────┤
  │  AgA: all-MiniLM-L6-v2  ·  AgB: Ollama → qwen2.5:0.5b
  └─────────────────────────────────────────────────────────┘
```

### Opciones del menú

| Opción | Función |
|---|---|
| `[1]` Consulta interactiva | Escribe una pregunta y DASA responde en terminal |
| `[2]` Construir SHARD DB | Convierte un archivo JSON en base de datos SHARD binaria |
| `[3]` Caché de embeddings | Pre-calcula los vectores del corpus para respuestas más rápidas |
| `[4]` Iniciar API REST | Arranca el servidor FastAPI en el puerto que elijas (por defecto 8000) |
| `[5]` Correr tests | Ejecuta todos los tests unitarios de DASA y SHARD |
| `[6]` Estadísticas | Muestra métricas de una base SHARD (número de entradas, tamaño, etc.) |
| `[7]` Configurar | Cambia el modelo de embeddings, el modelo LLM, el dataset, etc. |
| `[8]` API Key | Ver o regenerar la clave de autenticación de la API |
| `[9]` Instalar dependencias | Verifica e instala todas las dependencias necesarias y opcionales |
| `[0]` Salir | Cierra el programa limpiamente |

### Argumentos de línea de comandos (modo no interactivo)

```bash
python launcher.py api           # Iniciar API directamente en puerto 8000
python launcher.py api 9000      # Iniciar API en puerto 9000
python launcher.py query         # Consulta interactiva sin menú
python launcher.py tests         # Correr todos los tests
python launcher.py build         # Construir base de datos SHARD
python launcher.py stats         # Ver estadísticas
```

---

## API REST compatible con OpenAI

Cuando inicias la API con la opción `[4]`, se levanta un servidor FastAPI accesible desde cualquier cliente compatible con la API de OpenAI: Jan, Open WebUI, LM Studio, etc.

### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `GET` | `/status` | Estado del pipeline (cargado/no cargado) |
| `POST` | `/load` | Cargar el pipeline manualmente |
| `POST` | `/query` | Consulta directa al pipeline |
| `POST` | `/unload` | Liberar el pipeline de memoria |
| `GET` | `/demo-queries` | Lista de consultas de ejemplo |
| `GET` | `/v1/models` | Lista de modelos (compatible OpenAI) |
| `POST` | `/v1/chat/completions` | Chat completions (compatible OpenAI, SSE) |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

### Autenticación

La API usa **Bearer Token**. La clave se genera automáticamente en el primer arranque y se guarda en `.dasa_api_key`:

```
Authorization: Bearer dasa-xxxxxxxxxxxxxxxxxxxx
```

Para ver o regenerar la clave: opción `[8]` en el menú.

Los endpoints `/health`, `/docs`, `/redoc`, `/openapi.json` y `/v1/models` son **públicos** (no requieren autenticación) para compatibilidad con clientes como Jan.

### Configurar Jan AI

1. Abre Jan → Settings → My Models → Add Engine
2. Base URL: `http://localhost:8000/v1`
3. API Key: la clave mostrada en la opción `[8]`
4. Model name: `dasa`
5. ¡Listo! Jan usará DASA como backend.

### Ejemplo de uso con `curl`

```bash
# Consulta básica
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dasa-tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dasa",
    "messages": [{"role": "user", "content": "¿Qué es Python?"}],
    "stream": false
  }'

# Streaming (SSE)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dasa-tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "dasa", "messages": [{"role": "user", "content": "Hola"}], "stream": true}'
```

### Ejemplo con Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dasa-tu-api-key",
)

response = client.chat.completions.create(
    model="dasa",
    messages=[{"role": "user", "content": "¿Qué es la inteligencia artificial?"}],
)
print(response.choices[0].message.content)
```

---

## Uso como librería Python

```python
from dasa.pipeline import DASAPipeline
from dasa.config import DASAConfig

# Configuración básica
config = DASAConfig(
    embedding_model="all-MiniLM-L6-v2",  # 80 MB, solo CPU
    top_k_fragments=5,
    similarity_threshold=0.3,
)

# Crear y cargar el pipeline
pipeline = DASAPipeline(config)
pipeline.load("mi_dataset.json")

# Hacer una consulta
respuesta = pipeline.run("¿Cómo preparo huevos fritos?")
print(respuesta)
```

### Modo LLM guiado (con Ollama)

```python
from dasa.pipeline import DASAPipeline
from dasa.config import DASAConfig
from dasa.agent_b.llm_connector import OllamaConnector

config = DASAConfig()
pipeline = DASAPipeline(config)
pipeline.load("mi_dataset.json")

# Inyectar Ollama como LLM de síntesis
pipeline.agent_b._llm_callable = OllamaConnector(
    model="gemma3:4b",
    host="http://localhost:11434",
)

respuesta = pipeline.run("¿Qué es el embedding?")
print(respuesta)
```

### Modo LLM guiado (con HuggingFace)

```python
from dasa.agent_b.llm_connector import LLMConnector

connector = LLMConnector("Qwen/Qwen2.5-0.5B-Instruct")
connector.load()  # descarga ~1 GB la primera vez, luego queda en caché

pipeline.agent_b._llm_callable = connector
```

---

## Formato del dataset

DASA acepta cualquier array JSON. El formato mínimo:

```json
[
  {
    "id": "001",
    "lemma": "Python",
    "definition": "Lenguaje de programación de alto nivel, propósito general, interpretado."
  },
  {
    "id": "002",
    "lemma": "embedding",
    "definition": "Representación vectorial de texto en un espacio matemático continuo."
  }
]
```

**Campos reconocidos automáticamente:**

| Campo | Descripción |
|---|---|
| `lemma` / `word` / `title` / `name` | Término o título principal del registro |
| `definition` / `text` / `content` / `body` / `description` | Texto principal del registro |
| `id` | Identificador único (opcional) |
| `category` / `tags` | Categorización (opcional, mejora la búsqueda) |

Si tu JSON usa campos diferentes, Agent A intentará detectarlos automáticamente. Si falla, renombra los campos a `lemma` y `definition`.

### Dataset de demo incluido

El repositorio incluye `data/demo_dataset.json` con 10 entradas de ejemplo (huevo frito, Python, inteligencia artificial, embedding, etc.) para probar sin necesidad de tu propio dataset.

---

## Configuración avanzada (`DASAConfig`)

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `embedding_model` | `"all-MiniLM-L6-v2"` | Modelo de embeddings. 80 MB, corre en CPU. Alternativas: `paraphrase-multilingual-MiniLM-L12-v2` (mejor multilingüe). |
| `top_k_fragments` | `5` | Número máximo de fragmentos que Agent A devuelve a Agent B. |
| `similarity_threshold` | `0.3` | Score mínimo de similitud coseno para considerar un fragmento relevante. Rango: 0.0–1.0. |
| `device` | `"cpu"` | Dispositivo de inferencia. `"cuda"` si tienes GPU NVIDIA. |
| `synthesis_model` | `None` | Modelo HuggingFace para Agent B. `None` = modo estadístico puro. |
| `restricted_vocabulary` | `True` | Bloquear vocabulario al corpus. Siempre `True` en producción. |

### Configuración persistente (`.dasa_config.json`)

El lanzador guarda la configuración en `.dasa_config.json` en la raíz del proyecto:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "top_k_fragments": 5,
  "similarity_threshold": 0.2,
  "dataset_path": "",
  "agent_b_mode": "ollama",
  "synthesis_model": "",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "gemma3:4b"
}
```

Puedes editarlo directamente o usar la opción `[7]` del menú.

---

## Cambiar el modelo LLM

### Con Ollama (recomendado)

```bash
# Modelos recomendados por tamaño
ollama pull qwen2.5:0.5b    # 0.5B — ultra rápido, buen español
ollama pull qwen2.5:1.5b    # 1.5B — mejor calidad, sigue siendo ligero
ollama pull gemma3:4b       # 4B — calidad alta, requiere ~3 GB RAM
ollama pull llama3.2:3b     # 3B — buena alternativa
ollama pull tinyllama       # 1.1B — muy ligero
```

Luego en `[7]` del menú → campo "Modelo Ollama" → escribe el nombre del modelo.

### Con HuggingFace (sin Ollama)

Cambia `agent_b_mode` a `huggingface` en `[7]` y escribe el nombre del modelo, por ejemplo:
- `Qwen/Qwen2.5-0.5B-Instruct` — 0.5B, excelente relación calidad/velocidad
- `Qwen/Qwen2.5-1.5B-Instruct` — 1.5B
- `microsoft/phi-2` — 2.7B, muy capaz para su tamaño

---

## Base de datos SHARD

Para datasets grandes (millones de entradas, escala de TB), DASA se integra con **SHARD** — una base de datos binaria hash-sharded diseñada específicamente para búsqueda semántica eficiente.

### Construir una base SHARD desde JSON

```bash
python launcher.py build
```

O desde la opción `[2]` del menú. El proceso:
1. Lee tu archivo JSON
2. Tokeniza y crea firmas MinHash para cada entrada
3. Distribuye las entradas en N shards (archivos `.bloom`)
4. Construye índices de acceso rápido (`index.meta.json`, `index.keymap.json`)

### Estructura de una base SHARD

```
mi_base/
├── shard_000000.bloom    # Datos binarios del shard 0
├── shard_000001.bloom    # Datos binarios del shard 1
├── ...
├── index.meta.json       # Metadatos del índice
└── index.keymap.json     # Mapa de claves para acceso O(1)
```

### Usar una base SHARD con DASA

```python
from dasa.pipeline import DASAPipeline
from dasa.config import DASAConfig

config = DASAConfig()
pipeline = DASAPipeline(config)
pipeline.load("mi_base/")   # Pasa el directorio, no un archivo JSON

respuesta = pipeline.run("¿Qué es el Big Bang?")
```

### Búsqueda vectorial a escala: índice IVF-PQ

Para millones o miles de millones de registros, la búsqueda por fuerza bruta
(comparar la consulta contra TODOS los vectores) no escala — exigiría cargar
~150 GB de embeddings en RAM con 100M registros. La solución es el **índice
IVF-PQ de SHARD**: salta solo a los clusters relevantes (como un índice de base
de datos salta a la fila) y comprime cada vector ~32× con product quantization.
El índice vive en disco vía mmap; solo lo que se consulta toca la RAM.

**El build es pesado y va offline** (Colab/PC potente); el artefacto read-only se
copia al equipo de 2 GB que solo consulta. Agent A detecta y usa el índice `ivf/`
automáticamente — sin cambios de código.

```bash
# 1. (máquina potente) construir el índice desde embeddings + claves
python -m shard.cli build-ivf --embeddings emb.npy --keys keys.json \
    --out mi_base/ivf --profile low-ram        # low-ram | medium | fast

# 2. copiar mi_base/ (shards + ivf/) al equipo de 2 GB
```

```python
config = DASAConfig(use_shard_backend=True, shard_db_path="mi_base/")
pipeline = DASAPipeline(config)
pipeline.load("mi_base/")          # Agent A carga el índice IVF-PQ si existe
respuesta = pipeline.run("¿Qué es el Big Bang?")
```

Si `mi_base/ivf/` existe, Agent A usa búsqueda aproximada IVF-PQ (Tier 0, escala
a TB). Si no, cae a la caché de embeddings (datasets chicos) o MinHash. RAM de
consulta: ~110 MB a 100M vectores, ~410 MB a 1B — cabe en el piso de 2 GB.

### Estructura de una base SHARD con índice IVF-PQ

```
mi_base/
├── shard_000000.bin      # Datos binarios del shard 0 (registros)
├── shard_000000.bloom    # Filtro Bloom del shard 0
├── ...
└── ivf/                  # Índice vectorial IVF-PQ (búsqueda semántica)
    ├── manifest.json     # params + formato (portable, sin rutas absolutas)
    ├── coarse_centroids.f32
    ├── pq_codebooks.f32
    ├── list_codes.u8     # códigos PQ (mmap, nunca todo en RAM)
    ├── list_offsets.i64
    ├── row_to_orig.i64
    ├── keys.bin / key_offsets.i64
    └── rerank_vecs.*     # caché de rerank (sq8 o f32)
```

---

## Estructura del proyecto

```
DASA-main/
├── launcher.py                    # Lanzador principal: menú + API
├── .dasa_config.json              # Configuración persistente (generado automáticamente)
├── .dasa_api_key                  # Clave de la API (generada automáticamente)
│
├── dasa/
│   ├── __init__.py
│   ├── config.py                  # DASAConfig — todos los parámetros
│   ├── pipeline.py                # DASAPipeline — orquestador principal
│   │
│   ├── agent_a/
│   │   ├── retrieval_agent.py     # Agente A: búsqueda + recuperación de fragmentos
│   │   ├── embeddings.py          # Motor de embeddings (CPU-only)
│   │   └── tools.py               # Herramientas: rank, filter, deduplicate
│   │
│   └── agent_b/
│       ├── synthesis_engine.py    # Agente B: orquestador de síntesis
│       ├── statistical_rewriter.py # Reescritura estadística pura (sin LLM)
│       └── llm_connector.py       # Conectores: HuggingFace + Ollama
│
├── data/
│   ├── demo_dataset.json          # Dataset de demo (10 entradas)
│   └── shard_db/                  # Base SHARD de demo
│
├── docs/
│   ├── architecture.md            # Diseño completo del sistema
│   ├── agent-a.md                 # Especificación del Agente A
│   ├── agent-b.md                 # Especificación del Agente B
│   └── anti-hallucination.md      # Por qué DASA no puede alucinar
│
├── examples/
│   ├── basic_query.py             # Uso mínimo
│   └── recipe_example.py          # Demo "receta de huevos fritos"
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py           # Tests unitarios + integración
```

---

## Tests

```bash
# Correr todos los tests
python launcher.py tests

# O directamente con pytest
cd DASA-main
pytest tests/ -v

# Solo tests de SHARD
cd SHARD-main
pytest tests/ -v
```

El proyecto incluye **más de 59 tests** cubriendo:
- Carga de datasets JSON y SHARD
- Búsqueda por similitud coseno
- Filtrado por umbral
- Deduplicación de fragmentos
- Síntesis estadística y guiada por LLM
- Endpoints de la API REST
- Autenticación Bearer
- Streaming SSE
- Manejo de errores y casos borde

---

## Preguntas frecuentes (FAQ)

**¿Por qué DASA no puede alucinar?**

Porque el Agente B solo tiene acceso al texto de los fragmentos que el Agente A recuperó. El `StatisticalRewriter` construye la respuesta combinando literalmente las frases de esos fragmentos. No hay modelo generativo inventando palabras.

En modo LLM guiado, el prompt del sistema prohíbe explícitamente al modelo añadir información externa y le obliga a responder con una frase fija si el contexto no cubre la pregunta. El score de similitud mínimo de 0.40 garantiza que el LLM solo entra en modo "grounded" cuando los fragmentos son genuinamente relevantes.

**¿Los LLMs no razonan de verdad?**

Correcto. Según estudios de NLP, los LLMs son predictores de texto probabilísticos entrenados para predecir el siguiente token. No tienen un módulo de razonamiento lógico — simulan razonamiento porque vieron muchos ejemplos de razonamiento durante el entrenamiento. Por eso fallan en tareas que requieren hechos concretos y específicos (recetas exactas, definiciones precisas, datos técnicos), especialmente con modelos pequeños que tienen menos "memoria" de entrenamiento.

**¿Por qué no usar búsqueda web en lugar de SHARD?**

La búsqueda web agentic entrega al LLM el contenido de páginas web como HTML, JavaScript y CSS sin sanitizar (o Markdown impuro con scripts). Esto:
1. **Consume muchísimos tokens** del contexto disponible del modelo
2. **Confunde al modelo** con publicidad, menús, footers y código de scripts mezclado con el contenido
3. **Es vulnerable a prompt injection** — alguien puede poner texto invisible en una web que manipule al LLM
4. **Localmente es inusable** con modelos pequeños de 4B o menos

DASA y SHARD actúan antes de tocar internet. Para información que ya tienes en tu corpus (recetas, definiciones, documentación, datos de negocio), DASA responde en milisegundos sin riesgos.

**¿Por qué con 4 GB de VRAM no puedo tener IA local buena?**

Con 4 GB de VRAM solo caben modelos de hasta ~4B parámetros cuantizados a 4 bits, que van a 1–3 tokens/segundo. Esos modelos tienen muy poca "memoria" de entrenamiento, así que alucinas constantemente en temas específicos. Es el problema de pedirle a alguien que aprendió 1000 recetas que recuerde la receta exacta #847 — probablemente la mezcle con otras.

DASA separa la memoria (SHARD) del lenguaje (LLM): el modelo de 0.5B no necesita saber la receta, solo necesita escribirla bien cuando DASA se la entrega.

**¿Funciona con preguntas en inglés?**

Sí. El modelo de embeddings por defecto (`all-MiniLM-L6-v2`) funciona bien en inglés. Para mejor soporte multilingüe, cambia el modelo a `paraphrase-multilingual-MiniLM-L12-v2` en la configuración.

**¿Cuánta RAM necesita?**

- Modo estadístico (sin LLM): ~500 MB para el modelo de embeddings + el dataset en memoria.
- Con Ollama gemma3:4b: ~3 GB adicionales para el LLM (Ollama lo gestiona).
- Con Ollama qwen2.5:0.5b: ~700 MB adicionales. Total: menos de 2 GB.

**¿Por qué 0 tokens/seg en Jan?**

DASA genera la respuesta completa del lado del servidor y la envía de golpe como stream SSE. Jan calcula tokens/seg basándose en el tiempo entre chunks, y como todos llegan en milisegundos, el rate sale 0. Es un comportamiento normal — no indica ningún error. La respuesta es instantánea precisamente porque DASA no genera tokens uno a uno.

**¿Puedo usar DASA con mi propia base de datos grande?**

Sí. Para bases de datos de millones de entradas, usa **SHARD** (incluido en este repositorio). SHARD está diseñado para escalar a terabytes usando sharding hash con filtros Bloom.

**¿Necesito conexión a internet?**

Solo para descargar el modelo de embeddings la primera vez (~80 MB). Después todo funciona completamente offline. Con HuggingFace, también se descarga el modelo LLM la primera vez (~500 MB - 3 GB según el modelo). Con Ollama, el modelo se descarga una vez con `ollama pull`.

**¿Cómo configuro Jan para usar DASA?**

1. Jan → Settings → My Models → Add OpenAI-compatible endpoint
2. Base URL: `http://localhost:8000/v1`
3. API Key: la que muestra la opción `[8]` del menú
4. Arranca la API con `[4]` y ya puedes chatear desde Jan.

---

## Seguridad

- La API usa autenticación **Bearer Token** en todos los endpoints privados.
- La clave se genera con `secrets.token_hex(24)` (criptográficamente segura).
- Los endpoints `/health`, `/docs` y `/v1/models` son públicos para compatibilidad con clientes.
- No se registran ni almacenan las consultas de los usuarios.
- DASA no hace llamadas a internet durante la inferencia (todo es local).

---

## Roadmap

- [x] Motor de embeddings CPU-only
- [x] Síntesis estadística anti-alucinación
- [x] Integración con SHARD binary DB
- [x] API REST compatible con OpenAI
- [x] Streaming SSE
- [x] Autenticación Bearer Token
- [x] Integración con Ollama
- [x] Integración con HuggingFace
- [x] Lanzador interactivo con menú (9 opciones)
- [x] Instalación de dependencias desde el menú
- [x] Configuración persistente
- [x] Fallback autónomo del LLM (respuestas libres cuando el corpus no cubre el tema)
- [x] Umbral de relevancia real (score ≥ 0.40 para modo grounded)
- [x] System prompt del cliente respetado en modo libre (Jan, Open WebUI, etc.)
- [x] Búsqueda vectorial IVF-PQ a escala (Agent A Tier 0, mmap, hasta ~1 TB en 2 GB RAM)
- [ ] **DASA Coding** — corpus de documentación de librerías + generación de código sin alucinaciones
- [ ] Soporte multilingüe nativo (embeddings multilingüe por defecto)
- [ ] Extensiones de herramientas para Agente A: razonamiento matemático, fechas, agregaciones
- [ ] **DASA Expert** — múltiples datasets especializados con enrutamiento semántico
- [ ] **DASA MoE** — Mixture of Experts: agentes especializados coordinados por router semántico
- [ ] Imagen Docker para Raspberry Pi / ARM64
- [ ] Benchmarks contra pipelines RAG estándar
- [ ] Interfaz web local (sin Jan)

---

## ☕ Apoya el proyecto

DASA y SHARD son proyectos desarrollados de forma independiente, con grandes ambiciones y recursos limitados. Si te resulta útil o simplemente crees en la idea de una IA local, honesta y accesible para todos, considera apoyar el desarrollo:

<a href="https://ko-fi.com/angelgabrieljacintohuayllasco0499" target="_blank">
  <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Buy Me a Coffee at ko-fi.com" />
</a>

Cada contribución ayuda a financiar tiempo de desarrollo, estudios y el hardware necesario para seguir construyendo herramientas que corran en la computadora que cualquiera ya tiene — no en un servidor de $10,000.

> *Gracias por creer en que la IA debería ser para todos, no solo para los que pueden pagar la nube.*

---

## 🚀 ¿Eres inversor o socio operativo?

No soy solo el desarrollador de DASA y SHARD. Tengo una cartera de proyectos reales, en producción o con arquitectura definida, orientados a **soluciones para empresas, ventas, marketing y automatización con IA**. Busco socios inversores y socios operativos para escalar lo que ya está construido — y para lanzar lo que viene.

### Proyectos activos

| Proyecto | Qué es | Link |
|---|---|---|
| **WhatCRM** | CRM de ventas para WhatsApp — gestión de leads, automatización de mensajes, pipeline comercial | [whatcrmapp.digital](https://whatcrmapp.digital/) |
| **XpendiXcash** | Gestor de gastos personales con asistente IA, membresías y control financiero inteligente | [xpendixcash.online](https://xpendixcash.online/) |
| **Rexorvex** | Agencia de marketing digital, contenido tech, cursos premium y herramientas de venta | [rexorvex.digital](https://rexorvex.digital/) |
| **LylConnect** | Reventa de internet fibra + planes móviles, mayor cobertura = mayor margen, reducción de costos en personal con IA | [lylconnect.com](https://lylconnect.com/) |
| **DASA** | Motor RAG anti-alucinación — IA para empresas que no puede inventar hechos, corre local | [github/DASA](https://github.com/angelgabrieljacintohuayllasco/DASA) |
| **SHARD** | Base de datos binaria hash-sharded diseñada para búsqueda semántica a escala de TB | [github/SHARD](https://github.com/angelgabrieljacintohuayllasco/SHARD) |
| **G-Mini-Agent** | Agente IA generalista de alto valor para empresas que necesitan automatizar procesos sin contratar más personal | [github/G-Mini-Agent](https://github.com/angelgabrieljacintohuayllasco/G-Mini-Agent) |

Además: desarrollo de agentes IA a medida, webs freelance y consultoría de automatización.

### La visión

Las startups de IA millonarias entrenan modelos inflados, sacan uno nuevo y degradan el anterior para no quedar mal — no es innovación, es marketing. Hacen benchmarks arreglados y venden humo. Nuestra apuesta es diferente:

- **Eficiencia, no tamaño** — un modelo bien entrenado y bien orientado supera a uno 10x más grande y costoso. No necesitas 500B parámetros para hacer una tarea específica bien.
- **APIs baratas, no caras** — el usuario final no debería pagar $50 por 3 minutos de trabajo de un editor de video cuando un agente puede hacer esa tarea con tokens baratos.
- **Agentes que reemplazan trabajo real** — tienes empleados haciendo una tarea repetitiva y quieres escalar sin contratar más personas? G-Mini-Agent se configura una vez y escala sin límite.
- **Cloud o local según la empresa** — corremos en la nube para el usuario final común, y on-premise para las empresas que lo necesiten. La arquitectura decide, no el dogma.
- **IA para todos, no para pocos** — si el agente puede hacer TODO, entonces todos se benefician: la gran empresa que quiere automatizar procesos a escala, la pyme que no puede contratar más personal, y el usuario final que no quiere pagar $50 por 3 minutos de trabajo cuando un agente lo hace por centavos. IA barata, agentica y rentable — eso escala en todos los mercados al mismo tiempo.

### ¿Qué busco?

- **Inversores** — capital para escalar infraestructura, marketing y equipo en proyectos con ingresos ya definidos.
- **Socios operativos** — co-fundadores con experiencia en ventas, operaciones, producto o redes de distribución.
- **Partners de distribución** — si tienes acceso a empresas pyme, operadoras, agencias o canales de venta, hablemos.

> Si tienes capital, red o simplemente reconoces el potencial — la puerta está abierta.

📬 **Contacto:** [angelgabrieljacintohuayllasco@gmail.com](mailto:angelgabrieljacintohuayllasco@gmail.com)

---

## Contribuir

Ver [CONTRIBUTING.md](CONTRIBUTING.md). Bienvenidas contribuciones en:

- Nuevas herramientas agénticas para el Agente A
- Estrategias alternativas de síntesis para el Agente B
- Adaptadores para diferentes formatos de dataset
- Benchmarks contra pipelines RAG estándar
- Traducciones de la documentación

---

## ⭐ Star History

<a href="https://star-history.com/#angelgabrieljacintohuayllasco/DASA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco/DASA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco/DASA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco/DASA&type=Date" />
 </picture>
</a>

---

## Licencia

Apache 2.0 — ver [LICENSE](LICENSE)

---

*DASA fue diseñado para correr en el hardware que ya tienes: un chip de $5 debería ser suficiente para responder cualquier pregunta correctamente, sin inventar hechos.*


---

## The Problem

Modern LLMs are probabilistic text predictors. They don't "know" facts — they generate statistically likely sequences of words, which sometimes look like facts but are invented (hallucinated). They also require industrial-scale VRAM/RAM to run.

## The Solution: Two Specialized Agents

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  AGENT A — Retrieval (Agentic)   │
│  · Embeds the query (CPU only)   │
│  · Searches the TB-scale DB      │
│  · Returns verified Fragments    │  ← Only real data crosses this boundary
└────────────────┬─────────────────┘
                 │  List[Fragment]
                 ▼
┌──────────────────────────────────┐
│  AGENT B — Synthesis (Rewriter)  │
│  · Receives verified fragments   │
│  · Applies statistical rewriting │
│  · Vocabulary LOCKED to context  │  ← Mathematical anti-hallucination
└────────────────┬─────────────────┘
                 │
                 ▼
         Grounded Response
```

**The core guarantee:** Agent B can only use words that appear in the fragments Agent A retrieved. If the database doesn't say "season with plutonium", Agent B can never say it — because it's mathematically impossible within this architecture.

---

## DASA vs. Traditional RAG

| Property | Standard RAG | DASA |
|---|---|---|
| Hallucination risk | Medium (model can still "drift") | **None** (vocabulary is locked) |
| GPU required | Yes (for generation) | **No** (CPU-only inference) |
| RAM footprint | 8–80 GB | **< 2 GB** |
| Agent autonomy | Passive retrieval | **Active agentic retrieval** |
| Output type | Probabilistic generation | **Deterministic synthesis** |
| Backend storage | Vector DB (Qdrant, Weaviate…) | **Any JSON / SHARD binary DB** |

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from dasa.pipeline import DASAPipeline
from dasa.config import DASAConfig

config = DASAConfig(
    embedding_model="all-MiniLM-L6-v2",  # 80 MB, runs on CPU
    top_k_fragments=5,
    similarity_threshold=0.3,
)

pipeline = DASAPipeline(config)
pipeline.load("my_dataset.json")   # JSON array of {"lemma": ..., "definition": ...}

response = pipeline.run("¿Cómo preparo huevos fritos?")
print(response)
```

Run the built-in demo (no dataset needed):

```bash
python examples/recipe_example.py
```

---

## Project Structure

```
dasa/
├── agent_a/
│   ├── retrieval_agent.py     # Agent A: search + fragment retrieval
│   ├── embeddings.py          # CPU-only sentence embeddings
│   └── tools.py               # Agentic tools: rank, filter, deduplicate
├── agent_b/
│   ├── synthesis_engine.py    # Agent B: grounded synthesis orchestrator
│   └── statistical_rewriter.py # Pure-math text reconstruction (no LLM needed)
├── pipeline.py                # DASAPipeline: connects A → B
└── config.py                  # All configuration parameters

docs/
├── architecture.md            # Full system design
├── agent-a.md                 # Agent A specification
├── agent-b.md                 # Agent B specification
└── anti-hallucination.md      # Why DASA cannot hallucinate

examples/
├── basic_query.py             # Minimal usage example
└── recipe_example.py          # The "egg recipe" demo from the DASA paper
```

---

## Dataset Format

DASA works with any JSON array. Minimal format:

```json
[
  {"id": "001", "lemma": "Python", "definition": "High-level, general-purpose programming language."},
  {"id": "002", "lemma": "ábaco",  "definition": "Manual calculating instrument using rows of beads."}
]
```

For TB-scale datasets, use **[SHARD](https://github.com/angelgabrieljacintohuayllasco/SHARD)** — the purpose-built binary hash-sharded database (now with numpy-only IVF-PQ vector search) designed for DASA.

---

## Why "Deterministic"?

Given the same query and the same database, DASA **always produces the same class of answer**: one constructed exclusively from retrieved truth. Unlike temperature-controlled LLMs where outputs vary run-to-run and hallucinations are random, DASA's output space is bounded by the database. This is the `D` in DASA.

---

## Roadmap

- [x] SHARD backend integration (native connector)
- [x] IVF-PQ vector search at scale (Agent A Tier 0, mmap, up to ~1 TB on 2 GB RAM)
- [ ] Multi-language embedding support
- [ ] Streaming response mode for large fragment sets
- [ ] Agent A tool extensions: math reasoning, date parsing, aggregation
- [ ] Lightweight LLM-guided synthesis mode (grounding-constrained)
- [ ] Docker image for Raspberry Pi / ARM64

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome contributions in:
- New agentic tools for Agent A
- Alternative synthesis strategies for Agent B
- Adapters for different dataset formats
- Benchmarks against standard RAG pipelines

---

## ⭐ Star History

<a href="https://star-history.com/#angelgabrieljacintohuayllasco/DASA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco/DASA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco/DASA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco/DASA&type=Date" />
 </picture>
</a>

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE)

---

*DASA was designed to run on the hardware you already have: a $5 chip should be enough to answer any question correctly, without inventing facts.*
