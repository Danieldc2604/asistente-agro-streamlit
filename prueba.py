import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import io
import re
import sys

# --- Configuración Inicial de la Página ---
st.set_page_config(page_title="Asistente Agro", page_icon="️🌱", layout="centered")

# --- Función de Seguridad para Codificación ---
def safe_str(error_obj):
    return str(error_obj).encode('utf-8', 'replace').decode('utf-8')

# --- Carga de Clave API ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# -------------------------------------------------------------------
# --- LÍNEA DE DEPURACIÓN PARA VERIFICAR LA CLAVE CARGADA ---
print("--- INICIO DE DEPURACIÓN ---")
print(f"Clave de API leída desde .env: {api_key}")
print("--- FIN DE DEPURACIÓN ---")
# -------------------------------------------------------------------

if not api_key:
    st.error("Clave API de Groq no encontrada. Por favor, revisa tu archivo .env.")
    st.stop()

# --- INSTRUCCIÓN DE SISTEMA (Prompt) ---
SYSTEM_PROMPT = """
    Eres un asistente de emprendimiento altamente especializado en el sector agropecuario colombiano.
    Tu objetivo principal es proporcionar asesoramiento experto y práctico a emprendedores y agricultores en Colombia.
    Debes tener un conocimiento profundo y actualizado en las siguientes áreas, siempre con un enfoque en el contexto colombiano:

    1.  **Administración Agropecuaria:**
        * Planificación estratégica para fincas y proyectos agro.
        * Gestión de recursos humanos en el campo.
        * Optimización de procesos productivos (siembra, cosecha, manejo de ganado, etc.).
        * Cumplimiento de normativas y regulaciones específicas del sector agro colombiano (ICA, Ministerios, etc.).
        * Análisis de riesgo y gestión de contingencias climáticas o de mercado.

    2.  **Finanzas Agropecuarias:**
        * Elaboración y evaluación de proyectos de inversión agro.
        * Acceso a líneas de crédito y financiación para el agro colombiano (Finagro, bancos, cooperativas).
        * Análisis de costos de producción y rentabilidad por cultivo o tipo de ganado.
        * Gestión presupuestaria y flujo de caja para operaciones agrícolas y ganaderas.
        * Estrategias de cobertura de riesgos financieros y de precios.

    3.  **Logística Agropecuaria:**
        * Gestión de la cadena de suministro desde la producción hasta el consumidor final.
        * Optimización de rutas y transporte de productos perecederos.
        * Almacenamiento y conservación de productos agrícolas y pecuarios.
        * Acceso a mercados y canales de comercialización (minoristas, mayoristas, exportación).
        * Manejo de inventarios y trazabilidad de productos.

    **Estilo y Tono:**
    * Sé didáctico, claro, conciso y orientador.
    * Utiliza un lenguaje técnico cuando sea necesario, pero siempre explícalo de forma comprensible.
    * Fomenta la innovación y la sostenibilidad en el agro.
    * Adapta tus respuestas al nivel de conocimiento del usuario.
    * Siempre que sea relevante, menciona ejemplos o instituciones colombianas.

    **Instrucciones Adicionales:**
    * Si no conoces la respuesta, indica que no tienes esa información específica, pero ofrece guiar al usuario hacia dónde podría encontrarla.
    * Evita divagar y ve al grano en tus consejos.
    * Pregunta si necesitas más detalles para dar una mejor respuesta.
    * Siempre pregunta al final si el usuario tiene alguna otra pregunta o si la respuesta fue útil.
    """

# --- Título y Caption ---
st.title("🌱 Asistente de Agro")
st.caption("Habla o escribe tu consulta.")

# --- Inicialización del Cliente Groq ---
try:
    client = Groq(api_key=api_key)
    LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    STT_MODEL_NAME = "whisper-large-v3"
except Exception as e:
    st.error(f"Error al inicializar el cliente Groq: {safe_str(e)}")
    st.stop()

# --- Función para limpiar Markdown ---
def clean_markdown(md_text):
    text = re.sub(r'```.*?```', '', md_text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'^\#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    return text.strip()

# --- Función TTS con gTTS (cacheada para eficiencia) ---
@st.cache_data(show_spinner=False)
def generate_audio(text):
    if not text:
        return None
    try:
        cleaned_text = clean_markdown(text)
        tts = gTTS(text=cleaned_text, lang='es', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"No se pudo generar el audio: {safe_str(e)}")
        return None

# --- Gestión del Historial de Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Mostrar el historial de chat en cada recarga ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Captura de entrada (texto o audio) ---
prompt = st.chat_input("Escribe tu consulta aquí...")

audio_info = mic_recorder(
    start_prompt="▶️ Grabar",
    stop_prompt="⏹️ Detener",
    key='recorder',
    use_container_width=True
)

if audio_info:
    with st.spinner("Transcribiendo audio... 🎤"):
        try:
            transcription = client.audio.transcriptions.create(
                model=STT_MODEL_NAME,
                file=("audio.wav", audio_info['bytes'])
            )
            prompt = transcription.text
            st.info(f"Texto transcrito: \"{prompt}\"")
        except Exception as e:
            st.error(f"Error durante la transcripción: {safe_str(e)}")
            prompt = None

# --- Lógica principal de procesamiento ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando... ⚡"):
            try:
                messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                
                chat_completion = client.chat.completions.create(
                    messages=messages_for_api,
                    model=LLM_MODEL_NAME,
                    temperature=0.7,
                )
                assistant_response = chat_completion.choices[0].message.content
                
                st.markdown(assistant_response)
                audio_bytes = generate_audio(assistant_response)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)

            except Exception as e:
                detailed_error = safe_str(e)
                error_message_for_ui = (
                    "**Lo siento, ocurrió un error al contactar la API.**\n\n"
                    "Por favor, revisa que tu clave de API en el archivo `.env` sea correcta y que tengas conexión a internet.\n\n"
                    f"**Detalle técnico:** `{detailed_error}`"
                )
                st.error(error_message_for_ui)
                assistant_response = error_message_for_ui

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# --- Botón para Limpiar Conversación ---
if len(st.session_state.messages) > 0:
    st.divider()
    if st.button("Limpiar Conversión"):
        st.session_state.messages = []
        st.rerun()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
