import os
import streamlit as st
from openai.error import AuthenticationError
from PIL import Image
from text_splitting import split_text
from vecstore import VectorStore
from langchain.chains.summarize import load_summarize_chain
from langchain_utils import LangChainClient, split_text


####### Boilerplate code #######
# icon = Image.open(os.path.dirname(__file__) + '/icon.png')
st.set_page_config(
    page_title="GPT Long Text",
    # page_icon=icon,
)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


styl = f"""
<style>
  textarea[aria-label="Response: "] {{
    font-family: 'Consolas', monospace !important;
  }}
  }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

st.title(":art: Talk to Long Texts")

######## OpenAI Credentials ########
openai_api_key = ""

openai_model = st.selectbox(
    "Select the GPT model.", ["gpt-3.5-turbo-16k", "gpt-3.5-turbo"])

os.environ["OPENAI_API_KEY"] = st.text_input(
    "OpenAI API Key:", value="", type="password")

######## Article Loading ########
if 'loaded' not in st.session_state:
    st.session_state['loaded'] = False

article = st.text_area("Article",
                       "",
                       key="article_input")

load_btn_placeholder = st.empty()
btn_str = "Loaded!" if st.session_state['loaded'] else "Load article"
load_article = load_btn_placeholder.button(btn_str, type="primary",
                                           disabled=st.session_state['loaded'])

if load_article:
    load_btn_placeholder.button("Loading...", type="primary", disabled=True)

    lc_client = LangChainClient(
        os.environ['OPENAI_API_KEY'],
        VectorStore(),
        model=openai_model,
    )
    st.session_state['docs'] = split_text(
        article,
        chunk_size=13000 if openai_model == "gpt-3.5-turbo-16k" else 1300,
    )
    # TODO: make static
    lc_client.override_index_w_summary(st.session_state['docs'], verbose=True)
    st.session_state['client'] = lc_client
    st.session_state['loaded'] = True
    load_btn_placeholder.button("Loaded!", type="primary", disabled=True)


######## Question Submission ########
if 'submit_processing' not in st.session_state:
    st.session_state['submit_processing'] = False

prompt = st.text_area("Prompt", key="prompt_input")

submit_btn_placeholder = st.empty()


submit_question = submit_btn_placeholder.button(
    "Submit Question",
    type="primary",
    disabled=not st.session_state['loaded'],
)
if submit_question:
    submit_btn_placeholder.button(
        "Processing...", type="primary", disabled=True)
    if 'client' not in st.session_state:
        st.error(
            "Error: You need to load an article first before submitting a question.")
        st.stop()
    lc_client = st.session_state['client']

    if not prompt:
        st.error("Error: Please write a prompt.")
        st.stop()

    # Prompt must be trimmed of spaces at the beginning and end
    prompt = prompt.strip()

    qa = lc_client.get_chain(chain_type="stuff", verbose=True)

    if os.environ["OPENAI_API_KEY"]:
        openai_api_key = os.environ["OPENAI_API_KEY"]

    response = qa(
        {'question': prompt, 'chat_history': st.session_state['chat_history']}
    )['answer']
    st.session_state['chat_history'].append((prompt, response))
    st.session_state['chat_history'] = st.session_state['chat_history'][-5:]

    st.text_area(label="Response: ",
                 value=response,
                 key="response",)
    submit_btn_placeholder.button(
        "Submit Question", type="primary", disabled=False, key='submit_final')


# TODO: ratelimit retrying not showing up
