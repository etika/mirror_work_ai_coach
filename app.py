import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# --- Load OpenAI API key from environment ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# --- LLM setup ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# --- Conversation memory ---
memory = ConversationBufferMemory(input_key="input", output_key="output")

# --- Mirror Work Coach logic ---
def mirror_coach(user_input):
    context = memory.load_memory_variables({}).get("history", "")
    prompt = f"""
You are a calm and compassionate mirror work coach.
Encourage the user to practice self-love and gentle reflection.

Conversation so far:
{context}

User: {user_input}
Coach:
"""
    response = llm.predict(prompt).strip()
    memory.save_context({"input": user_input}, {"output": response})
    return response

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🪞 Mirror Work AI Coach")
    chatbot = gr.Chatbot()
    txt = gr.Textbox(label="Type your message here")

    def respond(msg, chat_history):
        response = mirror_coach(msg)
        chat_history.append((msg, response))
        return "", chat_history

    # Browser handles text-to-speech
    txt.submit(respond, [txt, chatbot], [txt, chatbot], text_to_speech=True)

# --- Dynamic port for Render ---
port = int(os.environ.get("PORT", 8080))
demo.launch(server_name="0.0.0.0", server_port=port)

