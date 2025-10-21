import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import pyttsx3

# --- LLM setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# --- Conversation memory ---
memory = ConversationBufferMemory(input_key="input", output_key="output")

# --- Optional: text-to-speech ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # slower, calm pace
engine.setProperty('volume', 0.9)

def mirror_coach(user_input):
    # Load previous memory
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

    # Speak the response (optional)
    engine.say(response)
    engine.runAndWait()

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
    
    txt.submit(respond, [txt, chatbot], [txt, chatbot])

# --- Run app ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)

