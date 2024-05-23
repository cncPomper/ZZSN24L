from transformers import pipeline, Conversation
import gradio as gr


def chat(user_input, chat_history):
    if chat_history is None:
        chat_history = []
    conversation = Conversation(user_input)
    for message in chat_history:
        conversation.add_user_input(message['user'])
        conversation.add_bot_response(message['bot'])
    response = chatbot(conversation)
    bot_response = response.generated_responses[-1]
    chat_history.append({'user': user_input, 'bot': bot_response})
    return bot_response, chat_history


if __name__ == "__main__":
    chatbot = pipeline(model="facebook/blenderbot-400M-distill")
    iface = gr.ChatInterface(chat)
    iface.launch()