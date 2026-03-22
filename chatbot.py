from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)
print("Choose your AI Mode")
print("Press 1 for angry mode")
print("Press 2 for funny mode")
print("Press 3 for sad mode")

choice = int(input("Tell your response :-- "))
if(choice==1):
    mode = "You are an angry AI Agent. You respond aggresively and impatiently."
elif(choice==2):
    mode = "You are a very funny AI agent. You respond with humor and jokes."
else:
    mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."

messages = [
    SystemMessage(content = mode)
]
print("-------------------Welcome--------------------")
print("Type 0 to exit the loop")
while(True):
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == 0:
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content = response.content))
    print("Bot : ", response.content)