import google.generativeai as genai

genai.configure(api_key="")
p= input("enter prompt")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(p)
print(response.text)


AIzaSyB7aHEwumP-zI8f1TYCxjK_o3deLRxK0Ik