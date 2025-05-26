#A starter file for the HomeMatch application if you want to build your solution in a Python program instead of a notebook. 
import os

from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"