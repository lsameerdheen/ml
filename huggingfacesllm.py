# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:41:25 2023

@author: lsame
"""


#need to max_token_size=250 somewhere to get all text generated


#from huggingface_hub import InferenceClient
#client = InferenceClient()
#text = f"""
#You should express what you want a model to do by \ 
#providing instructions that are as clear and \ 
#specific as you can possibly make them. \ 
#This will guide the model towards the desired output, \ 
#and reduce the chances of receiving irrelevant \ 
#or incorrect responses. Don't confuse writing a \ 
#clear prompt with writing a short prompt. \ 
#In many cases, longer prompts provide more clarity \ 
#and context for the model, which can lead to \ 
#more detailed and relevant outputs.
#"""

#prompt = f"""
#Summarize the text delimited by triple backticks \ 
#into a single sentence.
#```{text}```
#"""

#data  = client.text_generation(prompt, max_new_tokens=250)
#print(data)

import requests

API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output) 