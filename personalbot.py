from flask import Flask,request
from llmold import pipe,string_printer
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"
@app.route("/ans/", methods=['POST', 'GET'])
def getAnswer():
    if request.method == 'POST':
        thought = request.form['thought']
        print(thought)
        output =  pipe(thought, max_new_tokens=200, repetition_penalty=1.2, top_k=100)
        return string_printer(output,'cpu')
    elif request.method == 'GET':    
        thought = request.args.get('thought')
        print(thought)
        output = pipe(thought, max_new_tokens=200, repetition_penalty=1.2, top_k=100)
        return string_printer(output,'cpu')
    else:
        return "HTTP Method not  supported."    