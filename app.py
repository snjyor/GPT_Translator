from flask import Flask
from flask_restful import Api
from api_v1 import register_api_v1

app = Flask(__name__)

api = Api(app)


@app.get("/")
def root():
    return "Hello humans, I'm an AI translator"


register_api_v1(api)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
