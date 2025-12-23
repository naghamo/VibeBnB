import pandas as pd

from flask import Flask
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

flask_app = Flask(__name__)

@flask_app.route('/')
def hello_world():
    
    return f'<h1>Hello, World!</h1>'

if __name__ == '__main__':
    flask_app.run()