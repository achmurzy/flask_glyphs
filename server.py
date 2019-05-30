from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from config import Config

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

import routes, models

#https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-xxiii-application-programming-interfaces-apis
#"An API is a collection of HTTP routes that are designed as 
#low-level entry points into the application"
@app.route("/")
def hello():
	return "Hello World!"
