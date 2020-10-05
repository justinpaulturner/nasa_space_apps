import os

from flask import Flask, render_template

IMAGE_PATH = os.path.join('Images')

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    #test
    app.config['UPLOAD_FOLDER']=IMAGE_PATH

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    # def hello():
    #     return 'Hello, World!'
    def show_index():
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
        print(full_filename)
        return render_template("index.html", user_image = full_filename)

    return app
