import os

from flask import Flask, render_template

application = Flask(__name__, instance_relative_config=True)

SCHOOL_PATH=os.path.join(application.root_path,'school_list.txt')
school_list_file = open(SCHOOL_PATH, "r")
content = school_list_file.read()
school_list_file.close()

@application.route('/')
def show_index():
    return render_template("index.html", content = content)

if __name__ == "__main__":
    application.run(debug=True)
