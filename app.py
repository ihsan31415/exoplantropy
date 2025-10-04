from flask import Flask, render_template

# Initialize the Flask application
app = Flask(__name__)

# Route for the main page (Home)
@app.route('/')
def index():
    return render_template('index.html')

# Route for the About page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the Project page
@app.route('/project')
def project():
    return render_template('project.html')

# This block allows you to run the app directly from the script
if __name__ == '__main__':
    app.run(debug=True)

