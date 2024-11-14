<h1>GRADEWISE - Student Performance Prediction App</h1>

<h2>Overview</h2>
<p><strong>GRADEWISE</strong> is a web application developed using <strong>Streamlit</strong> and <strong>Scikit-learn</strong> that predicts student performance based on various factors such as study habits, previous academic performance, and lifestyle choices. Users can input these features into the app, and it will predict their performance index, helping them understand how different factors influence their academic outcomes.</p>
<p>This documentation provides an overview of the project, its structure, and how to run it locally. It also explains the model creation process and how the application works.</p>

<h2>Features</h2>
<ul>
  <li><strong>Input Parameters:</strong>
    <ul>
      <li><strong>Hours Studied:</strong> The number of hours the student has studied.</li>
      <li><strong>Previous Scores:</strong> The student’s previous academic scores.</li>
      <li><strong>Sleep Hours:</strong> The amount of sleep the student gets each day.</li>
      <li><strong>Sample Question Papers Practiced:</strong> The number of practice question papers the student has completed.</li>
      <li><strong>Extracurricular Activities (Yes/No):</strong> Whether the student participates in extracurricular activities (encoded as 0 for No and 1 for Yes).</li>
    </ul>
  </li>
  <li><strong>Prediction Output:</strong>
    <ul>
      <li><strong>Predicted Performance Index:</strong> A numerical prediction of the student's performance based on the input features.</li>
    </ul>
  </li>
</ul>
<p>The model is based on <strong>Linear Regression</strong> and uses data to predict how these factors impact student performance.</p>

<h2>Project Structure</h2>
<pre>
gradewise/
│
├── app.py                    # Main Streamlit app for user interface and model training
├── Student_Performance.csv    # Dataset for training the model
└── README.md                  # Project documentation
</pre>

<h3>Breakdown of Files:</h3>
<ul>
  <li><strong>`app.py`:</strong>
    <p>Contains the main logic for the Streamlit app, which collects user input, trains the machine learning model, and makes predictions. The model is trained on the `Student_Performance.csv` dataset and then used to predict performance based on the user inputs.</p>
  </li>
  <li><strong>Student_Performance.csv:</strong>
    <p>A CSV file containing historical data about students' performance, including various features like study hours, previous scores, and participation in extracurricular activities. This data is used for training the model.</p>
  </li>
  <li><strong>README.md:</strong>
    <p>Provides documentation on the project, installation instructions, and an overview of the functionality.</p>
  </li>
</ul>

<h2>Installation</h2>
<p>To run the app locally, follow these steps:</p>

<ol>
  <li><strong>Clone the repository:</strong>
    <pre>git clone https://github.com/yourusername/gradewise.git
cd gradewise</pre>
  </li>
  <li><strong>Create a virtual environment (optional but recommended):</strong>
    <pre>python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`</pre>
  </li>
  <li><strong>Install the required dependencies:</strong>
    <p>The required libraries will be installed via <strong>requirements.txt</strong>. Create the file using the following:</p>
    <pre>
streamlit
scikit-learn
pandas
numpy
</pre>
    <p>Install them using:</p>
    <pre>pip install -r requirements.txt</pre>
  </li>
  <li><strong>Run the app:</strong>
    <pre>streamlit run app.py</pre>
  </li>
  <li><strong>Open your browser:</strong>
    <p>Navigate to <a href="http://localhost:8501" target="_blank">http://localhost:8501</a> to access the application.</p>
  </li>
</ol>

<h2>How It Works</h2>
<p><strong>1. Input Collection:</strong><br>
The app uses sliders and checkboxes to collect user inputs such as hours studied, previous scores, sleep hours, and the number of practice question papers completed. The checkbox for extracurricular activities is converted into a binary value (0 for No, 1 for Yes) and used as input for the model.</p>

<p><strong>2. Model Training:</strong><br>
The model is trained using the <strong>Student_Performance.csv</strong> dataset. The features in this dataset include hours studied, previous scores, sleep hours, and extracurricular activity participation. A <strong>Linear Regression</strong> model is used to predict the performance index based on these features.</p>

<p><strong>3. Prediction:</strong><br>
Once the user provides the input features, the model predicts the student’s performance index. The prediction is displayed on the app alongside the input values.</p>

<p><strong>4. Results Display:</strong><br>
The app outputs a table displaying the input features and the predicted performance index.</p>

<h2>Model Creation Process</h2>

<p><strong>1. Preparing the Data:</strong><br>
The model uses a dataset (<strong>Student_Performance.csv</strong>) that includes various features:
<ul>
  <li><strong>Hours Studied</strong></li>
  <li><strong>Previous Scores</strong></li>
  <li><strong>Sleep Hours</strong></li>
  <li><strong>Sample Question Papers Practiced</strong></li>
  <li><strong>Extracurricular Activities (encoded as 0 for No, 1 for Yes)</strong></li>
</ul>
</p>

<p><strong>2. Model Training:</strong><br>
The model is trained using <strong>Linear Regression</strong>, which predicts a continuous output (performance index) based on the input features. The training process is done in the following steps:</p>
<pre>
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the dataset
dataset = pd.read_csv('Student_Performance.csv')

# Feature and target variable
X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Target (Performance Index)

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)
</pre>

<p><strong>3. Prediction with User Input:</strong><br>
After the model is trained, it can be used to predict the performance index based on user input in the app.</p>

<h2>Contributing</h2>
<p>If you'd like to contribute to this project, feel free to fork the repository and create a pull request. Improvements, bug fixes, and new features are welcome!</p>



<h2>Acknowledgements</h2>
<ul>
  <li><strong>Streamlit</strong> for building the interactive web interface.</li>
  <li><strong>Scikit-learn</strong> for providing machine learning tools.</li>
  <li><strong>Pandas</strong> and <strong>NumPy</strong> for data manipulation and analysis.</li>
  <li>The <strong>Student_Performance.csv</strong> dataset used for model training.</li>
</ul>

<h2>Project Image</h2>
<p>![Project Image]()</p>
