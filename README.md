# Sepsis-Classification-ML-Project-with-FAST-API-Integration
This repository houses a machine learning project focused on the early detection and classification of sepsis, and integrating the model into a web application using FAST API.

<p align="center">
  <img src="screenshots/profile_image.jpg"  width="800">
</p>

This project aims to provide a streamlined tool for healthcare professionals to predict sepsis cases quickly and effectively.

## Summary
|     Jupyter Notebook                       | Published Article|    Deployed App on Hugging Face
| -------------                  | -------------    |    -----------------
|[Notebook with code and full analysis](https://github.com/rasmodev/Sepsis-Classification-ML-Project-with-FastAPI-Deployment/blob/main/dev/Sepsis_ML_Prediction_Deployment_With_FastAPI.ipynb)|  [Published Article on Medium](https://medium.com/@rasmowanyama/fastapi-for-machine-learning-deployment-a-beginners-guide-ee74ee41316f) |[Link to working FastAPI](https://rasmodev-sepsis-prediction.hf.space/docs/)

# Repository Contents:
- [Project Overview](#project-overview)
- [Project Setup](#project-setup)
- [Data Fields](#data-fields)
- [Getting Started](#getting-started)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [Contact](#contact)


# Project Overview:
**i. Data Collection and Preprocessing:** I loaded and preprocessed a comprehensive dataset containing clinical and physiological data from patients to train and evaluate the sepsis classification model.

**ii. Machine Learning Model:** I implemented a state-of-the-art machine learning model tailored for sepsis classification. This model has been fine-tuned to achieve high accuracy in detecting sepsis early, which is crucial for timely intervention.

**iii. FAST API Integration:** I've seamlessly integrated the trained machine learning model into a web application using FAST API. This web application allows healthcare professionals to input patient data and receive instant predictions regarding sepsis risk.

**iv. Usage and Deployment:** In this README file, you will find detailed instructions on how to use and deploy this web application, making it user-friendly for both developers and healthcare practitioners.

# Project Setup:
To set up the project environment, follow these steps:

i. Clone the repository:

git clone my_github 

```bash 
https://github.com/rasmodev/Sepsis-Classification-ML-Project-with-FastAPI-Deployment.git
```

ii. Create a virtual environment and install the required dependencies:

- **Windows:**
  ```bash
  python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt
  ```

- **Linux & MacOS:**
  ```bash
  python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
  ```
You can copy each command above and run them in your terminal to easily set up the project environment.

## Data Fields

| Column Name | Data Features | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| ID          | N/A             | Unique identifier for each patient               |
| PRG         | Attribute 1     | Number of pregnancies (applicable only to females) |
| PL          | Attribute 2     | Plasma glucose concentration (mg/dL)             |
| PR          | Attribute 3     | Diastolic blood pressure (mm Hg)                 |
| SK          | Attribute 4     | Triceps skinfold thickness (mm)                  |
| TS          | Attribute 5     | 2-hour serum insulin (mu U/ml)                   |
| M11         | Attribute 6     | Body mass index (BMI) (weight in kg / (height in m)^2) |
| BD2         | Attribute 7     | Diabetes pedigree function (mu U/ml)             |
| Age         | Attribute 8     | Age of the patient (years)                       |
| Insurance   | N/A             | Whether the patient has insurance coverage (1 for Yes, 0 for No) |
| Sepsis      | Target          | Positive: if a patient in ICU will develop sepsis,<br> Negative: otherwise |

"ID" and "Insurance" are marked as "N/A" because they do not contribute to predicting whether a patient in the ICU will develop sepsis. These columns were excluded during feature selection because they do not provide relevant information for the sepsis prediction model.






# Contributing:
Your contributions are welcome to improve the model's performance, add new features, or enhance the web application's usability. Please refer to our contribution guidelines in the repository to get started.

# License:
This project is licensed under the MIT License.

# Acknowledgments:
We would like to thank the open-source community and the healthcare professionals who contributed to the dataset used in this project. Their efforts have made advancements in sepsis detection possible.

Feel free to explore the code, use the web application, and contribute to the project's development. Early sepsis detection can save lives, and together, we can make a difference in healthcare.
