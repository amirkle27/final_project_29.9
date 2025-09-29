#
# from server import app
#

import numpy as np

import dal
from server import app
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import processing_facade
from File_Converter_Factory import FileConverterFactory
from processing_facade import PolynomialFacade, RandomForestClassifierFacade, KNNFacade, ANNFacade, \
    LinearRegressionFacade
from preprocessoring_strategy import PolynomialRegressionPreprocessor
from spacy_nlp import SpacyNLP
import pandas as pd
from pathlib import Path
from processing_facade import LogisticRegressionFacade
from preprocessoring_strategy import LogisticRegressionPreprocessor
from processing_facade import DecisionTreeClassifierFacade





pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)          # רוחב גדול כדי שלא יעשה wrap
pd.set_option('display.expand_frame_repr', False)

def run_classification_model(model, df, new_row, model_name):
    results = model.train_and_evaluate(df, target_col="size")
    prediction = model.predict(new_row)

    if isinstance(prediction, pd.DataFrame) and 'prediction' in prediction.columns:
        pred_value = prediction['prediction'].iloc[0]
    elif isinstance(prediction, (pd.Series, np.ndarray)):
        pred_value = prediction[0]
    else:
        pred_value = str(prediction)

    print(f"\n--- {model_name} ---")
    print(f"Prediction for {'size'}")
    print("With the following values:\n")
    print(f"{new_row}\n")
    print(f"Prediction: {pred_value}")
    print(f"\nAccuracy: {results['accuracy']:.2f}\n")
    print("-"*50)



def run_regression_model(model, df, new_row, model_name, target_col):
    results = model.train_and_evaluate(df, target_col= target_col)
    prediction = model.predict(new_row)


    if isinstance(prediction, pd.DataFrame) and 'prediction' in prediction.columns:
        pred_value = prediction['prediction'].iloc[0]
    elif isinstance(prediction, (pd.Series, np.ndarray)):
        pred_value = prediction[0]
    else:
        pred_value = str(prediction)

    print(f"\n--- {model_name} ---\n")
    print(f"Prediction for {target_col}")
    print("With the following values:\n")
    print(f"{new_row}\n")
    print(f"Prediction: {pred_value}\n")
    print(f"MSE: {results['mse']:.2f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"R² Score: {results['r2']:.2f}\n")
    # print(f"\nAccuracy: {results['accuracy']:.2f}")
    print("-".center(50))





def main():


    # --- סטודנטים ---
    students_path = Path("C:/Users/444/Downloads/student_study_habits.csv")
    converter = FileConverterFactory().get(students_path)
    students_csv = converter.convert_to_csv(students_path)
    df_students = pd.read_csv(students_csv)

    new_row_students = pd.DataFrame([{
        "study_hours_per_week": 0.47651,
        "sleep_hours_per_day": 0.66513,
        "attendance_percentage": 0.718996,
        "assignments_completed": 0.33312,
        "participation_level_Low": 0,
        "participation_level_Medium": 1,
        "internet_access_Yes": 1,
        "parental_education_High School": 0,
        "parental_education_Master's": 1,
        "parental_education_PhD": 0,
        "extracurricular_Yes": 0,
        "part_time_job_Yes": 1
    }])

    print("Linear Regression for Students Grades File")
    run_regression_model(LinearRegressionFacade(), df_students, new_row_students, "Linear Regression",
                         target_col="final_grade")
    print("Polynomial Regression for Students Grades File")
    run_regression_model(PolynomialFacade(PolynomialRegressionPreprocessor(), 2), df_students, new_row_students,
                         "Polynomial Regression", target_col="final_grade")


    # --- בגדים ---
    clothes_path = Path("C:/Users/444/Downloads/clothes_dataset.csv")
    converter = FileConverterFactory().get(clothes_path)
    clothes_csv = converter.convert_to_csv(clothes_path)
    df_clothes = pd.read_csv(clothes_csv)

    new_row_clothes = pd.DataFrame([{
        "weight": 83,
        "age": 32,
        "height": 180,
    }])

    run_classification_model(LogisticRegressionFacade(), df_clothes, new_row_clothes, "Logistic Regression")
    run_classification_model(DecisionTreeClassifierFacade(), df_clothes, new_row_clothes, "Decision Tree Classifier")
    run_classification_model(RandomForestClassifierFacade(), df_clothes, new_row_clothes, "Random Forest Classifier")
    run_classification_model(KNNFacade(), df_clothes, new_row_clothes, "KNN")
    run_classification_model(ANNFacade(), df_clothes, new_row_clothes, "ANN")





    # Tree
    # Regressor
    # DecisionTreeRegressorFacade
    # ✅ Random
    # Forest
    # Regressor
    # RandomForestRegressorFacade
    # ✅ Ridge / Lasso
    # RidgeFacade, LassoFacade
    # ✅ KNN
    # Regressor
    # KNNRegressorFacade
    # # בחר מודל: Logistic Regression
    # preprocessor = LogisticRegressionPreprocessor()
    # model = LogisticRegressionFacade()
    #
    #
    #
    #
    # prediction = model.predict(new_row)
    # print("Logistic Regression prediction:")
    # print()
    # print()
    # print(f"Accuracy: {results['accuracy']}")
    # print("Predicted size:", prediction[0])
    # ######
    #
    # # בחר מDecision Tree Regression
    # preprocessor = LogisticRegressionPreprocessor()
    # model = DecisionTreeClassifierFacade()
    #
    # new_row = pd.DataFrame([{
    #     "weight": 83,
    #     "age": 32,
    #     "height": 180,
    # }])
    #
    # results = model.train_and_evaluate(df, target_col="size")  # שים את שם העמודה הרלוונטית
    #
    # prediction = model.predict(new_row)
    #
    # print("Decision Tree prediction:")
    # print()
    # print()
    # print(f"Accuracy: {results['accuracy']}")
    # print("Predicted size:", prediction[0])


    #print(results)

    # model.plot()
    # model.get_optimal_x('min')

    # הדגמה של NLP
    #sentence = "Taylor Swift performed in Los Angeles on March 3rd, 2023."
    #nlp = SpacyNLP()
    #print(nlp.get_ents(sentence))
    #print(nlp.get_persons(sentence))
    #   print(nlp.get_lemmas(sentence))

if __name__ == "__main__":
    main()


