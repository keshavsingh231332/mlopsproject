import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# I needed some sample data to train the model on
# It makes random numbers for things like study hours and stuff
def load_data(n=100):

    np.random.seed(42)

    # Let's generate the study hours first
    study_hours = np.random.uniform(0, 10, n)

    # Then attendance
    attendance = np.random.uniform(50, 100, n)

    # Sleep hours
    sleep_hours = np.random.uniform(4, 9, n)

    # Previous scores
    previous_score = np.random.uniform(30, 90, n)

    # Calculate score
    score = (
        1.8 * study_hours
        + 0.7 * attendance
        + 1.0 * sleep_hours
        + 0.5 * previous_score
        + np.random.normal(0, 3, n)
    )

    # Keep score between 0 and 100
    score = np.clip(score, 0, 100)

    # Create dataframe
    data = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "previous_score": previous_score,
        "score": score
    })

    return data


# Train model
def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


# Evaluate model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("------------------")
    print("Mean Absolute Error:", round(mae, 2))
    print("R² Score:", round(r2, 2))


# Take user input
def get_user_input():

    try:

        study_hours = float(
            input("\nEnter study hours per day: ")
        )

        attendance = float(
            input("Enter attendance percentage: ")
        )

        sleep_hours = float(
            input("Enter sleep hours per day: ")
        )

        previous_score = float(
            input("Enter previous exam score: ")
        )

        return (
            study_hours,
            attendance,
            sleep_hours,
            previous_score
        )

    except ValueError:

        print("\nInvalid Input! Please enter numeric values.")

        return None, None, None, None


# Validate input
def validate_input(
    study_hours,
    attendance,
    sleep_hours,
    previous_score
):

    if not (0 <= study_hours <= 24):

        print("Study hours must be between 0 and 24")

        return False

    if not (0 <= attendance <= 100):

        print("Attendance must be between 0 and 100")

        return False

    if not (0 <= sleep_hours <= 24):

        print("Sleep hours must be between 0 and 24")

        return False

    if not (0 <= previous_score <= 100):

        print("Previous score must be between 0 and 100")

        return False

    return True


# Predict score
def predict_score(
    model,
    study_hours,
    attendance,
    sleep_hours,
    previous_score
):

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "previous_score": [previous_score]
    })

    predicted_score = model.predict(input_data)

    predicted_score = np.clip(
        predicted_score[0],
        0,
        100
    )

    print(
        "\nPredicted Score:",
        round(float(predicted_score), 2)
    )


# Main function
def main():

    # Load data
    data = load_data(200)

    # Features and target
    X = data[
        [
            "study_hours",
            "attendance",
            "sleep_hours",
            "previous_score"
        ]
    ]

    y = data["score"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Train on full dataset
    model.fit(X, y)

    # Prediction loop
    while True:

        (
            study_hours,
            attendance,
            sleep_hours,
            previous_score
        ) = get_user_input()

        # Skip invalid input
        if study_hours is None:
            continue

        # Validate
        if validate_input(
            study_hours,
            attendance,
            sleep_hours,
            previous_score
        ):

            predict_score(
                model,
                study_hours,
                attendance,
                sleep_hours,
                previous_score
            )

        # Continue?
        choice = input(
            "\nDo you want to predict your score again? (y/n): "
        )

        if choice.lower() != 'y':

            print("\nExiting program... See you soon")

            break


# Run program
if __name__ == "__main__":
    main()