import pandas as pd #import pandas library to read from csv in datasheet

def test_prediction():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Load the dataset
    df = pd.read_csv('Datasheet/Housing.csv')

    # Step 1: Preprocess the Data
    # Select the features (independent variables) and the target variable
    X = df.drop(columns=['price'])  # Features
    y = df['price']  # Target variable

    # Convert categorical variables (e.g., 'mainroad', 'guestroom') into binary (0/1) using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Step 2: Split the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Choose a Machine Learning Model (Linear Regression)
    model = LinearRegression()

    # Step 4: Train the Model
    model.fit(X_train, y_train)

    # Step 5: Make Predictions
    y_pred = model.predict(X_test)

    # Step 6: Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

def predict(x_test):
    from sklearn.linear_model import LinearRegression

    # Load the dataset
    df = pd.read_csv('Datasheet/Housing.csv')

    # Step 1: Preprocess the Data
    # Select the features (independent variables) and the target variable
    x = df.drop(columns=['price'])  # Features
    y = df['price']  # Target variable


    x_test = pd.DataFrame([x_test], columns=x.columns)
    # Convert categorical variables (e.g., 'mainroad', 'guestroom') into binary (0/1) using one-hot encoding
    x = pd.get_dummies(x, drop_first=True)
    print(x)
    print(x_test)
    x_test_columns = pd.get_dummies(x_test)  # Get the columns from x_test
    missing_columns = set(x.columns) - set(x_test_columns.columns)
    print(x_test_columns)
    for col in missing_columns:
        x_test_columns[col] = 0  # Add missing columns to x_test with default value 0

    # Make sure the columns are in the same order
    x_test = x_test_columns[x.columns]

    # Step 2: Choose a Machine Learning Model (Linear Regression)
    model = LinearRegression()

    # Step 3: Train the Model
    model.fit(x, y)

    print(x_test)
    print(x)

    # Step 4: Make Predictions
    y_pred = model.predict(x_test)

    print(f"Model price prediction: {y_pred}")




# test_prediction()
predict([8580,4,3,4,"yes","no","no","no","yes",2,"yes","semi-furnished"])