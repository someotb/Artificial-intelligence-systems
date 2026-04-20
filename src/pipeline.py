import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import func

STOP = False
DATA_FILE = "../data/drugLibTest_raw.tsv"

data = pd.read_csv(DATA_FILE, sep="\t", engine="python")

while not STOP:
    print("""
    ACTIONS
    1) Info about Data Set
    2) Label Encoding
    3) One Hot Encoding
    4) Fill gaps
    5) Get tone of text feature
    6) Devide to Train and Test parts
    7) Save to file
    8) Exit
    Note: to return to past level enter 'b' to terminal
    """)
    try:
        chos = int(input("Choose what you want to do (1-8): "))
    except ValueError:
        print("Chose numbers 1-8!")
        continue
    except KeyboardInterrupt:
        break

    match chos:
        case 1:
            print("\nInfo about data set:")
            print(data.info())
            print("\nInfo about gaps:")
            print(data.isnull().sum())
            func.check_for_unic(data)
        case 2:
            READY = False
            BACK = False
            feature = ""
            while not READY:
                feature = input("Choose main feature: ")
                if feature == "b":
                    BACK = True
                    break

                if feature in data.columns.to_list():
                    READY = True
                else:
                    print("Invalid name of feature! Abailable features:")
                    for k, mean in enumerate(data.columns):
                        print(f"Feature: {mean}")

            if BACK:
                continue

            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature])
            print(f"Classes: {le.classes_}")
            print(data[feature].head())

        case 3:
            READY = False
            BACK = False
            feature = ""
            while not READY:
                feature = input("Choose feature for OHE: ")

                if feature == "b":
                    BACK = True
                    break

                if feature in data.columns.to_list():
                    READY = True
                else:
                    print("Invalid name of feature! Abailable features:")
                    for k, mean in enumerate(data.columns):
                        print(f"Feature: {mean}")

            if BACK:
                continue

            data = pd.get_dummies(data, columns=[feature])
            print(data.head())

        case 4:
            READY = False
            BACK = False
            feature = ""
            while not READY:
                feature = input("Choose feature to fill gaps: ")

                if feature == "b":
                    BACK = True
                    break

                if feature in data.columns.to_list():
                    READY = True
                else:
                    print("Invalid name of feature! Abailable features:")
                    for k, mean in enumerate(data.columns):
                        print(f"Feature: {mean}")

            if BACK:
                continue

            print("1) Mean  2) Median  3) Mode  4) Custom value")
            method = int(input("Choose method: "))
            match method:
                case 1:
                    data[feature] = data[feature].fillna(data[feature].mean())
                case 2:
                    data[feature] = data[feature].fillna(data[feature].median())
                case 3:
                    mode_value = data[feature].mode()
                    if not mode_value.empty:
                        data[feature] = data[feature].fillna(mode_value.iloc[0])
                case 4:
                    val = input("Enter value: ")
                    data[feature] = data[feature].fillna(val)
            print(f"Gaps left: {data[feature].isna().sum()}")

        case 5:
            READY = False
            BACK = False
            feature = ""
            while not READY:
                feature = input("Choose text feature for tone analysis: ")

                if feature == "b":
                    BACK = True
                    break

                if feature in data.columns.to_list():
                    READY = True
                else:
                    print("Invalid name of feature! Available features:")
                    for mean in data.columns:
                        print(f"Feature: {mean}")

            if BACK:
                continue

            data[feature] = data[feature].apply(func.get_tone)
            print("Done! Example values:")
            print(data[feature].head())

        case 6:
            BACK = False
            READY = False
            target = ""
            test_size = 0.2

            while not READY:
                target = input("Choose target feature (Y): ")
                if target == "b":
                    continue
                if target in data.columns.to_list():
                    READY = True
                else:
                    print("Invalid name of feature! Abailable features:")
                    for k, mean in enumerate(data.columns):
                        print(f"Feature: {mean}")

            READY = False

            while not READY:
                try:
                    test_size = float(input("Enter test size (e.g. 0.2 = 20% test): "))
                    if not 0 < test_size < 1:
                        print("Enter value between 0 and 1!")
                        continue
                    else:
                        READY = True
                except ValueError:
                    print("Enter a number!")
                    continue

            X = data.drop(columns=[target])
            Y = data[target]

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=42
            )

            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            Y_train = pd.Series(Y_train)
            Y_test = pd.Series(Y_test)

            print(f"\nTrain size: {len(X_train)} rows ({(1 - test_size) * 100:.0f}%)")
            print(f"Test size:  {len(X_test)} rows ({test_size * 100:.0f}%)")
            base_file = str(input("Enter base name for test and train files: "))
            X_train.to_csv(f"../data/{base_file}_X_train.csv", index=False)
            X_test.to_csv(f"../data/{base_file}_X_test.csv", index=False)
            Y_train.to_csv(f"../data/{base_file}_Y_train.csv", index=False)
            Y_test.to_csv(f"../data/{base_file}_Y_test.csv", index=False)
            print(
                f"Saved: {base_file}_X_train.csv, {base_file}_X_test.csv, {base_file}_Y_train.csv, {base_file}_Y_test.csv"
            )

        case 7:
            output_file = str(input("Enter output filename: "))
            data.to_csv(f"../data/{output_file}.csv", index=False)
            print(f"Saved to ../data/{output_file}.csv")

        case 8:
            STOP = True

        case _:
            print("Choose 1-8!")
