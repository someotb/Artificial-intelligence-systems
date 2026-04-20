import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    5) Save to file
    6) Exit
    Note: to return to past level enter 'b' to terminal
    """)
    try:
        chos = int(input("Choose what you want to do (1-5): "))
    except ValueError:
        print("Chose numbers 1-6!")
        continue
    except KeyboardInterrupt:
        break

    match chos:
        case 1:
            print("\nInfo about data set:")
            print(data.info())
            print("\nInfo about gaps:")
            print(data.isnull().sum())
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
            output_file = str(input("Enter output filename: "))
            data.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        case 6:
            STOP = True

        case _:
            print("Choose 1-6!")
