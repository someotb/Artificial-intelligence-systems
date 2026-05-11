import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

import func

STOP = False
DATA_FILE = "data/drugLibTrain_raw.tsv"

data = pd.read_csv(DATA_FILE, sep="\t", engine="python")

# Словарь для хранения энкодеров и параметров
encoders = {}
fill_params = {}

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
    8) Drop columns
    9) Frequency Encoding
    10) Add text length features
    11) Save encoders/params
    12) Load encoders/params
    13) Process both train and test
    14) Exit
    Note: to return to past level enter 'b' to terminal
    """)
    try:
        chos = int(input("Choose what you want to do (1-14): "))
    except ValueError:
        print("Chose numbers 1-14!")
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
            encoders[f"label_{feature}"] = le
            print(f"Classes: {le.classes_}")
            print(data[feature].head())
            print(f"Encoder saved as 'label_{feature}'")

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
            fill_value = None
            match method:
                case 1:
                    fill_value = data[feature].mean()
                    data[feature] = data[feature].fillna(fill_value)
                case 2:
                    fill_value = data[feature].median()
                    data[feature] = data[feature].fillna(fill_value)
                case 3:
                    mode_value = data[feature].mode()
                    if not mode_value.empty:
                        fill_value = mode_value.iloc[0]
                        data[feature] = data[feature].fillna(fill_value)
                case 4:
                    fill_value = input("Enter value: ")
                    data[feature] = data[feature].fillna(fill_value)

            fill_params[feature] = {"method": method, "value": fill_value}
            print(f"Gaps left: {data[feature].isna().sum()}")
            print(f"Fill params saved for '{feature}'")

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

            X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=42
            )

            X_train_scaled = pd.DataFrame(X_train_scaled)
            X_test_scaled = pd.DataFrame(X_test_scaled)
            Y_train = pd.Series(Y_train)
            Y_test = pd.Series(Y_test)

            print(f"\nTrain size: {len(X_train_scaled)} rows ({(1 - test_size) * 100:.0f}%)")
            print(f"Test size:  {len(X_test_scaled)} rows ({test_size * 100:.0f}%)")
            base_file = str(input("Enter base name for test and train files: "))
            X_train_scaled.to_csv(f"data/{base_file}_X_train.csv", index=False)
            X_test_scaled.to_csv(f"data/{base_file}_X_test.csv", index=False)
            Y_train.to_csv(f"data/{base_file}_Y_train.csv", index=False)
            Y_test.to_csv(f"data/{base_file}_Y_test.csv", index=False)
            print(
                f"Saved: {base_file}_X_train.csv, {base_file}_X_test.csv, {base_file}_Y_train.csv, {base_file}_Y_test.csv"
            )

        case 7:
            output_file = str(input("Enter output filename: "))
            data.to_csv(f"data/{output_file}.csv", index=False)
            print(f"Saved to data/{output_file}.csv")

        case 8:
            print("\nAvailable columns:")
            for idx, col in enumerate(data.columns, 1):
                print(f"{idx}. {col}")

            cols_to_drop = input("\nEnter column names to drop (comma-separated): ")
            if cols_to_drop.strip() and cols_to_drop != "b":
                cols_list = [c.strip() for c in cols_to_drop.split(',')]
                valid_cols = [c for c in cols_list if c in data.columns]

                if valid_cols:
                    data = data.drop(columns=valid_cols)
                    print(f"Dropped: {valid_cols}")
                    print(f"Remaining columns ({len(data.columns)}): {data.columns.tolist()}")
                else:
                    print("No valid columns to drop!")

        case 9:
            READY = False
            BACK = False
            feature = ""
            while not READY:
                feature = input("Choose feature for Frequency Encoding: ")

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

            freq_map = data[feature].value_counts(normalize=True).to_dict()
            new_col_name = f"{feature}_freq"
            data[new_col_name] = data[feature].map(freq_map)
            encoders[f"freq_{feature}"] = freq_map

            print(f"Created new column: {new_col_name}")
            print(f"Frequency map saved as 'freq_{feature}'")
            print(data[[feature, new_col_name]].head())

        case 10:
            READY = False
            BACK = False
            features = ""

            while not READY:
                features = input("Choose text features for length extraction (comma-separated): ")

                if features == "b":
                    BACK = True
                    break

                feature_list = [f.strip() for f in features.split(',')]
                valid_features = [f for f in feature_list if f in data.columns]

                if valid_features:
                    READY = True
                else:
                    print("Invalid feature names! Available features:")
                    for mean in data.columns:
                        print(f"Feature: {mean}")

            if BACK:
                continue

            for feature in valid_features:
                len_col = f"{feature}_length"
                data[len_col] = data[feature].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
                print(f"Created: {len_col}")

            print("\nExample values:")
            print(data[[f"{f}_length" for f in valid_features]].head())

        case 11:
            filename = input("Enter filename to save encoders/params (without extension): ")
            if filename and filename != "b":
                save_data = {
                    "encoders": encoders,
                    "fill_params": fill_params
                }
                with open(f"data/{filename}.pkl", "wb") as f:
                    pickle.dump(save_data, f)
                print(f"Saved encoders and params to data/{filename}.pkl")
                print(f"Encoders: {list(encoders.keys())}")
                print(f"Fill params: {list(fill_params.keys())}")

        case 12:
            filename = input("Enter filename to load encoders/params (without extension): ")
            if filename and filename != "b":
                try:
                    with open(f"data/{filename}.pkl", "rb") as f:
                        save_data = pickle.load(f)
                    encoders = save_data.get("encoders", {})
                    fill_params = save_data.get("fill_params", {})
                    print(f"Loaded encoders and params from data/{filename}.pkl")
                    print(f"Encoders: {list(encoders.keys())}")
                    print(f"Fill params: {list(fill_params.keys())}")
                except FileNotFoundError:
                    print(f"File data/{filename}.pkl not found!")

        case 13:
            print("\n=== AUTOMATED TRAIN & TEST PROCESSING ===")
            print("This will process both train and test files with the same transformations")

            train_file = input("Enter train file name (default: drugLibTrain_raw.tsv): ").strip()
            if not train_file:
                train_file = "drugLibTrain_raw.tsv"

            test_file = input("Enter test file name (default: drugLibTest_raw.tsv): ").strip()
            if not test_file:
                test_file = "drugLibTest_raw.tsv"

            output_prefix = input("Enter output prefix (default: processed): ").strip()
            if not output_prefix:
                output_prefix = "processed"

            try:
                train_data = pd.read_csv(f"data/{train_file}", sep="\t", engine="python")
                test_data = pd.read_csv(f"data/{test_file}", sep="\t", engine="python")
                print(f"\nLoaded train: {train_data.shape}, test: {test_data.shape}")

                if "Unnamed: 0" in train_data.columns:
                    train_data = train_data.drop(columns=["Unnamed: 0"])
                    test_data = test_data.drop(columns=["Unnamed: 0"])
                    print(" Dropped 'Unnamed: 0'")

                if train_data["condition"].isnull().sum() > 0:
                    mode_val = train_data["condition"].mode()[0]
                    train_data["condition"] = train_data["condition"].fillna(mode_val)
                    test_data["condition"] = test_data["condition"].fillna(mode_val)
                    print(f"Filled 'condition' with mode: {mode_val}")

                text_cols = ["benefitsReview", "sideEffectsReview", "commentsReview"]
                for col in text_cols:
                    train_data[col] = train_data[col].fillna("")
                    test_data[col] = test_data[col].fillna("")
                print(f"Filled text columns with empty string")

                for feature in ["effectiveness", "sideEffects"]:
                    le = LabelEncoder()
                    train_data[feature] = le.fit_transform(train_data[feature])
                    test_data[feature] = le.transform(test_data[feature])
                    print(f"Label encoded '{feature}': {le.classes_}")

                for feature in ["condition", "urlDrugName"]:
                    freq_map = train_data[feature].value_counts(normalize=True).to_dict()
                    new_col = f"{feature}_freq"
                    train_data[new_col] = train_data[feature].map(freq_map)
                    test_data[new_col] = test_data[feature].map(freq_map).fillna(0)
                    print(f"Frequency encoded '{feature}' -> '{new_col}'")

                for col in text_cols:
                    sentiment_col = f"{col}_sentiment"
                    train_data[sentiment_col] = train_data[col].apply(func.get_tone)
                    test_data[sentiment_col] = test_data[col].apply(func.get_tone)
                    print(f"Extracted sentiment from '{col}' -> '{sentiment_col}'")

                for col in text_cols:
                    len_col = f"{col}_length"
                    train_data[len_col] = train_data[col].apply(lambda x: len(str(x).split()))
                    test_data[len_col] = test_data[col].apply(lambda x: len(str(x).split()))
                    print(f"Extracted length from '{col}' -> '{len_col}'")

                # Drop original text columns and categorical columns after feature extraction
                cols_to_drop = text_cols + ["condition", "urlDrugName"]
                train_data = train_data.drop(columns=cols_to_drop)
                test_data = test_data.drop(columns=cols_to_drop)
                print(f"✓ Dropped original text/categorical columns: {cols_to_drop}")

                train_output = f"data/{output_prefix}_train.csv"
                test_output = f"data/{output_prefix}_test.csv"

                train_data.to_csv(train_output, index=False)
                test_data.to_csv(test_output, index=False)

                print(f"\nSaved train to: {train_output}")
                print(f"Saved test to: {test_output}")
                print(f"\nFinal shapes - Train: {train_data.shape}, Test: {test_data.shape}")
                print(f"Final columns ({len(train_data.columns)}): {train_data.columns.tolist()}")

            except Exception as e:
                print(f"Error during processing: {e}")

        case 14:
            STOP = True

        case _:
            print("Choose 1-14!")
