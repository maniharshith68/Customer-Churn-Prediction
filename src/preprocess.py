import pandas as pd

def preprocess_data(input_path, output_path):
    # load data
    df = pd.read_csv(input_path)

    # drop unnecessary columns
    df.drop(['Names', 'Location', 'Company', 'Onboard_date'], axis=1, inplace=True)

    # save processed data
    df.to_csv(output_path, index=False)

    print("Preprocessing complete. File saved to:", output_path)


if __name__ == "__main__":
    preprocess_data('../data/customer_churn.csv',
                    '../data/processed_customer_churn.csv')
