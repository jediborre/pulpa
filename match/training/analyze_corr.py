import pandas as pd

def compute_correlations():
    print("Computing Q3 correlations...")
    try:
        df_q3 = pd.read_csv("model_outputs_v4/q3_dataset.csv")
        corr_q3 = df_q3.corr(numeric_only=True)["target_q3_home_win"].sort_values(key=abs, ascending=False)
        print("Top 15 Q3 Correlations:")
        print(corr_q3.head(16)[1:])
    except Exception as e:
        print("Error with Q3:", e)

    print("\nComputing Q4 correlations...")
    try:
        df_q4 = pd.read_csv("model_outputs_v4/q4_dataset.csv")
        corr_q4 = df_q4.corr(numeric_only=True)["target_q4_home_win"].sort_values(key=abs, ascending=False)
        print("Top 15 Q4 Correlations:")
        print(corr_q4.head(16)[1:])
    except Exception as e:
        print("Error with Q4:", e)

if __name__ == "__main__":
    compute_correlations()
