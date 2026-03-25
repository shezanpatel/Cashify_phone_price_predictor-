import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

class FinalBcaPredictor:
    def __init__(self, data_path="my_ultimate_10000_data.csv"):
        self.data_path = data_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.X_cols = None
        self.brand_list = [] 
        
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
        except:
            pass

    def perform_eda_charts(self, df):
        print("\n--- [PHASE 1] DATA ANALYSIS VISUALIZATIONS ---")
        plt.figure(figsize=(6, 6))
        df['Condition'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        plt.title("Unit 1.2: Condition Distribution")
        plt.show()

        plt.figure(figsize=(10, 5))
        df.groupby('Brand')['Resale_Price'].mean().sort_values().plot(kind='bar', color='skyblue')
        plt.title("Unit 1.2.1: Avg Price by Brand")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        corr = df[['RAM_GB', 'Storage_GB', 'Age_Months', 'Resale_Price']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Unit 2.2.3: Feature Correlation")
        plt.show()

    def train_system(self):
        try:
            df = pd.read_csv(self.data_path)
            self.brand_list = sorted(df['Brand'].unique())
        except FileNotFoundError:
            print("❌ Error: CSV not found!"); return False

        self.perform_eda_charts(df)

        print("\n--- [PHASE 2] MACHINE LEARNING METRICS (UNIT 3) ---")
        df_encoded = pd.get_dummies(df, columns=['Brand', 'Model', 'Condition'])
        X = df_encoded.drop(['Resale_Price'], axis=1)
        y = df_encoded['Resale_Price']
        self.X_cols = X.columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Actual vs Predicted Graph
        y_pred = self.model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3, color='green')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.title("Unit 3.3: Actual vs Predicted")
        plt.show()

        print(f"✅ R2 Score (Accuracy) : {r2_score(y_test, y_pred):.4f}")
        print(f"✅ MAE (Mean Error)   : ₹{mean_absolute_error(y_test, y_pred):.2f}")
        
        self.is_trained = True
        return True

    def sanitize(self, b_raw, m_raw):
        brand = b_raw.strip().title()
        if brand == "Apple":
            model = m_raw.lower().replace("iphone", "iPhone").strip()
            model = model.replace("pro", "Pro").replace("max", "Max").replace("plus", "Plus")
        else:
            model = m_raw.strip().title()
        return brand, model

    def run_live_predictor(self):
        if not self.is_trained:
            if not self.train_system(): return

        print("\n" + "="*60)
        print(" [PHASE 3] LIVE PRICE PREDICTOR (HIGH ACCURACY)")
        print("="*60)
        
        print("\nSUPPORTED COMPANIES:")
        print(", ".join(self.brand_list))
        print("-" * 60)

        while True:
            try:
                b_in = input("\nEnter Brand: ")
                m_in = input("Enter Model: ")
                brand, model = self.sanitize(b_in, m_in)
                ram = int(input("Enter RAM (GB): "))
                st = int(input("Enter Storage (GB): "))
                age = int(input("Enter Age (Months): "))
                
                # --- NEW: CONDITION SELECTION ---
                print("\nSelect Device Condition:")
                print("1. Flawless (No scratches, like new)")
                print("2. Minor Scratches (Normal wear and tear)")
                print("3. Cracked Screen (Heavy damage)")
                choice = input("Enter Choice (1-3): ")
                
                c_map = {"1": "Flawless", "2": "Minor Scratches", "3": "Cracked Screen"}
                selected_cond = c_map.get(choice, "Minor Scratches")

                def get_price(target_age, target_cond):
                    row = pd.DataFrame([{'Brand':brand, 'Model':model, 'RAM_GB':ram, 'Storage_GB':st, 'Age_Months':target_age, 'Condition':target_cond}])
                    row_encoded = pd.get_dummies(row).reindex(columns=self.X_cols, fill_value=0)
                    return int(self.model.predict(row_encoded)[0])

                # Use the selected condition for current value
                price_now = get_price(age, selected_cond)
                # Use 'Flawless' at age 0 for the original price reference
                price_new = get_price(0, "Flawless") 
                total_depreciation = price_new - price_now
                
                print("\nFINAL VALUATION SUMMARY")
                print("-" * 45)
                print(f"DEVICE      : {brand} {model}")
                print(f"CONDITION   : {selected_cond}")
                print(f"CONFIG      : {ram}GB RAM / {st}GB Storage")
                print("-" * 45)
                print(f"ORIGINAL PRICE     : ₹{price_new:,}")
                print(f"CURRENT VALUE      : ₹{price_now:,}")
                print(f"TOTAL DEPRECIATION : ₹{total_depreciation:,} ({ (total_depreciation/price_new)*100:.1f}%)")
                print("-" * 45)

            except Exception as e:
                print(f"⚠️ Input Error: {e}")

            if input("\nAnalyze another device? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    predictor = FinalBcaPredictor()
    predictor.run_live_predictor()
