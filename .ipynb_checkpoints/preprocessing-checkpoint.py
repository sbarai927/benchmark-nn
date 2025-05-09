{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "931daf50-afb9-4b47-a4f2-ee0e6cb7f806",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Preprocessing complete — processed files in data/processed/\n"
     ]
    }
   ],
   "source": [
    "# src/preprocessing.py\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.compose import ColumnTransformer\n",
    "import joblib\n",
    "\n",
    "def load_data(path=\"data/raw/diamonds.csv\"):\n",
    "    df = pd.read_csv(path)\n",
    "    # drop that extra index column if present\n",
    "    if \"Unnamed: 0\" in df.columns:\n",
    "        df = df.drop(columns=[\"Unnamed: 0\"])\n",
    "    return df\n",
    "\n",
    "def clean_and_split(df, test_size=0.2, val_size=0.1, random_state=42):\n",
    "    df = df.drop_duplicates()\n",
    "    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)\n",
    "    train, val = train_test_split(\n",
    "        train_val,\n",
    "        test_size=val_size / (1 - test_size),\n",
    "        random_state=random_state\n",
    "    )\n",
    "    return train, val, test\n",
    "\n",
    "def build_preprocessor(df):\n",
    "    # numeric features include everything float/int; drop the target\n",
    "    num_feats = df.select_dtypes(include=[\"int64\", \"float64\"]).columns.tolist()\n",
    "    if \"total_sales_price\" in num_feats:\n",
    "        num_feats.remove(\"total_sales_price\")\n",
    "\n",
    "    # categorical: all object or category dtype\n",
    "    cat_feats = df.select_dtypes(include=[\"object\", \"category\"]).columns.tolist()\n",
    "\n",
    "    preprocessor = ColumnTransformer([\n",
    "        (\"scale\", StandardScaler(), num_feats),\n",
    "        # use sparse_output=False for sklearn>=1.2\n",
    "        (\"ohe\", OneHotEncoder(sparse_output=False, handle_unknown=\"ignore\"), cat_feats),\n",
    "    ])\n",
    "    return preprocessor\n",
    "\n",
    "def main():\n",
    "    # 1) Load & split\n",
    "    df = load_data()\n",
    "    train, val, test = clean_and_split(df)\n",
    "\n",
    "    # 2) Build & fit preprocessor\n",
    "    preprocessor = build_preprocessor(df)\n",
    "    X_train = preprocessor.fit_transform(train)\n",
    "    y_train = train[\"total_sales_price\"].values\n",
    "\n",
    "    # 3) Transform val/test\n",
    "    X_val   = preprocessor.transform(val)\n",
    "    y_val   = val[\"total_sales_price\"].values\n",
    "    X_test  = preprocessor.transform(test)\n",
    "    y_test  = test[\"total_sales_price\"].values\n",
    "\n",
    "    # 4) Save everything\n",
    "    joblib.dump((X_train, y_train), \"data/processed/train.pkl\")\n",
    "    joblib.dump((X_val,   y_val),   \"data/processed/val.pkl\")\n",
    "    joblib.dump((X_test,  y_test),  \"data/processed/test.pkl\")\n",
    "    print(\"Preprocessing complete — processed files in data/processed/\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12e659e0-3cf2-4f35-ad8c-571aad566597",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
