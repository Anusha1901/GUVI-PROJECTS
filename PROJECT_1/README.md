# LOCAL FOOD WASTE MANAGEMENT SYSTEM 

This is an interactive Streamlit web application designed to manage food donations by connecting food **Providers** with **Receivers** through a centralized platform. The system is powered by a SQLite database and offers full CRUD (Create, Read, Update, Delete) operations for managing:

- Food Providers
- Food Receivers
- Food Listings
- Claims on food donations

---

## 📁 Project Structure
```
├── app.py                                               # Main Streamlit application
├── food_storage.db                                      # SQLite database file
└── README.md                                            # Project documentation
```
## 🔧 Features

### 1. **View Tables**
- View the complete data from any of the four tables:
  - `Providers`
  - `Receivers`
  - `Food_Listings`
  - `Claims`
- Data is fetched live from the database.

### 2. **Add Records**
- Add new entries to each table using a user-friendly form.
- Fields are dynamically generated based on the table schema.

### 3. **Update Records**
- Select an existing record using its primary key.
- Modify only the necessary fields.
- Automatically updates the database.

### 4. **Delete Records**
- Safely delete records from any table.
- Confirmation provided before deletion.

---

## 🗃️ Database Schema

### `Providers`
- `Provider_ID` (Integer)
- `Name` (String)
- `Type` (String)
- `Address` (String)
- `City` (String)
- `Contact` (String)

### `Receivers`
- `Receiver_ID` (Integer)
- `Name` (String)
- `Type` (String)
- `City` (String)
- `Contact` (String)

### `Food_Listings`
- `Food_ID` (Integer)
- `Food_Name` (String)
- `Quantity` (Integer)
- `Expiry_Date` (Date)
- `Provider_ID` (Integer)
- `Provider_Type` (String)
- `Location` (String)
- `Food_Type` (String)
- `Meal_Type` (String)

### `Claims`
- `Claim_ID` (Integer)
- `Food_ID` (Integer)
- `Receiver_ID` (Integer)
- `Status` (String)
- `Timestamp` (Datetime)

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. Then install the required libraries:

```bash
pip install streamlit pandas sqlite3
```

### Run the app

```bash
streamlit run app.py
```


