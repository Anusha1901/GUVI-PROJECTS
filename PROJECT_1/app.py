import streamlit as st
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('food_storage.db')
cursor = conn.cursor()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Project Introduction",
    "View Tables",
    "CRUD Operations",
    "SQL Queries & Visualization",
    "Learner SQL Queries",
    "User Introduction"
])

# Project Introduction
if page == "Project Introduction":
    st.title("Local Food Wastage Management System")
    st.markdown("""
    This project helps manage surplus food and reduce wastage by connecting providers with those in need.

    - **Providers**: Restaurants, households, and businesses list surplus food.
    - **Receivers**: NGOs and individuals claim available food.
    - **Geolocation**: Helps locate nearby food.
    - **SQL Analysis**: Powerful insights using SQL queries.
    """)

# View Tables
elif page == "View Tables":
    st.title("View Tables")
    table = st.selectbox("Select Table", ["Providers", "Receivers", "Food_Listings", "Claims"])
    if st.button("Show Table"):
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        st.dataframe(df)

# CRUD Operations
elif page == "CRUD Operations":
    st.subheader("🛠️ Perform CRUD Operations")
    crud_action = st.selectbox("Choose Action", ["Add", "Update", "Delete"])
    crud_table = st.selectbox("Choose Table", ["Providers", "Receivers", "Food_Listings", "Claims"])

    if crud_action == "Add":
        st.markdown(f"### ➕ Add to {crud_table}")

        if crud_table == "Providers":
            pid = st.number_input("Provider ID", step=1)
            name = st.text_input("Name")
            type_ = st.text_input("Type")
            address = st.text_input("Address")
            city = st.text_input("City")
            contact = st.text_input("Contact")
            if st.button("Add Provider"):
                cursor.execute("INSERT INTO Providers VALUES (?, ?, ?, ?, ?, ?)", (pid, name, type_, address, city, contact))
                conn.commit()
                st.success("Provider added successfully!")

        elif crud_table == "Receivers":
            rid = st.number_input("Receiver ID", step=1)
            name = st.text_input("Name")
            type_ = st.text_input("Type")
            city = st.text_input("City")
            contact = st.text_input("Contact")
            if st.button("Add Receiver"):
                cursor.execute("INSERT INTO Receivers VALUES (?, ?, ?, ?, ?)", (rid, name, type_, city, contact))
                conn.commit()
                st.success("Receiver added successfully!")

        elif crud_table == "Food_Listings":
            fid = st.number_input("Food ID", step=1)
            fname = st.text_input("Food Name")
            qty = st.number_input("Quantity", step=1)
            exp = st.date_input("Expiry Date")
            pid = st.number_input("Provider ID", step=1)
            ptype = st.text_input("Provider Type")
            location = st.text_input("Location")
            ftype = st.text_input("Food Type")
            mtype = st.text_input("Meal Type")
            if st.button("Add Food Listing"):
                cursor.execute("INSERT INTO Food_Listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (fid, fname, qty, exp, pid, ptype, location, ftype, mtype))
                conn.commit()
                st.success("Food Listing added successfully!")

        elif crud_table == "Claims":
            cid = st.number_input("Claim ID", step=1)
            fid = st.number_input("Food ID", step=1)
            rid = st.number_input("Receiver ID", step=1)
            status = st.selectbox("Status", ["Pending", "Completed", "Cancelled"])
            timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)")
            if st.button("Add Claim"):
                cursor.execute("INSERT INTO Claims VALUES (?, ?, ?, ?, ?)", (cid, fid, rid, status, timestamp))
                conn.commit()
                st.success("Claim added successfully!")

    elif crud_action == "Update":
        st.markdown(f"### ✏️ Update a row in {crud_table}")
        st.warning("Note: This is a simple update by ID. You may need to refresh after update.")

        if crud_table == "Providers":
            pid = st.number_input("Enter Provider ID to Update", step=1)
            column = st.selectbox("Column to Update", ["Name", "Type", "Address", "City", "Contact"])
            value = st.text_input("New Value")
            if st.button("Update Provider"):
                cursor.execute(f"UPDATE Providers SET {column} = ? WHERE Provider_ID = ?", (value, pid))
                conn.commit()
                st.success("Provider updated successfully!")

        elif crud_table == "Receivers":
            rid = st.number_input("Enter Receiver ID to Update", step=1)
            column = st.selectbox("Column to Update", ["Name", "Type", "City", "Contact"])
            value = st.text_input("New Value")
            if st.button("Update Receiver"):
                cursor.execute(f"UPDATE Receivers SET {column} = ? WHERE Receiver_ID = ?", (value, rid))
                conn.commit()
                st.success("Receiver updated successfully!")

        elif crud_table == "Food_Listings":
            fid = st.number_input("Enter Food ID to Update", step=1)
            column = st.selectbox("Column to Update", ["Food_Name", "Quantity", "Expiry_Date", "Provider_ID", "Provider_Type", "Location", "Food_Type", "Meal_Type"])
            value = st.text_input("New Value")
            if st.button("Update Food Listing"):
                cursor.execute(f"UPDATE Food_Listings SET {column} = ? WHERE Food_ID = ?", (value, fid))
                conn.commit()
                st.success("Food listing updated successfully!")

        elif crud_table == "Claims":
            cid = st.number_input("Enter Claim ID to Update", step=1)
            column = st.selectbox("Column to Update", ["Food_ID", "Receiver_ID", "Status", "Timestamp"])
            value = st.text_input("New Value")
            if st.button("Update Claim"):
                cursor.execute(f"UPDATE Claims SET {column} = ? WHERE Claim_ID = ?", (value, cid))
                conn.commit()
                st.success("Claim updated successfully!")

    elif crud_action == "Delete":
        st.markdown(f"### ❌ Delete a row from {crud_table}")
        if crud_table == "Providers":
            pid = st.number_input("Enter Provider ID to Delete", step=1)
            if st.button("Delete Provider"):
                cursor.execute("DELETE FROM Providers WHERE Provider_ID = ?", (pid,))
                conn.commit()
                st.success("Provider deleted successfully!")

        elif crud_table == "Receivers":
            rid = st.number_input("Enter Receiver ID to Delete", step=1)
            if st.button("Delete Receiver"):
                cursor.execute("DELETE FROM Receivers WHERE Receiver_ID = ?", (rid,))
                conn.commit()
                st.success("Receiver deleted successfully!")

        elif crud_table == "Food_Listings":
            fid = st.number_input("Enter Food ID to Delete", step=1)
            if st.button("Delete Food Listing"):
                cursor.execute("DELETE FROM Food_Listings WHERE Food_ID = ?", (fid,))
                conn.commit()
                st.success("Food listing deleted successfully!")

        elif crud_table == "Claims":
            cid = st.number_input("Enter Claim ID to Delete", step=1)
            if st.button("Delete Claim"):
                cursor.execute("DELETE FROM Claims WHERE Claim_ID = ?", (cid,))
                conn.commit()
                st.success("Claim deleted successfully!")

# SQL Queries & Visualization
elif page == "SQL Queries & Visualization":
    st.title("Assignment SQL Queries")
    query_question = st.selectbox("Select a question", [
        "1. How many food providers and receivers are there in each city?",
        "2. Which type of food provider contributes the most food?",
        "3. What is the contact information of food providers in a specific city?",
        "4. Which receivers have claimed the most food?",
        "5. What is the total quantity of food available from all providers?",
        "6. Which city has the highest number of food listings?",
        "7. What are the most commonly available food types?",
        "8. How many food claims have been made for each food item?",
        "9. Which provider has had the highest number of successful food claims?",
        "10. What percentage of food claims are completed vs. pending vs. canceled?",
        "11. What is the average quantity of food claimed per receiver?",
        "12. Which meal type is claimed the most?",
        "13. What is the total quantity of food donated by each provider?"
    ])

    if query_question:
            query_map = {
                # SQL queries (fill with actual queries from your notebook)
                "1. How many food providers and receivers are there in each city?": """SELECT 
                City,
                COUNT(CASE WHEN source_table = 'Provider' THEN 1 END) AS Provider_Count,
                COUNT(CASE WHEN source_table = 'Receiver' THEN 1 END) AS Receiver_Count,
                COUNT(*) AS Total_Count
                FROM (
                SELECT City, 'Provider' AS source_table FROM Providers
                UNION ALL
                SELECT City, 'Receiver' AS source_table FROM Receivers
                ) AS combined_data
                GROUP BY City
                ORDER BY City;""",

                "2. Which type of food provider contributes the most food?": """SELECT Provider_Type, COUNT(*) AS Total_Contributions
                FROM Food_Listings
                GROUP BY Provider_Type
                ORDER BY Total_Contributions DESC;""",

                "3. What is the contact information of food providers in a specific city?": """SELECT Name, Contact
                FROM Providers
                WHERE City = 'Zimmermanville';""",

                "4. Which receivers have claimed the most food?": """SELECT r.Name, COUNT(c.Claim_ID) AS Total_Claims
                FROM Claims c
                JOIN Receivers r ON c.Receiver_ID = r.Receiver_ID
                GROUP BY r.Receiver_ID
                ORDER BY Total_Claims DESC
                LIMIT 20;""",

                "5. What is the total quantity of food available from all providers?": """SELECT SUM(Quantity) AS Total_Quantity_Available
                FROM Food_Listings;""",

                "6. Which city has the highest number of food listings?": """SELECT Location AS City, COUNT(*) AS Number_of_Listings
                FROM Food_Listings
                GROUP BY Location
                ORDER BY Number_of_Listings DESC
                LIMIT 1;""",

                "7. What are the most commonly available food types?": """SELECT Food_Type, COUNT(*) AS Count
                FROM Food_Listings
                GROUP BY Food_Type
                ORDER BY Count DESC;""",

                "8. How many food claims have been made for each food item?": """SELECT 
                f.Food_Name,
                f.Food_ID,
                COUNT(c.Claim_ID) AS Total_Claims
                FROM Claims c
                JOIN Food_Listings f ON c.Food_ID = f.Food_ID
                GROUP BY c.Food_ID
                ORDER BY Total_Claims DESC;""",

                "9. Which provider has had the highest number of successful food claims?": """SELECT 
                p.Name AS Provider_Name,
                COUNT(c.Claim_ID) AS Completed_Claims
                FROM Claims c
                JOIN Food_Listings f ON c.Food_ID = f.Food_ID
                JOIN Providers p ON f.Provider_ID = p.Provider_ID
                WHERE c.Status = 'Completed'
                GROUP BY p.Provider_ID
                ORDER BY Completed_Claims DESC
                LIMIT 1;""",

                "10. What percentage of food claims are completed vs. pending vs. canceled?": """SELECT 
                Status,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Claims) AS Percentage
                FROM Claims
                GROUP BY Status;""",

                "11. What is the average quantity of food claimed per receiver?": """SELECT 
                r.Name AS Receiver_Name,
                AVG(f.Quantity) AS Avg_Quantity_Claimed
                FROM Claims c
                JOIN Food_Listings f ON c.Food_ID = f.Food_ID
                JOIN Receivers r ON c.Receiver_ID = r.Receiver_ID
                GROUP BY c.Receiver_ID
                ORDER BY Avg_Quantity_Claimed DESC;""",

                "12. Which meal type is claimed the most?": """SELECT 
                f.Meal_Type,
                COUNT(c.Claim_ID) AS Total_Claims
                FROM Claims c
                JOIN Food_Listings f ON c.Food_ID = f.Food_ID
                GROUP BY f.Meal_Type
                ORDER BY Total_Claims DESC;""",

                "13. What is the total quantity of food donated by each provider?": """SELECT 
                p.Name AS Provider_Name,
                SUM(f.Quantity) AS Total_Quantity_Donated
                FROM Food_Listings f
                JOIN Providers p ON f.Provider_ID = p.Provider_ID
                GROUP BY p.Provider_ID
                ORDER BY Total_Quantity_Donated DESC;""",
                # Add remaining queries...
            }
            query = query_map.get(query_question)
            if query:
                result_df = pd.read_sql_query(query, conn)
                st.dataframe(result_df)

# Learner SQL Queries
elif page == "Learner SQL Queries":
    st.title("Learner SQL Queries")
    learner_query = st.selectbox("Select a learner query", [
        "14. Which top 5 cities have the most completed food claims?",
        "15. Which food items have expired and were not claimed in time?",
        "16. Which food items have never been claimed by any receiver?",
        "17. Which cities have the highest number of food providers?",
        "18. What is the average number of claims made per food item?",
        "19. Which receivers have canceled the most food claims?",
        "20. Which providers have donated more than 100 units of food?",
        "21. What is the average time gap between food claim and its expiry date?",
        "22. Which food items have been claimed multiple times by different receivers?",
        "23. What types of meals are most commonly claimed from a specific provider?"
    ])

    learner_map = {
        # Add SQL queries for learner questions
        "14. Which top 5 cities have the most completed food claims?": """SELECT 
        f.Location AS City,
        COUNT(c.Claim_ID) AS Completed_Claims
        FROM Claims c
        JOIN Food_Listings f ON c.Food_ID = f.Food_ID
        WHERE c.Status = 'Completed'
        GROUP BY f.Location
        ORDER BY Completed_Claims DESC
        LIMIT 5;""",

        "15. Which food items have expired and were not claimed in time?": """SELECT *
        FROM Food_Listings
        WHERE DATE(Expiry_Date) < DATE('now');""",

        "16. Which food items have never been claimed by any receiver?": """SELECT f.Food_ID, f.Food_Name
        FROM Food_Listings f
        LEFT JOIN Claims c ON f.Food_ID = c.Food_ID
        WHERE c.Claim_ID IS NULL;""",

        "17. Which cities have the highest number of food providers?": """SELECT City, COUNT(*) AS Provider_Count
        FROM Providers
        GROUP BY City
        ORDER BY Provider_Count DESC;""",

        "18. What is the average number of claims made per food item?": """SELECT 
        ROUND(AVG(Claim_Count), 2) AS Avg_Claims_Per_Item
        FROM (
        SELECT Food_ID, COUNT(*) AS Claim_Count
        FROM Claims
        GROUP BY Food_ID
        );""",

        "19. Which receivers have canceled the most food claims?": """SELECT 
        r.Name,
        COUNT(*) AS Cancelled_Claims
        FROM Claims c
        JOIN Receivers r ON c.Receiver_ID = r.Receiver_ID
        WHERE c.Status = 'Cancelled'
        GROUP BY c.Receiver_ID
        ORDER BY Cancelled_Claims DESC
        LIMIT 5;""",

        "20. Which providers have donated more than 100 units of food?": """SELECT 
        p.Name, 
        SUM(f.Quantity) AS Total_Donated
        FROM Providers p
        JOIN Food_Listings f ON p.Provider_ID = f.Provider_ID
        GROUP BY p.Provider_ID
        HAVING SUM(f.Quantity) > 100
        ORDER BY Total_Donated DESC;""",

        "21. What is the average time gap between food claim and its expiry date?": """SELECT 
        ROUND(AVG(JULIANDAY(f.Expiry_Date) - JULIANDAY(c.Timestamp)), 2) AS Avg_Days_Before_Expiry
        FROM Claims c
        JOIN Food_Listings f ON c.Food_ID = f.Food_ID
        WHERE c.Status = 'Completed';""",

        "22. Which food items have been claimed multiple times by different receivers?": """SELECT 
        f.Food_Name,
        COUNT(c.Claim_ID) AS Number_of_Claims
        FROM Claims c
        JOIN Food_Listings f ON c.Food_ID = f.Food_ID
        GROUP BY f.Food_ID
        HAVING COUNT(c.Claim_ID) > 1
        ORDER BY Number_of_Claims DESC;""",

        "23. What types of meals are most commonly claimed from a specific provider?": """SELECT 
        p.Name AS Provider_Name,
        f.Meal_Type,
        COUNT(c.Claim_ID) AS Total_Claims
        FROM Claims c
        JOIN Food_Listings f ON c.Food_ID = f.Food_ID
        JOIN Providers p ON f.Provider_ID = p.Provider_ID
        WHERE p.Name = 'Gonzales-Cochran'
        GROUP BY f.Meal_Type
        ORDER BY Total_Claims DESC;""",
        # Add remaining queries...
    }
    query = learner_map.get(learner_query)
    if query:
        df = pd.read_sql_query(query, conn)
        st.dataframe(df)

# User Introduction
elif page == "User Introduction":
    st.title("User Introduction")
    st.markdown("""
    **Anusha Dixit**    
    Background in Computer Science with projects in NLP, SQL, Machine Learning and Data Analysis.  
    Passionate about solving real-world problems using data.  
    """)
