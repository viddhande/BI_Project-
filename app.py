import os
import pandas as pd
from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Get the feature names from the model
feature_names = model.feature_names_in_

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            data = {
                'Item Fat Content': request.form['Item_Fat_Content'],
                'Item Identifier': request.form['Item_Identifier'],
                'Item Type': request.form['Item_Type'],
                'Outlet Establishment Year': int(request.form['Outlet_Establishment_Year']),
                'Outlet Identifier': request.form['Outlet_Identifier'],
                'Outlet Location Type': request.form['Outlet_Location_Type'],
                'Outlet Type': request.form['Outlet_Type'],
                'Item Visibility': float(request.form['Item_Visibility']),
                'Item Weight': float(request.form['Item_Weight']),
                'Sales': float(request.form['Sales']),
                'Rating': float(request.form['Rating'])
            }

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame([data])

            # Perform one-hot encoding for categorical variables
            df_encoded = pd.get_dummies(df, drop_first=True)

            # Ensure the DataFrame has the same columns as expected by the model
            for col in feature_names:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0  # Fill missing columns with 0

            # Reorder columns to match the model's expected order
            df_encoded = df_encoded[feature_names]

            # Make a prediction using the model
            prediction = model.predict(df_encoded)

            # Add the prediction to the dataframe
            df['Outlet Size'] = prediction

            # Save the data to Excel
            if not os.path.isfile('BlinkIT Grocery Data.xlsx'):
                df.to_excel('BlinkIT Grocery Data.xlsx', index=False)
            else:
                existing_data = pd.read_excel('BlinkIT Grocery Data.xlsx')
                updated_data = pd.concat([existing_data, df], ignore_index=True)
                updated_data.to_excel('BlinkIT Grocery Data.xlsx', index=False)

            # Return the prediction result to the user
            return render_template('result.html', prediction=prediction[0])

        except Exception as e:
            return f"Error Occurred: {e}"

    # Redirect to start.html instead of index.html
    return render_template('start.html')

# Function to generate analysis charts based on your dataset columns
def generate_analysis_charts(data):
    chart_paths = []

    # Create the 'static' directory if it doesn't exist
    os.makedirs('static', exist_ok=True)

    # 1. Pie chart for 'Item Type' distribution
    plt.figure(figsize=(10, 6))
    item_type_counts = data['Item Type'].value_counts()
    plt.pie(item_type_counts, labels=item_type_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Distribution of Item Types')
    pie_chart_path = os.path.join('static', 'item_type_distribution.png')
    plt.savefig(pie_chart_path)
    plt.close()
    chart_paths.append('item_type_distribution.png')

    # 2. Box plot of 'Sales' by 'Item Type'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Item Type', y='Sales', data=data)
    plt.title('Sales by Item Type')
    plt.xlabel('Item Type')
    plt.ylabel('Sales')
    sales_boxplot_path = os.path.join('static', 'sales_by_item_type_boxplot.png')
    plt.savefig(sales_boxplot_path)
    plt.close()
    chart_paths.append('sales_by_item_type_boxplot.png')

    # 3. Count Plot for 'Outlet Location Type'
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Outlet Location Type', data=data, palette='Set2')
    plt.title('Distribution of Outlet Location Types')
    plt.xlabel('Outlet Location Type')
    plt.ylabel('Count')
    outlet_location_count_path = os.path.join('static', 'outlet_location_type_countplot.png')
    plt.savefig(outlet_location_count_path)
    plt.close()
    chart_paths.append('outlet_location_type_countplot.png')

    # 4. Scatter plot of 'Item Weight' vs 'Sales' colored by 'Outlet Size'
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data['Item Weight'], data['Sales'], c=pd.Categorical(data['Outlet Size']).codes, cmap='coolwarm')
    plt.colorbar(scatter, ticks=[0, 1], label='Outlet Size')
    plt.title('Item Weight vs Sales by Outlet Size')
    plt.xlabel('Item Weight')
    plt.ylabel('Sales')
    weight_sales_scatter_path = os.path.join('static', 'item_weight_vs_sales.png')
    plt.savefig(weight_sales_scatter_path)
    plt.close()
    chart_paths.append('item_weight_vs_sales.png')

    # 5. Histogram for 'Sales' distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Sales'], kde=True, color='skyblue')
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    sales_distribution_path = os.path.join('static', 'sales_distribution.png')
    plt.savefig(sales_distribution_path)
    plt.close()
    chart_paths.append('sales_distribution.png')

    return chart_paths

# Route to render the analysis graphs
@app.route('/analysis')
def analysis():
    try:
        # Load data from the Excel file
        file_path = os.path.abspath('BlinkIT Grocery Data.xlsx')
        data = pd.read_excel(file_path)

        # Ensure necessary columns exist
        expected_columns = ['Item Type', 'Sales', 'Item Weight', 'Outlet Size', 'Outlet Location Type']
        if not all(col in data.columns for col in expected_columns):
            return "Error: Required columns are missing from the data."

        # Generate and save the charts
        chart_paths = generate_analysis_charts(data)

        # Pass the charts to the HTML template
        return render_template('analysis.html', chart_urls=chart_paths)

    except Exception as e:
        return f"Error Occurred: {e}"

@app.route('/index')
def input_form():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
