# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:17:50 2023

@author: lisar
"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Function to create a line graph showing the variation of percentage of workers with different education levels over the years
def line(df):
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Plotting high school and bachelor's degree percentages over the years
    plt.plot(df['year'], df['high_school'], label="High School Degree")
    plt.plot(df['year'], df['bachelors_degree'], label="Bachelors Degree")

    # Display legends, title, and axis labels
    plt.legend()
    plt.title("Variation of Percentage of workers with different education levels over the years")
    plt.xlabel("Year")
    plt.ylabel("Percentage of workers with education level and ESI coverage")
    
    # Save the plot as an image and display it
    plt.legend()
    plt.savefig("linegraph.png")
    plt.show()

# Function to create a bar graph showing the percentage of workers from different racial backgrounds for a specific period
def bar(df, from_year, to_year):
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Filter data for the specified years and select relevant columns
    data = df[df['year'].between(from_year, to_year)].loc[:, ['year', 'white', 'black', 'hispanic']]
    data = data.set_index('year')

    # Plotting bar graph for racial demographics
    data.plot(kind='bar', ylim=(0, 80))

    # Display title and axis labels, and format x-axis labels
    plt.title("Percentage of workers from different racial backgrounds between {} and {}".format(from_year, to_year))
    plt.xlabel("Year")
    plt.ylabel('Percentage of workers')
    plt.legend()
    plt.xticks(rotation=0)
    
    # Save the plot as an image and display it
    plt.savefig("bargraph.png")
    plt.show()

# Function to create a pie chart comparing the population of men and women for a specific year
def pie(df, year):
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Filter data for the specified year and select men and women columns
    data = df[df['year'] == year].loc[:, ['men', 'women']]

    # Plotting pie chart for men and women population comparison
    plt.pie(data.values.flatten().tolist(), autopct='%1.1f%%')

    # Display legends, title, and save the plot as an image
    plt.legend(data.columns)
    plt.title("Comparison of men and women population in {}".format(year))
    plt.savefig("piegraph.png")
    
    # Show the pie chart
    plt.show()

# Read data from CSV file
df = pd.read_csv("health_insurance_coverage.csv")

# Call functions to generate visualizations
line(df)
bar(df, 2015, 2019)
pie(df, 2019)
