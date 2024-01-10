#!/usr/bin/env python
# coding: utf-8

"""
Created on Thu Nov  9 21:17:50 2023

@author: lisar
"""
# python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 20})


# Loading the data from csv file
df = pd.read_csv('starbucks.csv')

# Handle missing values
df.fillna(0, inplace=True)
print(df.columns)

vitamin_columns = ['Vitamin A (% DV) ', 'Vitamin C (% DV)', ' Calcium (% DV) ',
                   'Iron (% DV) ']
for col in vitamin_columns:
    df[col] = df[col].str.rstrip('%')  
    df[col] = pd.to_numeric(df[col], errors='coerce')   # Convert to numeric, converting non-numeric entries to Na
df.dropna(subset=vitamin_columns, how='any', inplace=True)

# Creating function to format plot elements
def formatplot_with_border(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    ax.tick_params(direction='in', length=10, width=8)
    ax.title.set_color('#1a237e')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_fontsize(25)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Add border to the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

# To create figure and gridspec
fig = plt.figure(figsize=(40, 30), facecolor='white')
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 0.1], hspace=0.5)


# Title of the main plot
plt.suptitle("Nutritional analysis of Starbucks menu", fontsize=50, fontweight='bold', y=0.98, color='#EB144C')

# Calculate percentage distribution of Beverage categories
categories = df['Beverage_category'].value_counts(normalize=True) * 100

# Filter categories with percentages greater than 5%
filtered_categories = categories[categories > 5]
filtered_categories_index = filtered_categories.index
filtered_categories_values = filtered_categories.values



# Plotting line chart to visualize the total cholesteral by beverage prep
ax1 = plt.subplot(gs[0, 0])
df.groupby('Beverage_prep')['Cholesterol (mg)'].sum().plot(kind='line', ax=ax1, marker='o', color='#7e2aa1', linewidth=2) 
ax1.set_title('Total Cholesterol by Beverage Prep', fontsize=35, pad=20, fontweight='bold', color='#009688')
ax1.set_xlabel('Beverage Prep', fontsize=30)
ax1.set_ylabel('Total Cholesterol (mg)', fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=25)
plt.xticks(rotation=0)
ax1.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)

# creating donut pie chart to illustrate distribuation of beverages
ax2 = plt.subplot(gs[0, 1])
# Identify the index of the highest percentage slice
max_percentage_index = filtered_categories_values.argmax()

# To create an explode array to move the identified slice away
explode = [0.1 if i == max_percentage_index else 0 for i in range(len(filtered_categories_index))]
# Use a different color palette for the pie chart
custom_colors = ['#f47373', '#2ccce4', '#37d67a', '#ff8a65', '#ba68c8', '#f78da7']
# Plot flutter donut pie chart with percentages greater than 13%
wedges, texts, autotexts = ax2.pie(
    filtered_categories_values,
    labels=None,
    colors=custom_colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(width=0.4),  # Increase width for donut effect
    explode=explode
)
# Add white circle in the center to create a flutter donut chart
centre_circle = plt.Circle((0, 0), 0.2, color='white', linewidth=0)
ax2.add_patch(centre_circle)
# Setting aspect ratio to be equal for a circular pie chart
ax2.axis('equal')
ax2.set_title('Distribution of Beverages by Beverage Category (Percentages > 5%)', fontsize=40, pad=20, fontweight='bold', color='#009688')
ax2.tick_params(labelsize=20)
# adjusting the positions of percentage labels inside the pie chart
custom_positions = [(-0.7, 0.5), (-0.5, -0.6), (0.5, -0.6), (0.8, 0.2), (0.5, 0.6), (0.2, 0.8)]

for autotext, position in zip(autotexts, custom_positions):
    autotext.set_position(position)
# Creating custom legend beside the pie chart
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=20) for color in custom_colors]
ax2.legend(legend_patches, filtered_categories_index, loc='upper right',bbox_to_anchor=(1.15, 0.90), fontsize=18)






# A barplot Vitamins by Beverage Category
ax3 = plt.subplot(gs[1, 0])
# Grouping and summing up the vitamin columns for each beverage category
vitamins = df.groupby('Beverage_category')[['Vitamin A (% DV) ', 'Vitamin C (% DV)',' Calcium (% DV) ', 'Iron (% DV) ']].sum()
vitamins.plot(kind='barh', ax=ax3)  # Horizontal bar plot
ax3.set_title('Vitamins by Beverage Category', fontsize=30, pad=20, fontweight='bold', color='#009688')
ax3.set_xlabel('Total Vitamins (% Daily Value)', fontsize=20)
ax3.set_ylabel('Beverage Category', fontsize=20)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.legend(fontsize=20)

# Add space to the left of the graph
ax3.margins(0.2)




# Extract relevant columns for the new plot
df_total_fat_calories = df[['Beverage_category', ' Total Fat (g)', 'Calories']].copy()

# Plot multiple lines for Total Fat and Calories, grouped by Beverage Category
ax4 = plt.subplot(gs[1, 1])
ax4.set_title('Total Fat vs Calories in Beverages (Grouped by Category)', fontsize=30, pad=20, fontweight='bold', color='#009688')
sns.lineplot(x='Calories', y=' Total Fat (g)', hue='Beverage_category', data=df_total_fat_calories, ax=ax4, palette='Set1', linewidth=2, legend='brief', ci=None, marker='o', markersize=10)
ax4.set_xlabel('Calories', fontsize=25)
ax4.set_ylabel(' Total Fat (g)', fontsize=25)
ax4.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left', title='Beverage Category')  # Adjust legend position
ax4.tick_params(labelsize=20)
ax4.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for a creative touch

# Customize background color
ax4.set_facecolor('white')




# Format plots with borders
for ax in [ax1, ax2, ax3, ax4]:
    formatplot_with_border(ax, 'black')

# Description for infographics
ax_text = plt.subplot(gs[2, 0])
ax_text.axis('off')
text_content = (
    "These infographics states about Nutritional analysis of Starbucks menu:\n"
    "• In first graph each line represents a beverage preparation method, showcasing variations in cholesterol levels. According to the graph, Solo and Venti bevrage prep contain the highest cholesterol levels.\n"
    "• In Pie Chart different colors represent different beverage categories,  each showing category's proportion contributing to the total. The graph displays only proportions greater than 5%.\n"
    "• In Horizontal Bar Plot helps compare nutritional content across different beverage categories in terms of vitamins. According to the graph, Classic Espresso Drinks contain the highest calcium value.\n"
    "• Compares trends of Total Fats (in grams) and calories across beverages. According to the graph, The beverages which contain both total fats and calories shows fluctuations in the graph."
)
ax_text.text(0, 3, text_content, ha='left', va='top', fontsize=20, color='black', wrap=True, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))

# for Student and ID
ax_name_id = plt.subplot(gs[2, 1])
ax_name_id.axis('off')
ax_name_id.text(1, 1.7, "Name:Lisarika Kanchumarthi \nStudent ID:22070338 ", ha="left", fontsize=20, color='black', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))


# to adjust the layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# adjusting the layout with margins
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

# Saves the figure
plt.savefig("22070338.png", dpi=300)