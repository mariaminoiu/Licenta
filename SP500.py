# Pandas is used for data manipulation
import pandas

# Read in data as pandas dataframe and display it
df = pandas.read_csv('S&P500.csv')
print(df)

print('The shape of our features is:', df.shape)

# Descriptive statistics for each column
dsts=df.describe()
print(df.describe())

# Use datetime for dealing with dates
import datetime

# Get years, months, and days
years = df['Year']
months = df['Month']
days = df['Day']

# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

# Set the style
plt.style.use('fivethirtyeight')

# Set up the plotting layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)

# Actual max closing price
ax1.plot(dates, df['Close'])
ax1.set_xlabel(''); ax1.set_ylabel('S&P 500 PRICE'); ax1.set_title('Closing price')

# Lowest price
ax2.plot(dates, df['Low'])
ax2.set_xlabel(''); ax2.set_ylabel('S&P 500 PRICE'); ax2.set_title('Lowest price')

# Highest price
ax3.plot(dates, df['High'])
ax3.set_xlabel('Date'); ax3.set_ylabel('S&P 500 PRICE'); ax3.set_title('Highest price')

# Average price
ax4.plot(dates, df['Average'])
ax4.set_xlabel('Date'); ax4.set_ylabel('S&P 500 PRICE'); ax4.set_title('Average price')

plt.tight_layout(pad=2)


df = pandas.get_dummies(df)
print(df)

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(df['Close'])

# Remove the labels from the df
# axis 1 refers to the columns
df= df.drop('Close', axis = 1)

# Saving df names for later use
df_list = list(df.columns)

# Convert to numpy array
df = np.array(df)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25,
                                                                           random_state = 2889)


print('Training Features Shape:', train_df.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_df.shape)
print('Testing Labels Shape:', test_labels.shape)



# The baseline predictions are the historical averages
baseline_preds = test_df[:, df_list.index('Average')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), ' USD')


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=2889)

# Train the model on training data
rf.fit(train_df, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_df)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), ' USD')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, 
                               min_samples_split = 2, min_samples_leaf = 1)


#----------------------------------------------------------------------------

#Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = df_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png');



# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_df, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = df_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');
#--------------------
# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
df_importances = [(df, round(importance, 2)) for df, importance in zip(df_list, importances)]

# Sort the feature importances by most important first
df_importances = sorted(df_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in df_importances];
#-----------------------
# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [df_list.index('Average'), df_list.index('Average')]
train_important = train_df[:, important_indices]
test_important = test_df[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'USD')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


#-----------------------------------

#

#----------------------------------
# Use datetime for creating date objects for plotting
import datetime
# Dates of training values
months = df[:, df_list.index('Month')]
days = df[:, df_list.index('Day')]
years = df[:, df_list.index('Year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pandas.DataFrame(data = {'date': dates, 'Close': labels})
# Dates of predictions
months = test_df[:, df_list.index('Month')]
days = test_df[:, df_list.index('Day')]
years = test_df[:, df_list.index('Year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pandas.DataFrame(data = {'date': test_dates, 'Average': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['Close'], 'b-', label = 'Close')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['Average'], 'ro', label = 'Predicted')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Price (USD)'); plt.title('Actual and Predicted Values');

#-------------------------------


