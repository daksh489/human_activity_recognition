# Human Activity Recognition
Our aim is to build a machine learning model to recognize daily human activities using Ambient Sensor data collected from 30 volunteer homes.

# Description
Data are collected continuously while residents perform their normal routines.
Ambient PIR motion sensors, door/temperature sensors, and light switch sensors are placed throughout the home of the volunteer.
The sensors are placed in locations throughout the home that are related to specific activities of daily living that we wish to capture. 

### Data Size:
  Train:  1325468 samples
  Test:    574189 samples

Following are the variables and their description:
1. lastSensorEventHours: Hour of the day
2. lastSensorEventSeconds: Seconds since midnight
3. lastSensorDayOfWeek: Integer day of the week
4. windowDuration: Time duration of the 30 event sliding
window in seconds.
5. timeLastSensorEvent: Seconds since the last sensor event.
6. prevDominantSensor1: Dominant sensor ID from the previous window.
7. prevDominantSensor2: Dominant sensor ID from the second
previous window.
8. lastSensorID: lastSensorID
9. lastSensorLocation: Last sensor location ID in the window.
10. lastMotionLocation: Last motion sensor location ID in the
window, can be -1 if none within the sliding window.
11. complexity: Complexity or entropy in sensor counts.
12. activityChange: Change in activity levels between 2 halves
of the sliding window, bisected temporally.
13. areaTransitions: Number of transitions between major
sensor locations in the window.
14. sensorCount-Location: The weighted count of this sensor,
starting at 1.0 for the most recent event each sensor event
previous is worth n-0.01 the current event.
15. sensorElTime-Location: The number of seconds since this
sensor was last seen, up to a maximum of 86400.

# Our Approach:

### Data Compilation: 
The data for numerical and categorical features are captured separately. To merge the data sets, we have to find the data points which were not captured in the other data set and exclude them from our analysis.
Since de didn’t have any primary key to merge the data sets, we used the date-time columns to find the common data points in both the pool.
It was challenging to do so since the Numerical dataset didn’t have the date column to identify the day and we had to use a logic to extract it using the categorical dataset.
Then only selecting the samples whose date-time counts match in both the datasets.

### Data Processing
We then carried out the standard data pre-processing which includes:
1. Checking for duplicates and missing values
2. Treating outliers of some numerical variables: Limiting the required variables at their 97.5% percentile value.
3. Label encoding the categorical features: The model does not read string formats.
4. MinMax scaling the numerical features: To bring all the variables in one scale.
5. Splitting the data into train, test and validation using stratified sampling to maintain the target proportion.

### EDA
After doing our EDA on the dataset, we have found that certain target variables co-relate to specific areas of the house. For e.g., prepping for meal co-relates to Kitchen sensor count, Washroom activities (Bathing, Toilet, etc) co-relates to Washroom sensor count and so on.
We can also see co-relation of hour of the day vs the activities. For e.g., sleep activity is mostly observed from 10 pm to 10 am and the opposite for working. We can also we prepping for meal and eating, and getting ready and washroom go hand in hand.
We can also see some of the messages from a sensor type is also co-related to specific set of activities. For e.g., change in temperature reading is co-related to sleeping, increase in Light and Motion sensor count is co-related to washroom.
All of the above analysis can be found in the plots in the EDA notebook.

### Machine Learning Algorithms
Following are the ML models we have used to solve our problem:
* Decision Tree: Tree based model would be better given the complexity of the features in data.
* Random Forest: Because its an Horizontal Ensemble of Decision Tree.
* OnevsOne: Used for Multi-label Classification.
* OnevsRest: Used for Multi-label Classification.
* XGBoost: Because its an Vertical Ensemble of Decision Tree.

# CONCLUSION
The amount of data and our multi-classification problem in this project burdened the computational power of our model.
We performed EDA (exploratory data analysis) and base model fitting operations after data stitching and data preprocessing.
Our EDA showed strong correlation between certain features of human activity occurrence and our base model (decision tree) showed 81% accuracy on the test samples.
We used Random Forest algorithm to improve the results and achieved an accuracy of 87%.
However, the models were over-fitting and so we tuned the parameters of Random Forest impacting over accuracy to 83%.
We then tried OneVsOne and OneVsRest approach but achieved similar results.
We then used XGBoost and found the best results. we achieved an accuracy of 96% on train and 90% on test at 2k epoch.
It was computationally heavy to train the model. However we ran the model at 4k epoch and achieved an accuracy of 99% on train and 92% on test.
The model size is double for the one at 4k epoch and the marginal improvement is accuracy is not that much.
If accuracy is our highest priority, we select 4k epoch XGBoost as our final model but we select 2k epoch XGBoost model if are looking for an more computaionally efficient model.
