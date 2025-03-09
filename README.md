# Student Performance Analysis and Improvement System

## Overview
This application is a comprehensive student performance analysis and improvement system built with Streamlit. It processes student performance data, analyzes strengths and weaknesses, recommends personalized learning paths, facilitates peer learning through student pairing, tracks improvements using machine learning, and generates detailed performance reports.

## Features

The application consists of seven main phases:

### 1. Data Loading (Anar1)
- Upload and process student performance data from CSV or Excel files
- Standardizes column names and data formats 
- Displays processed data in an interactive table

### 2. Strength/Weakness Analysis (Anar2)
- Analyzes performance trends across multiple attempts
- Uses machine learning to categorize student performance
- Identifies and categorizes student strengths and weaknesses by subject
- Generates a report showing each student's high, medium, and poor performance areas

### 3. Course Recommendations (Anar3)
- Generates personalized To-Do lists for students based on their performance
- Uses Google's Generative AI (Gemini) to create tailored learning paths
- Maps the appropriate To-Do items to each student based on their performance category
- Saves recommendations to CSV for further use

### 4. Peer Learning Pairs (Anar4)
- Implements the Hungarian algorithm to pair poor and medium performers with strong performers
- Creates optimal knowledge sharing pairs to maximize learning effectiveness
- Generates separate pairing lists for poor and medium performers
- Saves pairing data to CSV files for implementation

### 5. Improvement Tracking (Anar5)
- Uses machine learning to predict and track student improvements
- Analyzes historical data to identify improvement patterns
- Provides overall and subject-wise improvement statistics
- Lists students showing improvement in each subject

### 6. Dashboard (Visualization)
- Provides visual representations of performance data
- Displays charts and graphs for better data interpretation
- Offers insights into overall class performance

### 7. Individual Reports
- Generates personalized performance reports for each student
- Creates downloadable PDF reports with detailed performance analysis
- Includes overall feedback and recommendations

## Requirements

```
streamlit
pandas
seaborn
matplotlib
streamlit_option_menu
scikit-learn
google-generativeai
plotly
scipy
numpy
reportlab
```

## Setup Instructions

1. Clone the repository:
```
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Create a `secrets.toml` file in the `.streamlit` directory with your Google API key:
```toml
api_key = "your_google_api_key_here"
```

4. Run the application:
```
streamlit run app.py
```

## Usage Guide

### Phase 1: Read Data
1. Navigate to "Read Data" in the sidebar
2. Upload a CSV file containing student performance data
3. Review the processed data displayed in the table

### Phase 2: Highlight Strength/Weakness
1. Navigate to "Highlight Strength/Weakness" in the sidebar
2. View trend analysis showing average marks by attempt
3. Examine the machine learning model predictions
4. Review the table showing each candidate's performance categories by subject

### Phase 3: Recommend Courses
1. Navigate to "Recommend Courses" in the sidebar
2. Click "Generate and Assign To-Do Lists" button
3. View the personalized recommendations for each student based on their performance level
4. Download the CSV file for implementation

### Phase 4: Pair Performers
1. Navigate to "Pair Performers" in the sidebar
2. View the automatically generated knowledge sharing pairs for both poor and medium performers
3. Download the CSV files containing the pairs for implementation

### Phase 5: Track Improvements
1. Navigate to "Track Improvements" in the sidebar
2. View overall improvement metrics and subject-wise improvement counts
3. Review the list of students showing improvement in each subject

### Phase 6: Dashboard
1. Navigate to "Dashboard" in the sidebar
2. Explore the various visualizations and insights about student performance

### Phase 7: Student Wise Report
1. Navigate to "Student Wise Report" in the sidebar
2. Select a student from the dropdown menu
3. View the comprehensive performance report
4. Generate and download a PDF report by clicking the "Generate PDF" button

## Notes
- The system requires a Google API key for generating course recommendations
- Future student performance data should be saved as "updated_student_marks.csv" for the improvement tracking phase
- The syllabus dictionary needs to be defined with course content for recommendation generation

## Troubleshooting
- If you encounter issues with data processing, ensure your CSV file follows the expected format
- For API-related errors, verify that your Google API key is correctly configured
- For package-related errors, ensure all dependencies are properly installed
