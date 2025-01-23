import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder,StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import json
import seaborn as sns
import plotly.express as px
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
import plotly.graph_objects as go

# Streamlit page configuration
st.set_page_config(page_title="Wipro Project", layout="wide")

# Configure the Gemini API key
api_key = st.secrets["api_key"]
genai.configure(api_key=api_key)

def shorten_subject_name(subject):
    return ''.join([word[0] for word in subject.split()])

# Function to load and process data
def load_and_process_data(uploaded_file):
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)

        # Round and convert 'mark' to integer
        if 'mark' in data.columns:
            data['mark'] = data['mark'].round(0).astype(int)

        # Rename columns for better readability
        column_rename_mapping = {
            'course_name': 'Course Name',
            'course_id': 'Course ID',
            'attempt': 'Attempt ID',
            'candidate_name': 'Candidate Name',
            'candidate_email': 'Candidate Email',
            'mark': 'Marks',
            'grade': 'Grade',
            'Performance_category': 'Performance Category'
        }
        data.rename(columns=column_rename_mapping, inplace=True)

        st.session_state["processed_data"] = data
        st.success("File processed successfully!")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# Function to generate To-Do lists for all courses
def generate_todo_list_all_courses(courses, syllabus_dict):
    combined_todo_list = {}
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    for course in courses:
        if course in syllabus_dict:
            syllabus_text = syllabus_dict[course]

            prompt = f"""
            Course Name: {course}
            Below is the syllabus for this course:

            {syllabus_text}

            Based on the syllabus, generate a To-Do list to help students prepare for the course. 
            Limit the list to a maximum of 3 topics for each level:
            - 'Weak': Give me easier topic to gain knowleadge for basics
            - 'Medium': Intermediate topics for further learning.
            - 'Strong': Advanced topics for in-depth understanding.

            Format the To-Do list like this:
            {{
                '{course}': {{
                    'Weak': ['topic 1', 'topic 2', 'topic 3'],
                    'Medium': ['topic 1', 'topic 2', 'topic 3'],
                    'Strong': ['topic 1', 'topic 2', 'topic 3'],
                }}
            }}
            """
            try:
                response = model.generate_content([prompt])
                response_text = response.text

                # Extract JSON from response
                start_index = response_text.find("{")
                end_index = response_text.rfind("}") + 1
                course_todo_list = json.loads(response_text[start_index:end_index])

                # Merge with the combined To-Do list
                combined_todo_list.update(course_todo_list)

            except Exception as e:
                st.error(f"Error generating To-Do list for course: {course}. Details: {e}")
                continue

    return combined_todo_list

def pair_performers(poor_medium_df, strong_df, category):
    pairs = []
    for course_name in poor_medium_df['Course Name'].unique():
        # Filter performers by course
        poor_medium_course = poor_medium_df[poor_medium_df['Course Name'] == course_name]
        strong_course = strong_df[strong_df['Course Name'] == course_name]        
        if not len(strong_course) or not len(poor_medium_course):
            continue  # Skip if no pairs can be formed
        # Build cost matrix (negative marks difference to maximize improvement)
        cost_matrix = np.zeros((len(poor_medium_course), len(strong_course)))
        for i, (_, poor_row) in enumerate(poor_medium_course.iterrows()):
            for j, (_, strong_row) in enumerate(strong_course.iterrows()):
                cost_matrix[i, j] = -abs(poor_row['Marks'] - strong_row['Marks'])
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Form pairs based on Hungarian output
        for r, c in zip(row_ind, col_ind):
            poor = poor_medium_course.iloc[r]
            strong = strong_course.iloc[c]
            pairs.append({
                'Course Name': course_name,
                f'{category} Performer': f"{poor['Candidate Name']} ({poor['Candidate Email']})",
                'Strong Performer': f"{strong['Candidate Name']} ({strong['Candidate Email']})"
            })
    return pd.DataFrame(pairs)

# Prepare the data for ML
def prepare_data(current_df, future_df):
    # Ensure consistent column names
    column_rename_mapping = {
        'course_name': 'Course Name',
        'course_id': 'Course ID',
        'attempt': 'Attempt ID',
        'candidate_name': 'Candidate Name',
        'candidate_email': 'Candidate Email',
        'mark': 'Marks',
        'grade': 'Grade',
        'Performance_category': 'Performance Category'
    }
    current_df.rename(columns=column_rename_mapping, inplace=True)
    future_df.rename(columns=column_rename_mapping, inplace=True)
    # Combine current and future data, selecting only the last attempt
    merged_df = pd.merge(
        current_df,
        future_df,
        on=["Candidate Name", "Course ID"],
        suffixes=("_current", "_future"),
        how="inner"
    )
    # Keep only the last attempt for each candidate and course
    merged_df = merged_df.loc[merged_df.groupby(["Candidate Name", "Course ID"])["Attempt ID_current"].idxmax()]
    # Define target variable (Improved: 1, Not Improved: 0)
    merged_df["Improved"] = (
        (merged_df["Marks_future"] > merged_df["Marks_current"]) |
        (merged_df["Performance Category_future"] > merged_df["Performance Category_current"])
    ).astype(int)
    return merged_df

def train_and_predict(merged_df):
    # Define features and target
    features = [
        "Marks_current", "Grade_current", "Performance Category_current",
        "Marks_future", "Grade_future", "Performance Category_future"
    ]
    X = merged_df[features]
    y = merged_df["Improved"]
    # Separate categorical and numeric columns
    numeric_cols = ["Marks_current", "Marks_future"]
    categorical_cols = [
        "Grade_current", "Performance Category_current", "Grade_future", "Performance Category_future"
    ]
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[ 
        ("imputer", SimpleImputer(strategy="mean")),  # Replace missing values with mean
        ("scaler", StandardScaler())  # Standardize the numeric features
    ])
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with most frequent category
        ("onehot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode categorical variables
    ])
    # Combine preprocessors into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[ 
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    # Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),  # Apply preprocessing
        ("classifier", RandomForestClassifier(random_state=42))  # Train Random Forest
    ])
    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model_pipeline.fit(X_train, y_train)
    # Predict improvement on the entire dataset
    merged_df["Improved Prediction"] = model_pipeline.predict(X)
    return merged_df, model_pipeline

# Function to generate individual report
def generate_report(student_name, student_data):
    email = student_data['Candidate Email'].iloc[0]
    report = f"### Student Performance Report\n"
    report += f"*Candidate Name:* {student_name}\n\n"
    report += f"*Candidate Email:* {email}\n\n"
    report += "#### Courses Attempted:\n"

    for _, row in student_data.iterrows():
        report += f"- *Course Name:* {row['Course Name']}\n"
        report += f"  - Course ID: {row['Course ID']}\n"
        report += f"  - Attempt: {row['Attempt ID']}\n"
        report += f"  - Marks: {row['Marks']}\n"
        report += f"  - Grade: {row['Grade']}\n"
        report += f"  - Performance Category: {row['Performance Category']}\n\n"

    avg_mark = student_data['Marks'].mean()
    if avg_mark >= 75:
        feedback = "Excellent Performance"
    elif avg_mark >= 50:
        feedback = "Good Performance"
    else:
        feedback = "Needs Improvement"

    report += f"#### Overall Feedback:\n*{feedback}*\n"
    return report

# Function to generate PDF
def generate_pdf(student_name, student_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    # Styles
    title_style = ParagraphStyle(
        name="Title",
        fontName="Times-Roman",
        fontSize=18,
        textColor=colors.HexColor("#4CAF50"),
        leading=20,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        fontName="Times-Roman",
        fontSize=14,
        textColor=colors.HexColor("#000000"),
        leading=18
    )
    normal_style = ParagraphStyle(
        name="Normal",
        fontName="Times-Roman",
        fontSize=12,
        textColor=colors.HexColor("#333333"),
        leading=14
    )

    # Title Section
    elements = [
        Paragraph("Student Performance Report", title_style),
        Spacer(1, 12),
        Paragraph(f"Candidate Name: <b>{student_name}</b>", subtitle_style),
        Paragraph(f"Candidate Email: <b>{student_data['Candidate Email'].iloc[0]}</b>", subtitle_style),
        Spacer(1, 12),
        Paragraph("Courses Attempted:", subtitle_style),
    ]

    # Course Details Table
    course_data = [["Course Name", "Course ID", "Attempt ID", "Marks", "Grade", "Performance Category"]]
    for _, row in student_data.iterrows():
        course_data.append([row["Course Name"], row["Course ID"], row["Attempt ID"], row["Marks"], 
                            row["Grade"], row["Performance Category"]])

    table = Table(course_data, colWidths=[200, 70, 70, 50, 50, 150])  # Adjusted column widths
    table.setStyle(TableStyle([  
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),  # Reduced padding
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9F9F9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)

    # Overall Feedback
    avg_mark = student_data['Marks'].mean()
    if avg_mark >= 75:
        feedback = "Excellent Performance"
    elif avg_mark >= 50:
        feedback = "Good Performance"
    else:
        feedback = "Needs Improvement"
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Overall Feedback: <b>{feedback}</b>", subtitle_style))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("<i>Generated by the Student Performance Report System</i>", normal_style))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


# Sidebar with option menu
with st.sidebar:
    selected_phase = option_menu(
        menu_title="Select Phase",
        options=["Read Data","Highlight Strength/Weakness","Recommend Courses","Pair Performers","Track Improvements","Dashboard","Student Wise Report"],
        icons=["1-circle", "2-circle", "3-circle", "4-circle", "5-circle", "6-circle","7-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Initialize session state for processed data
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None
if "data_with_todo" not in st.session_state:
    st.session_state["data_with_todo"] = None
                            
# Content for each phase
if selected_phase == "Read Data":
    st.title("Phase 1: Anar1")
    st.subheader("Read performance report from multiple format (CSV data and excel data) ")
    st.write("Upload a performance report file to process data for Anar1.")
    # File uploader for Anar1
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv","excel"])
    if uploaded_file is not None:
        with st.spinner("Processing the file..."):
            load_and_process_data(uploaded_file)
    # Display the processed data
    if st.session_state["processed_data"] is not None:
        st.subheader("Processed Data")
        st.dataframe(st.session_state["processed_data"])

elif selected_phase == "Highlight Strength/Weakness":
    st.title("Phase 2: Anar2")
    st.subheader("Highlight the Strength, Weakness of candidates")
    st.write("Highlight the Strength and Weakness of candidates.")

    # Check if processed data is available
    if st.session_state["processed_data"] is not None:
        data = st.session_state["processed_data"]
        # Correlation Heatmap
        label_encoder_grade = LabelEncoder()
        label_encoder_performance = LabelEncoder()
        data['grade_encoded'] = label_encoder_grade.fit_transform(data['Grade'])
        data['performance_encoded'] = label_encoder_performance.fit_transform(data['Performance Category'])
        # Step 2: Trend Analysis
        st.subheader("Trend Analysis")
        mark_by_attempt = data.groupby('Attempt ID')['Marks'].mean()
        st.write("Average Marks by Attempt:")
        st.dataframe(mark_by_attempt)

        # Step 3: Model Training and Evaluation
        st.subheader("Model Training and Evaluation")

        # Encode 'course_id' for model training
        label_encoder_course = LabelEncoder()
        data['course_id_encoded'] = label_encoder_course.fit_transform(data['Course ID'])

        # Define features and target (exclude Candidate Name and Course Name)
        X = data[['Marks', 'Attempt ID', 'course_id_encoded']]
        y = label_encoder_performance.fit_transform(data['Performance Category'])

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest Classifier
        random_forest = RandomForestClassifier(random_state=42)
        random_forest.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred_forest = random_forest.predict(X_test)

        # Add Candidate Name and Course Name to X_test
        X_test['Candidate Name'] = data.loc[X_test.index, 'Candidate Name']
        X_test['Course Name'] = data.loc[X_test.index, 'Course Name']
        X_test['Course id'] = label_encoder_course.inverse_transform(data.loc[X_test.index, 'course_id_encoded'])
        X_test['Predicted Performance'] = label_encoder_performance.inverse_transform(y_pred_forest)

        # Display the updated DataFrame
        st.write("Test features along with predictions, including Candidate Name and Course Name:")
        st.dataframe(X_test[['Candidate Name', 'Course Name','Course id', 'Marks', 'Attempt ID', 'Predicted Performance']])

        # Save updated X_test to CSV
        X_test.to_csv('X_test_with_predictions.csv', index=False)
        # Create a new DataFrame to store the categorized subjects for each candidate
        subject_categories = {
            'Candidate Name': [],
            'High': [],
            'Medium': [],
            'Poor': []
        }
        # Filter the dataset to keep only the highest attempt per candidate per course
        filtered_data_coursewise = data.loc[data.groupby(['Candidate Email', 'Course ID'])['Attempt ID'].idxmax()]

        # Sort the data for better clarity (optional)
        filtered_data_coursewise = filtered_data_coursewise.sort_values(by=['Candidate Email', 'Course ID', 'Attempt ID']).reset_index(drop=True)
        # Iterate through the X_test to categorize subjects
        for _, row in filtered_data_coursewise.iterrows():
            candidate_name = row['Candidate Name']
            predicted_performance = row['Performance Category']
            course_name = row['Course Name']

            # Add the course to the corresponding category
            if predicted_performance == 'High':
                subject_categories['Candidate Name'].append(candidate_name)
                subject_categories['High'].append(course_name)
                subject_categories['Medium'].append('')
                subject_categories['Poor'].append('')
            elif predicted_performance == 'Medium':
                subject_categories['Candidate Name'].append(candidate_name)
                subject_categories['High'].append('')
                subject_categories['Medium'].append(course_name)
                subject_categories['Poor'].append('')
            else:
                subject_categories['Candidate Name'].append(candidate_name)
                subject_categories['High'].append('')
                subject_categories['Medium'].append('')
                subject_categories['Poor'].append(course_name)

        # Create a DataFrame from the dictionary
        subject_category_df = pd.DataFrame(subject_categories)

        # Remove any duplicates, so that each candidate only appears once
        subject_category_df = subject_category_df.groupby('Candidate Name').agg(lambda x: ', '.join(filter(None, x))).reset_index()

        # Display the new table
        st.write("Candidate Performance Categories by Subject:")
        st.dataframe(subject_category_df)

        # Save the new table to CSV
        subject_category_df.to_csv('candidate_performance_by_subject.csv', index=False)
        st.success("Candidate performance by subject saved to 'candidate_performance_by_subject.csv'.")
    else:
        st.info("📥 Please ensure Anar1 data is loaded to proceed.")

# Anar3 Phase Logic
elif selected_phase == "Recommend Courses":
    st.title("Phase 3: Anar3")
    st.subheader("Recommend the To Do Course list for Weak Performers")
    st.write("Generate To-Do Lists for All Courses and Map Them to Students.")

    # Ensure processed data is available
    if st.session_state["processed_data"] is not None:
        data = st.session_state["processed_data"].copy()
        filtered_data_coursewise = data.loc[data.groupby(['Candidate Email', 'Course ID'])['Attempt ID'].idxmin()]
        # Sort the data for better clarity (optional)
        data = filtered_data_coursewise.sort_values(by=['Candidate Email', 'Course ID', 'Attempt ID']).reset_index(drop=True)
        courses = data['Course Name'].unique()      
        # Predefined syllabus for each course
        syllabus_dict = {
            "Engineering Mathematics": """
UNIT I MULTIPLE INTEGRALS

(12 Hrs)

Multiple Integrals, change of order of integration and change of variables in double integrals (Cartesian to polar). Applications: Areas by double integration and volumes by triple integration (Cartesian and polar).

(12 Hrs)

UNIT II LAPLACE TRANSFORMS AND INVERSE LAPLACE TRANSFORMS

Definition, Transforms of elementary functions, properties. Transform of derivatives and integrals. Multiplication by t and division by t. Transform of unit step function, transform of periodic functions. Initial and final value theorems, Methods for determining inverse Laplace Transforms, Convolution theorem, Application to differential equations and integral equations. Evaluation of integrals by Laplace transforms.

UNIT III FOURIER SERIES

(12 Hrs)

Dirichlet's conditions - General Fourier series - Expansion of periodic function into Fourier series - Fourier series for odd and even functions - Hall-range Fourier cosine and sine series - Change of interval - Related problems.

UNIT IV FOURIER TRANSFORMS

(12 Hrs)

Fourier Integral theorem. Fourier transform and its inverse, properties. Fourier sine and cosine transforms, their properties, Convolution and Parseval's identity.

UNIT V Z-TRANSFORMS

(12 Hrs)

Difference equations, basic definition, z-transform - definition, Standard z-transforms, Damping rule, Shifting rule, Initial value and final value theorems and problems, Inverse z-transform. Applications of z-transforms to solve difference equations.

            """,
            "Data Structure and Applications": """
UNIT I BASIC TERMINOLOGIES OF DATA STRUCTURES

Introduction: Basic Terminologies - Elementary Data Organizations. Data Structure Operations: Insertion Deletion-Traversal. Array and its operations. Polynomial Manipulation

UNIT II STACK AND QUEUE OPERATIONS

(9 Hrs)

Stacks and Queues: ADT Stack and its operations. Applications of Stacks: Expression Conversion and evaluation. ADT Queue: Types of Queue - Simple Queue - Circular Queue - Priority Queue - Operations on each type of Queues.

UNIT III LINKED LIST OPERATIONS

(9 Hrs)

Linked Lists: Singly linked lists Representation in memory. Algorithms of several operations: Traversing Searching Insertion - Deletion in linked list. Linked representation of Stack and Queue. Doubly linked list: operations. Circular Linked Lists: operations.
UNIT IV TREES

(9 Hrs)

Trees: Basic Tree Terminologies Different types of Trees: Binary Tree Threaded Binary Tree - Binary Search Tree-Binary Tree Traversals - AVL Tree. Introduction to B-Tree and B+ Tree. Heap-Applications of heap.

(9 Hrs)

UNIT V HASHING AND GRAPHS

Hashing. Hash Table - Hash Function and its characteristics. Graph: Basic Terminologies and Representations - Graph traversal algorithms. Definition - Representation of Graph-Types of graph-Breadth-first traversal- Depth-first traversal-Topological Sort-Bi-connectivity-Cut vertex-Euler circuits-Applications of graphs.


            """,
            "Object Oriented Programming": """
                UNIT I OBJECT ORIENTED PROGRAMMING IN C++

Object Oriented Programming Concepts: Basic Program Construction Data Types Type Conversion - Operators-Key Concepts of Object Oriented Programming. Introduction and Structure of the C++ program- Stream Classes Formatted and Unformatted Data Unformatted Console I/O Operations Bit Fields Manipulators. Decision making statements-jump statement-switch case statement-looping statements

UNIT II CLASSES AND OBJECTS, CONSTRUCTORS AND DESTRUCTORS

Introduction to Classes and Objects Constructors and its Types Overloading Constructors Constructors-Destructors.

(9 Hrs)

Copy

UNIT III FUNCTIONS AND INHERITANCE

Functions: Passing arguments LValues and RValues. Library Functions Inline functions - Friend Functions. Inheritance: Introduction-Types of Inheritance.

(9 Hrs)

UNIT IV POLYMORPHISM AND VIRTUAL FUNCTION

(9 Hrs)

Polymorphism: Compile Time and Run Time Polymorphism. Overloading: Function Overloading and Operator Overloading Overloading Unary Operators - Overloading Binary Operators. Virtual Functions - Abstract Classes.

UNIT V TEMPLATES AND EXCEPTION HANDLING

(9 Hrs)

Generic Functions-Need of Templates - Function Templates-Class Templates. Exception Handling: Need of Exceptions - Keywords - Simple and Multiple Exceptions.

            """,
            "Computer and Communication Networks": """
UNIT I INTRODUCTION
Network Applications Network Hardware and Software OSI TCP/IP model - Example Networks - Internet protocols and standards Connection Oriented Network X25 Frame Relay Guided Transmission Media Wireless Transmission Mobile Telephone System Topologies. Case Study: Simple network communication with corresponding cables Transmission modes

UNIT II DATA LINK LAYER

(9 Hrs)

Framing-Error Detection and Correction-Checksum. DLC services-Sliding window protocols - Flow and HDLCPPP Multiple access protocols Multiplexing Ethernet- IEEE 802.11 Error control IEEE802.16-Bluetooth-RFID

UNIT III NETWORK LAYER

(9 Hrs)

Network layer services Packet Switching - IPV4 Addresses subnetting Routing algorithms. Network layer protocols: RIP-OSPF-BGP-ARP-DHCP-ICMP-IPv4 and IPv6-Mobile IP-Congestion control algorithms-Virtual Networks and Tunnels-Global Internet. Case study-Different routing algorithms to select the network path with its optimum and economical during data transfer Link State routing - Flooding - Distance vector.

UNIT IV TRANSPORT LAYER

(9 Hrs)

Introduction-Transport layer protocol - UDP-Reliable byte stream (TCP)-Connection management-Flow control-Retransmission-TCP Congestion control-Congestion avoidance-Queuing-QoS-Application requirements.

UNIT V APPLICATION LAYER

(9 Hrs) DNS-E-Mail-WWW-Architectural Overview - Dynamic web document and http. Protocols: SSH-SNMP -FTP-SMTP-SONET/SDH-ATM-Telnet-POP.

            """,
            "Computer Programming": """
UNIT I INTRODUCTION TO PYTHON

(9 Hrs)

Structure of Python Program Underlying mechanism of Module Execution Branching and Looping Problem Solving Using Branches and Loops - Functions - Lambda Functions-Lists and Mutability-Problem Solving Using Lists and Functions.

UNIT II SEQUENCE DATATYPES AND OBJECT ORIENTED PROGRAMMING

Sequences Mapping and Sets Dictionaries. Classes: Classes and Instances - Inheritance - Exception Handling-Introduction to Regular Expressions using "re" module.

(9 Hrs)

UNIT III USING NUMPY

(9 Hrs)

Basics of NumPy Computation on NumPy-Aggregations-Computation on Arrays - Comparisons - Masks and Boolean Arrays-Fancy Indexing - Sorting Arrays - Structured Data: NumPy's Structured Array.

(9 Hrs)

UNIT IV DATA MANIPULATION WITH PANDAS

Introduction to Pandas Objects Data indexing and Selection Operating on Data in Pandas Handling Missing Data Hierarchical Indexing Combining Data Sets. Aggregation and Grouping - Pivot Tables - Vectorized String Operations-Working with Time Series - High Performance Pandas - eval() and query().

(9 Hrs)

UNIT V VISUALIZATION WITH MATPLOTLIB

Basic functions of Matplotlib Simple Line Plot Scatter Plot - Density and Contour Plots - Histograms Binnings and Density-Customizing Plot Legends - Colour Bars-Three-Dimensional Plotting in Matplotlib.

            """
        }


        # Generate To-Do lists for all courses
        if st.button("Generate and Assign To-Do Lists"):
            with st.spinner("Generating To-Do Lists..."):
                todo_list = generate_todo_list_all_courses(courses, syllabus_dict)

                if todo_list:
                    st.session_state["todo_list"] = todo_list
                    st.success("To-Do Lists generated successfully!")

                    # Assign To-Do lists to students
                    def assign_todo(course, performance):
                        category_mapping = {
                            "Poor": "Weak",
                            "Medium": "Medium",
                            "High": "Strong"
                        }
                        normalized_performance = category_mapping.get(performance, performance)
                        return todo_list.get(course, {}).get(normalized_performance, [])

                    try:
                        data['To-Do List'] = data.apply(
                            lambda x: assign_todo(x['Course Name'], x['Performance Category']),
                            axis=1
                        )

                        st.subheader("Updated Data with To-Do Lists")
                        st.dataframe(data[['Course Name', 'Candidate Name', 'Performance Category', 'To-Do List']])
                        st.session_state["data_with_todo"] = data
                        # Save updated data to CSV
                        data.to_csv('anar3_final.csv', index=False)
                        st.success("Data with To-Do Lists saved to 'anar3_final.csv'.")
                    except Exception as e:
                        st.error(f"Error mapping To-Do lists to students: {e}")
                else:
                    st.error("Failed to generate To-Do lists. Please check the inputs.")
    else:
        st.info("📥 Please ensure Anar1 data is loaded to proceed.")

elif selected_phase == "Pair Performers":
    st.title("Phase 4: Anar4")
    st.subheader("Find the poor performers in class. Pair them with good performers for Knowledge sharing & performance improvement.")
    if st.session_state.get("processed_data") is not None:
        df = st.session_state["processed_data"]
        # Filter the dataset to keep only the highest attempt per candidate per course
        filtered_data_coursewise = df.loc[df.groupby(['Candidate Email', 'Course ID'])['Attempt ID'].idxmax()]
        df_last_attempt = filtered_data_coursewise.sort_values(by=['Candidate Email', 'Course ID', 'Attempt ID']).reset_index(drop=True)

        # Pair poor performers with strong performers
        poor_performers = df_last_attempt[df_last_attempt['Performance Category'] == 'Poor']
        strong_performers = df_last_attempt[df_last_attempt['Performance Category'] == 'High']
        pairs_poor_df = pair_performers(poor_performers, strong_performers, 'Poor')
        st.session_state["knowledge_sharing_pairs_poor"] = pairs_poor_df
        
        st.subheader("Knowledge Sharing Pairs For Poor Performers:")
        st.dataframe(pairs_poor_df)

        # Save to CSV
        pairs_poor_df.to_csv('knowledge_sharing_pairs_poor.csv', index=False)

        # Pair medium performers with strong performers
        medium_performers = df_last_attempt[df_last_attempt['Performance Category'] == 'Medium']
        pairs_medium_df = pair_performers(medium_performers, strong_performers, 'Medium')
        st.session_state["knowledge_sharing_pairs_medium"] = pairs_medium_df

        st.subheader("Knowledge Sharing Pairs For Medium Performers:")
        st.dataframe(pairs_medium_df)

        # Save to CSV
        pairs_medium_df.to_csv('knowledge_sharing_pairs_medium.csv', index=False)

    else:
        st.info("📥 Please ensure Anar1 data is loaded to proceed.")

elif selected_phase == "Track Improvements":
    st.title("Phase 5: Anar5")
    st.subheader("Highlight data on who is improving in the class using an ML Model.")
    
    # Load current data from session state
    if st.session_state.get("processed_data") is not None:
        current_data = st.session_state["processed_data"]
        
        # Load future data
        future_file_path = "./updated_student_marks.csv"
        try:
            future_data = pd.read_csv(future_file_path)
        except FileNotFoundError:
            st.error(f"Future data file not found at: {future_file_path}. Please check the file path.")
            future_data = None
        
        if future_data is not None:
            # Prepare the data
            merged_data = prepare_data(current_data, future_data)
            
            # Train and predict
            merged_data, trained_model = train_and_predict(merged_data)

            # Compute improvement counts
            overall_improvement_count = merged_data["Improved Prediction"].sum()
            subject_wise_counts = (
                merged_data[merged_data["Improved Prediction"] == 1]
                .groupby("Course Name_current")["Candidate Name"]
                .nunique()
                .reset_index(name="Improved Count")
            )

            # Prepare subject-wise improved student data
            improved_students = (
                merged_data[merged_data["Improved Prediction"] == 1]
                .groupby("Course Name_current")[["Candidate Name", "Candidate Email_future"]]
                .apply(lambda x: x.values.tolist())
                .to_dict()
            )

            # Display overall improvement count
            st.write("### Overall Improvement Count")
            st.metric(label="Total Improved Students", value=overall_improvement_count)

            # Display subject-wise improvement counts
            st.write("### Subject-Wise Improvement Counts")
            st.dataframe(subject_wise_counts, use_container_width=True)

            # Display improved student names and emails for each subject
            st.write("### Improved Students (Subject-Wise):")
            if improved_students:
                for subject, students in improved_students.items():
                    # Create a DataFrame for the improved students of this subject
                    student_table = pd.DataFrame(students, columns=["Candidate Name", "Candidate Email_future"])
                    student_table.rename(columns={"Candidate Email_future": "Candidate Email"}, inplace=True)
                    st.write(f"#### {subject}")
                    st.dataframe(student_table)
            else:
                st.write("No students showed improvement in any subject.")
    else:
        st.info("📥 Please ensure Anar1 data is loaded to proceed.")


elif selected_phase == "Dashboard":
    st.title("Phase 6: Anar6")
    st.subheader("Provide a Dashboard to get an overall class performance.")
    # Add custom CSS for styling
    st.markdown("""
        <style>
        .stMetric-value {
            color: #2E86C1;
            font-weight: bold;
        }
        .main-title {
            text-align: center;
            color: #2ECC71;  /* Green title */
        }
        .section-title {
            color: #5D6D7E;  /* Gray section headers */
        }
        .center-options {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

   # Title and description
    st.markdown('<h1 class="main-title">📊 ClassPulse Dashboard</h1>', unsafe_allow_html=True)

    if st.session_state["processed_data"] is not None:
        # Load CSV into a DataFrame
        df = st.session_state["processed_data"]
        # Filters
        selected_student = st.selectbox("👤 Select a Student", ["All Students"] + list(df["Candidate Name"].unique()))
        
        if selected_student == "All Students":
            student_data = df
        else:
            student_data = df[df["Candidate Name"] == selected_student]

        # Filters for the selected student or all students
        selected_course = st.selectbox("📚 Select a Course", ["All"] + list(student_data["Course Name"].unique()))
        selected_category = st.selectbox("🏷 Select Performance Category", ["All"] + list(student_data["Performance Category"].unique()))
        min_marks, max_marks = st.slider(
            "🎯 Select Marks Range",
            min_value=int(student_data["Marks"].min()),
            max_value=int(student_data["Marks"].max()),
            value=(int(student_data["Marks"].min()), int(student_data["Marks"].max()))
        )

        # Apply filters
        filtered_student_data = student_data
        if selected_course != "All":
            filtered_student_data = filtered_student_data[filtered_student_data["Course Name"] == selected_course]
        if selected_category != "All":
            filtered_student_data = filtered_student_data[filtered_student_data["Performance Category"] == selected_category]
        filtered_student_data = filtered_student_data[(filtered_student_data["Marks"] >= min_marks) & (filtered_student_data["Marks"] <= max_marks)]

        # Display key insights
        st.markdown('<h3 class="section-title">📊 Key Insights</h3>', unsafe_allow_html=True)
        total_candidates = filtered_student_data["Candidate Name"].nunique()
        avg_marks = filtered_student_data["Marks"].mean()

        # Calculate top performance category based on filtered data
        if not filtered_student_data.empty:
            top_performance = filtered_student_data["Performance Category"].value_counts().idxmax()
        else:
            top_performance = "No Data"

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", total_candidates, "👤")
        col2.metric("Average Marks", f"{avg_marks:.2f}" if total_candidates > 0 else "0.00", "📈")
        col3.metric("Top Performance Category", top_performance, "🏆")

        # Display filtered data
        st.markdown('<h3 class="section-title">🔍 Filtered Data</h3>', unsafe_allow_html=True)
        st.dataframe(filtered_student_data, use_container_width=True)

        # Visualization options in the center
        st.markdown('<h3 class="section-title">📊 Visualization Options</h3>', unsafe_allow_html=True)
        visualization_type = st.radio(
            "Choose a Visualization",
            [
                "Marks by Course (Bar Chart)", "Performance Category Distribution (Pie Chart)",
                "Marks Comparison (Radar Chart)", "Marks vs Attempts (Scatter Plot)"
            ],
            horizontal=True
        )

        # Visualizations
        if visualization_type == "Marks by Course (Bar Chart)":
            st.subheader(f"📊 {selected_student}'s Marks by Course" if selected_student != "All Students" else "📊 All Students' Marks by Course")
            student_performance = filtered_student_data.groupby('Course Name').agg({
                'Marks': 'mean',
                'Performance Category': 'first'
            }).reset_index()

            # Round the mean values
            student_performance['Marks'] = student_performance['Marks'].round(2)

            fig = px.bar(
                student_performance, 
                x="Course Name", 
                y="Marks", 
                title=f"Marks by Course" if selected_student == "All Students" else f"Marks by Course for {selected_student}",
                labels={"Course Name": "Course", "Marks": "Average Marks"},
                color="Marks",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Performance Category Distribution (Pie Chart)":
            st.subheader(f"📊 {selected_student}'s Performance Category Distribution" if selected_student != "All Students" else "📊 All Students' Performance Category Distribution")
            category_counts = filtered_student_data["Performance Category"].value_counts()
            fig = px.pie(
                names=category_counts.index,
                values=category_counts.values,
                title=f"Performance Category Distribution" if selected_student == "All Students" else f"Performance Category Distribution for {selected_student}",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Marks Comparison (Radar Chart)":
            st.subheader(f"📊 {selected_student}'s Marks Comparison Across Courses" if selected_student != "All Students" else "📊 All Students' Marks Comparison Across Courses")
            student_performance = filtered_student_data.groupby('Course Name').agg({
                'Marks': 'mean',
                'Performance Category': 'first'
            }).reset_index()

            student_performance['Marks'] = student_performance['Marks'].round(2)  # Round marks

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=student_performance['Marks'],
                theta=student_performance['Course Name'],
                fill='toself',
                name=f"Marks of {selected_student}" if selected_student != "All Students" else "Marks of All Students"
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]  # Assuming marks range from 0 to 100
                    )
                ),
                title=f"Radar Chart of {selected_student}'s Marks" if selected_student != "All Students" else "Radar Chart of All Students' Marks"
            )

            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Marks vs Attempts (Scatter Plot)":
            st.subheader(f"📊 {selected_student}'s Marks vs Attempts" if selected_student != "All Students" else "📊 All Students' Marks vs Attempts")
            student_data_filtered = filtered_student_data
            fig = px.scatter(
                student_data_filtered, 
                x="Attempt ID", 
                y="Marks", 
                title=f"Marks vs Attempts for {selected_student}" if selected_student != "All Students" else "Marks vs Attempts for All Students",
                labels={"Attempt ID": "Attempts", "Marks": "Marks"},
                color="Performance Category",
                color_continuous_scale=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📥 Please ensure Anar1 data is loaded to proceed.")

elif selected_phase == "Student Wise Report":
    # App Header
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Performance Report Generator 🎓</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Ensure data exists in session state
    if st.session_state["processed_data"] is not None:
        data = st.session_state['processed_data']
        data = data.loc[data.groupby(['Candidate Email', 'Course ID'])['Attempt ID'].idxmax()]
        # Select a student
        student_names = data['Candidate Name'].unique()
        selected_student = st.selectbox("Select a Student", student_names)

        # Generate report for the selected student
        student_data = data[data['Candidate Name'] == selected_student]
        report = generate_report(selected_student, student_data)

        # Main Content
        st.markdown(f"<h2 style='text-align: center; color: #333333;'>Performance Report for {selected_student}</h2>", unsafe_allow_html=True)

        st.markdown(f"*Candidate Name:* {selected_student}")
        st.markdown(f"*Candidate Email:* {student_data['Candidate Email'].iloc[0]}")
        # Generate PDF button
        if st.button("📄 Generate PDF"):
            pdf_buffer = generate_pdf(selected_student, student_data)
            st.download_button(
                label="💾 Download Report as PDF",
                data=pdf_buffer,
                file_name=f"{selected_student}_report.pdf",
                mime="application/pdf",
            )
    else:
        st.info("📥 Please ensure Anar1 data is loaded to proceed.")
