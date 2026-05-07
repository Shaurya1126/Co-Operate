"""
report_generator.py
────────────────────
Script 2 — Analysis & PDF Report Generation
Reads the parquet file produced by collect_jobs.py, runs NLP skill
extraction, trains a Random Forest classifier, scores the student
against live job postings, and generates a personalized PDF report.

Usage:
    python report_generator.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from collections  import Counter
from datetime     import datetime

from sklearn.ensemble                    import RandomForestClassifier
from sklearn.model_selection             import train_test_split
from sklearn.metrics                     import classification_report
from sklearn.preprocessing               import LabelEncoder
from sklearn.feature_extraction.text     import TfidfVectorizer
from sklearn.feature_extraction          import text as sk_text

from reportlab.lib.pagesizes  import letter
from reportlab.lib            import colors
from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units      import inch
from reportlab.platypus       import (SimpleDocTemplate, Paragraph,
                                       Spacer, Table, TableStyle, HRFlowable)
from reportlab.lib.enums      import TA_CENTER
from reportlab.platypus import Image
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SKILLS = [
    # Programming Languages
    "Python", "SQL", "Java", "JavaScript", "TypeScript", "C++",
    "C#", "Golang", "Rust", "Swift", "Kotlin", "Scala", "MATLAB", "Julia",
    "Ruby", "PHP", "Bash", "Shell", "Perl", "SAS", "SPSS",
    "R programming", "C programming",

    # Web Development
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Django",
    "Flask", "FastAPI", "Spring Boot", "REST API", "GraphQL", "Next.js",
    "Express.js", "Bootstrap", "Tailwind CSS", "WordPress",

    # Data Science & ML
    "Machine Learning", "Deep Learning", "Natural Language Processing",
    "Computer Vision", "Reinforcement Learning", "Statistical Modeling",
    "Predictive Modeling", "Feature Engineering", "Data Mining",
    "Time Series Analysis", "A/B Testing", "Hypothesis Testing",
    "Regression", "Classification", "Clustering", "Neural Networks",
    "Transformer Models", "LLMs", "Generative AI", "Prompt Engineering",

    # ML Frameworks & Libraries
    "scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost",
    "LightGBM", "CatBoost", "Hugging Face", "spaCy", "NLTK",
    "OpenCV", "pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn",
    "Plotly", "Statsmodels",

    # Data Engineering
    "ETL", "Data Pipelines", "Apache Spark", "Apache Kafka", "Hadoop",
    "Airflow", "dbt", "Snowflake", "Databricks", "PySpark",
    "Data Warehousing", "Data Modeling", "Data Cleaning",
    "Data Wrangling", "Web Scraping", "BeautifulSoup", "Scrapy",

    # Databases
    "MySQL", "PostgreSQL", "SQLite", "MongoDB", "Redis", "Cassandra",
    "DynamoDB", "Firebase", "Oracle", "Microsoft SQL Server",
    "NoSQL", "Elasticsearch", "Neo4j",

    # Cloud & DevOps
    "AWS", "Azure", "Google Cloud", "GCP", "Docker", "Kubernetes",
    "Terraform", "CI/CD", "Jenkins", "GitHub Actions", "Linux",
    "Unix", "Bash Scripting", "Ansible", "Helm", "Microservices",
    "Serverless", "CloudFormation", "EC2", "S3", "Lambda",

    # Data Visualization & BI
    "Tableau", "Power BI", "Looker", "Excel", "Google Sheets",
    "D3.js", "Grafana", "Kibana", "QlikView",

    # Version Control & Collaboration
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
    "Trello", "Asana", "Notion",

    # Software Engineering Practices
    "Agile", "Scrum", "Kanban", "DevOps", "Test-Driven Development",
    "Unit Testing", "Integration Testing", "Object-Oriented Programming",
    "Functional Programming", "Design Patterns", "System Design",
    "API Development", "Code Review",

    # Cybersecurity
    "Network Security", "Penetration Testing", "Ethical Hacking",
    "Vulnerability Assessment", "Cryptography", "Firewalls", "IAM",
    "OWASP", "Zero Trust", "SOC", "SIEM",

    # Mobile Development
    "Android", "iOS", "React Native", "Flutter", "Xcode", "Android Studio",

    # Mathematics & Statistics
    "Linear Algebra", "Calculus", "Probability", "Statistics",
    "Bayesian Inference", "Optimization", "Graph Theory",
    "Discrete Mathematics", "Numerical Methods",

    # Business & Soft Skills
    "Communication", "Teamwork", "Problem Solving", "Critical Thinking",
    "Project Management", "Leadership", "Presentation Skills",
    "Stakeholder Management", "Time Management", "Adaptability",
    "Attention to Detail", "Cross-functional Collaboration",

    # Domain Specific
    "Financial Modeling", "Accounting", "Supply Chain",
    "Operations Research", "Marketing Analytics", "SEO", "CRM",
    "Salesforce", "SAP", "AutoCAD", "SolidWorks", "ANSYS", "Simulink",
    "Embedded Systems", "FPGA", "IoT", "Bioinformatics",

    # Certifications
    "AWS Certified", "Google Analytics", "PMP",
    "Tableau Certified", "Microsoft Certified", "CompTIA", "CFA",
]

SKILL_ALIASES = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "postgres": "PostgreSQL",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "py": "Python",
    "go": "Golang",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "rl": "Reinforcement Learning",
    "llm": "LLMs",
    "gen ai": "Generative AI",
    "sklearn": "scikit-learn",
    "tf": "TensorFlow",
    "hf": "Hugging Face",
    "k8s": "Kubernetes",
    "gcp": "Google Cloud",
    "gha": "GitHub Actions",
    "r": "R programming",
    "c": "C programming",
    "oop": "Object-Oriented Programming",
    "tdd": "Test-Driven Development",
    "fp": "Functional Programming",
}

SOFT_SKILLS = [
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Critical Thinking", "Project Management", "Attention to Detail",
    "Time Management", "Adaptability", "Presentation Skills",
    "Stakeholder Management", "Cross-functional Collaboration",
]

DEGREE_ROLE_MAP = {
    "Computer Science":        ["Software Engineering", "Data Science",
                                "Data Analytics", "DevOps/Cloud",
                                "Cybersecurity", "Product/Project Management"],
    "Software Engineering":    ["Software Engineering", "DevOps/Cloud",
                                "Data Science", "Cybersecurity"],
    "Data Science":            ["Data Science", "Data Analytics",
                                "Software Engineering", "Research/Science"],
    "Electrical Engineering":  ["Engineering (Non-CS)", "Software Engineering",
                                "DevOps/Cloud"],
    "Mechanical Engineering":  ["Engineering (Non-CS)"],
    "Business Administration": ["Finance/Accounting", "Consulting",
                                "Operations/Business", "Data Analytics",
                                "Product/Project Management"],
    "Accounting":              ["Finance/Accounting", "Consulting"],
    "Finance":                 ["Finance/Accounting", "Consulting",
                                "Data Analytics"],
    "Mathematics":             ["Data Science", "Data Analytics",
                                "Research/Science", "Finance/Accounting"],
    "Statistics":              ["Data Science", "Data Analytics",
                                "Research/Science"],
    "Information Technology":  ["Software Engineering", "DevOps/Cloud",
                                "Data Analytics", "Cybersecurity"],
    "Biology":                 ["Research/Science", "Operations/Business",
                                "Data Analytics"],
    "Chemistry":               ["Research/Science", "Engineering (Non-CS)",
                                "Operations/Business"],
    "Biochemistry":            ["Research/Science", "Data Analytics",
                                "Operations/Business"],
    "Neuroscience":            ["Research/Science", "Data Analytics",
                                "Software Engineering"],
    "Physics":                 ["Research/Science", "Engineering (Non-CS)",
                                "Software Engineering", "Data Science"],
    "Environmental Science":   ["Research/Science", "Operations/Business",
                                "Data Analytics", "Engineering (Non-CS)"],
    "Kinesiology":             ["Research/Science", "Operations/Business"],
    "Psychology":              ["Research/Science", "Operations/Business",
                                "Consulting", "Data Analytics"],
    "Cognitive Science":       ["Research/Science", "Data Science",
                                "Software Engineering", "Data Analytics"],
    "Bioinformatics":          ["Research/Science", "Data Science",
                                "Software Engineering", "Data Analytics"],
    "Biomedical Engineering":  ["Research/Science", "Engineering (Non-CS)",
                                "Software Engineering"],
    "Chemical Engineering":    ["Engineering (Non-CS)", "Research/Science",
                                "Operations/Business"],
    "Genomics":                ["Research/Science", "Data Science",
                                "Data Analytics"],
    "Pharmacology":            ["Research/Science", "Operations/Business"],
    "Health Sciences":         ["Research/Science", "Operations/Business",
                                "Data Analytics"],
}

DEGREE_BASELINE_SKILLS = {
    "Computer Science":        ["Python", "Java", "Git",
                                "Linux", "Object-Oriented Programming"],
    "Software Engineering":    ["Java", "Git", "System Design", "C++",
                                "Object-Oriented Programming", "Unit Testing"],
    "Data Science":            ["Python", "R programming", "Statistics",
                                "Machine Learning", "pandas", "NumPy"],
    "Electrical Engineering":  ["MATLAB", "Simulink", "Embedded Systems",
                                "C programming", "FPGA"],
    "Mechanical Engineering":  ["AutoCAD", "SolidWorks", "ANSYS", "MATLAB"],
    "Business Administration": ["Excel", "Communication", "Project Management",
                                "Leadership"],
    "Accounting":              ["Excel", "Accounting", "Financial Modeling",
                                "Attention to Detail"],
    "Finance":                 ["Excel", "Financial Modeling", "SQL",
                                "Communication"],
    "Mathematics":             ["Python", "R programming", "MATLAB",
                                "Statistics", "Linear Algebra"],
    "Statistics":              ["R programming", "Python", "Statistics",
                                "SAS", "SPSS"],
    "Information Technology":  ["SQL", "Linux", "Git"],
    # ── Science & Research ────────────────────────────────────────────────────
    "Biology":                 ["Microsoft Office", "Data Analysis",
                                "Lab Techniques", "Scientific Writing",
                                "R programming", "Statistics"],
    "Chemistry":               ["Lab Techniques", "HPLC", "Spectroscopy",
                                "Scientific Writing", "Microsoft Office",
                                "Data Analysis"],
    "Biochemistry":            ["Lab Techniques", "PCR", "Western Blot",
                                "Scientific Writing", "R programming",
                                "Data Analysis", "Microsoft Office"],
    "Neuroscience":            ["R programming", "Python", "Statistics",
                                "MATLAB", "Scientific Writing",
                                "Data Analysis", "Lab Techniques"],
    "Physics":                 ["MATLAB", "Python", "C++", "LaTeX",
                                "Statistics", "Linear Algebra",
                                "Scientific Writing"],
    "Environmental Science":   ["ArcGIS", "R programming", "Data Analysis",
                                "Scientific Writing", "Microsoft Office",
                                "Field Research", "Statistics"],
    "Kinesiology":             ["Data Analysis", "Microsoft Office",
                                "Scientific Writing", "Statistics",
                                "Lab Techniques"],
    "Psychology":              ["SPSS", "R programming", "Statistics",
                                "Scientific Writing", "Data Analysis",
                                "Microsoft Office"],
    "Cognitive Science":       ["Python", "R programming", "MATLAB",
                                "Statistics", "Machine Learning",
                                "Scientific Writing", "Data Analysis"],
    "Bioinformatics":          ["Python", "R programming", "Bash",
                                "Linux", "SQL", "Machine Learning",
                                "Statistics", "Git"],
    "Biomedical Engineering":  ["MATLAB", "Python", "Lab Techniques",
                                "SolidWorks", "Scientific Writing",
                                "Data Analysis", "C programming"],
    "Chemical Engineering":    ["MATLAB", "Aspen Plus", "Lab Techniques",
                                "Scientific Writing", "Data Analysis",
                                "Microsoft Office"],
    "Genomics":                ["Python", "R programming", "Bash",
                                "Linux", "Bioinformatics Tools",
                                "Statistics", "Scientific Writing"],
    "Pharmacology":            ["Lab Techniques", "Scientific Writing",
                                "Statistics", "R programming",
                                "Microsoft Office", "Data Analysis"],
    "Health Sciences":         ["Microsoft Office", "Data Analysis",
                                "Statistics", "Scientific Writing",
                                "Communication", "R programming"]
}

YEAR_TO_LEVEL = {
    1: ["Entry", "Any"],
    2: ["Entry", "Junior", "Any"],
    3: ["Junior", "Intermediate", "Any"],
    4: ["Intermediate", "Senior", "Any"],
}

CANADIAN_UNIVERSITIES = [
    # Ontario
    "University of Toronto", "University of Waterloo", "Western University",
    "McMaster University", "Queen's University", "University of Ottawa",
    "York University", "Toronto Metropolitan University", "Ryerson University",
    "Carleton University", "University of Guelph", "Ontario Tech University",
    "Brock University", "Wilfrid Laurier University", "Lakehead University",
    "Laurentian University", "Nipissing University", "Trent University",
    "University of Windsor", "Algoma University",
    "Royal Military College of Canada", "OCAD University",
    # Quebec
    "McGill University", "Université de Montréal", "Concordia University",
    "Université du Québec à Montréal", "Université Laval",
    "Université de Sherbrooke", "École Polytechnique de Montréal",
    "HEC Montréal", "Bishop's University",
    # British Columbia
    "University of British Columbia", "UBC", "Simon Fraser University",
    "University of Victoria", "University of Northern British Columbia",
    "Kwantlen Polytechnic University", "Thompson Rivers University",
    "Royal Roads University", "Vancouver Island University",
    "Trinity Western University",
    # Alberta
    "University of Alberta", "University of Calgary",
    "University of Lethbridge", "Athabasca University", "MacEwan University",
    "Mount Royal University", "Concordia University of Edmonton",
    # Saskatchewan
    "University of Saskatchewan", "University of Regina",
    "First Nations University of Canada",
    # Manitoba
    "University of Manitoba", "University of Winnipeg", "Brandon University",
    "Canadian Mennonite University", "Université de Saint-Boniface",
    # Nova Scotia
    "Dalhousie University", "Saint Mary's University", "Acadia University",
    "Cape Breton University", "Mount Saint Vincent University",
    "St. Francis Xavier University", "University of King's College",
    # New Brunswick
    "University of New Brunswick", "Université de Moncton",
    "Mount Allison University", "St. Thomas University",
    # Newfoundland
    "Memorial University of Newfoundland",
    # PEI
    "University of Prince Edward Island",
    # Territories
    "Yukon University",
]

# Skill aliases for fuzzy normalization

SKILL_RESOURCES = {
    "Python": [
        {"title": "Python Official Docs", "url": "https://docs.python.org/3/tutorial/", "level": "beginner"},
        {"title": "Kaggle Python Course", "url": "https://www.kaggle.com/learn/python", "level": "beginner"},
    ],
    "SQL": [
        {"title": "SQLZoo", "url": "https://sqlzoo.net/", "level": "beginner"},
        {"title": "Mode SQL Tutorial", "url": "https://mode.com/sql-tutorial/", "level": "intermediate"},
    ],
    "R": [
        {"title": "R for Data Science", "url": "https://r4ds.had.co.nz/", "level": "beginner"},
        {"title": "Swirl (learn R in R)", "url": "https://swirlstats.com/", "level": "beginner"},
    ],
    "Java": [
        {"title": "Oracle Java Tutorials", "url": "https://docs.oracle.com/javase/tutorial/", "level": "beginner"},
        {"title": "Codecademy Java", "url": "https://www.codecademy.com/learn/learn-java", "level": "beginner"},
    ],
    "JavaScript": [
        {"title": "MDN Web Docs", "url": "https://developer.mozilla.org/en-US/docs/Learn", "level": "beginner"},
        {"title": "The Odin Project", "url": "https://www.theodinproject.com/", "level": "beginner"},
    ],
    "TypeScript": [
        {"title": "TypeScript Handbook", "url": "https://www.typescriptlang.org/docs/handbook/", "level": "intermediate"},
        {"title": "Execute Program", "url": "https://www.executeprogram.com/", "level": "intermediate"},
    ],
    "C++": [
        {"title": "LearnCpp.com", "url": "https://www.learncpp.com/", "level": "beginner"},
        {"title": "CPP Reference", "url": "https://en.cppreference.com/", "level": "advanced"},
    ],
    "C": [
        {"title": "CS50x (Harvard)", "url": "https://cs50.harvard.edu/x/", "level": "beginner"},
        {"title": "LearnC.org", "url": "https://www.learn-c.org/", "level": "beginner"},
    ],
    "Go": [
        {"title": "Go Tour", "url": "https://go.dev/tour/", "level": "beginner"},
        {"title": "Go by Example", "url": "https://gobyexample.com/", "level": "intermediate"},
    ],
    "Rust": [
        {"title": "The Rust Book", "url": "https://doc.rust-lang.org/book/", "level": "beginner"},
        {"title": "Rustlings", "url": "https://github.com/rust-lang/rustlings", "level": "intermediate"},
    ],
    "Swift": [
        {"title": "Swift.org Docs", "url": "https://www.swift.org/documentation/", "level": "beginner"},
        {"title": "Hacking with Swift", "url": "https://www.hackingwithswift.com/", "level": "intermediate"},
    ],
    "Kotlin": [
        {"title": "Kotlin Docs", "url": "https://kotlinlang.org/docs/", "level": "beginner"},
        {"title": "JetBrains Academy", "url": "https://www.jetbrains.com/academy/", "level": "intermediate"},
    ],
    "React": [
        {"title": "React Official Docs", "url": "https://react.dev/learn", "level": "beginner"},
        {"title": "Scrimba React Course", "url": "https://scrimba.com/learn/learnreact", "level": "beginner"},
    ],
    "Node.js": [
        {"title": "Node.js Docs", "url": "https://nodejs.org/en/docs/", "level": "intermediate"},
        {"title": "The Odin Project Node", "url": "https://www.theodinproject.com/paths/full-stack-javascript", "level": "beginner"},
    ],
    "Django": [
        {"title": "Django Official Tutorial", "url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/", "level": "beginner"},
        {"title": "Django Girls Tutorial", "url": "https://tutorial.djangogirls.org/", "level": "beginner"},
    ],
    "Flask": [
        {"title": "Flask Docs", "url": "https://flask.palletsprojects.com/", "level": "beginner"},
        {"title": "Miguel Grinberg Flask Tutorial", "url": "https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world", "level": "intermediate"},
    ],
    "FastAPI": [
        {"title": "FastAPI Docs", "url": "https://fastapi.tiangolo.com/tutorial/", "level": "beginner"},
        {"title": "TestDriven.io FastAPI", "url": "https://testdriven.io/blog/fastapi-crud/", "level": "intermediate"},
    ],
    "Docker": [
        {"title": "Docker Getting Started", "url": "https://docs.docker.com/get-started/", "level": "beginner"},
        {"title": "Play with Docker", "url": "https://labs.play-with-docker.com/", "level": "beginner"},
    ],
    "Kubernetes": [
        {"title": "Kubernetes Docs", "url": "https://kubernetes.io/docs/tutorials/", "level": "intermediate"},
        {"title": "KodeKloud K8s Course", "url": "https://kodekloud.com/courses/kubernetes-for-the-absolute-beginners/", "level": "beginner"},
    ],
    "AWS": [
        {"title": "AWS Skill Builder", "url": "https://skillbuilder.aws/", "level": "beginner"},
        {"title": "A Cloud Guru AWS", "url": "https://acloudguru.com/aws-cloud-training", "level": "intermediate"},
    ],
    "Azure": [
        {"title": "Microsoft Learn Azure", "url": "https://learn.microsoft.com/en-us/training/azure/", "level": "beginner"},
        {"title": "AZ-900 Study Guide", "url": "https://learn.microsoft.com/en-us/certifications/exams/az-900", "level": "beginner"},
    ],
    "GCP": [
        {"title": "Google Cloud Skills Boost", "url": "https://cloudskillsboost.google/", "level": "beginner"},
        {"title": "GCP Free Tier", "url": "https://cloud.google.com/free", "level": "beginner"},
    ],
    "Git": [
        {"title": "Pro Git Book", "url": "https://git-scm.com/book/en/v2", "level": "beginner"},
        {"title": "Learn Git Branching", "url": "https://learngitbranching.js.org/", "level": "beginner"},
    ],
    # ⚠️ Removed "Linux/Unix" to avoid fuzzy conflict with "Linux"
    "Linux": [
        {"title": "Linux Journey", "url": "https://linuxjourney.com/", "level": "beginner"},
        {"title": "The Linux Command Line Book", "url": "https://linuxcommand.org/tlcl.php", "level": "intermediate"},
        {"title": "OverTheWire Wargames", "url": "https://overthewire.org/wargames/", "level": "advanced"},
    ],
    "Bash": [
        {"title": "Bash Scripting Tutorial", "url": "https://linuxconfig.org/bash-scripting-tutorial-for-beginners", "level": "beginner"},
        {"title": "ShellCheck", "url": "https://www.shellcheck.net/", "level": "intermediate"},
    ],
    "Machine Learning": [
        {"title": "fast.ai", "url": "https://www.fast.ai/", "level": "intermediate"},
        {"title": "Andrew Ng ML Specialization", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "level": "beginner"},
    ],
    "Deep Learning": [
        {"title": "fast.ai Deep Learning", "url": "https://course.fast.ai/", "level": "intermediate"},
        {"title": "MIT 6.S191", "url": "https://introtodeeplearning.com/", "level": "advanced"},
    ],
    "TensorFlow": [
        {"title": "TensorFlow Tutorials", "url": "https://www.tensorflow.org/tutorials", "level": "beginner"},
        {"title": "DeepLearning.AI TF", "url": "https://www.coursera.org/professional-certificates/tensorflow-in-practice", "level": "intermediate"},
    ],
    "PyTorch": [
        {"title": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "level": "beginner"},
        {"title": "Zero to Mastery PyTorch", "url": "https://www.learnpytorch.io/", "level": "intermediate"},
    ],
    "Pandas": [
        {"title": "Pandas Docs", "url": "https://pandas.pydata.org/docs/getting_started/", "level": "beginner"},
        {"title": "Kaggle Pandas Course", "url": "https://www.kaggle.com/learn/pandas", "level": "beginner"},
    ],
    "NumPy": [
        {"title": "NumPy Quickstart", "url": "https://numpy.org/doc/stable/user/quickstart.html", "level": "beginner"},
        {"title": "Kaggle NumPy", "url": "https://www.kaggle.com/learn/intro-to-programming", "level": "beginner"},
    ],
    "Matplotlib": [
        {"title": "Matplotlib Tutorials", "url": "https://matplotlib.org/stable/tutorials/", "level": "beginner"},
        {"title": "Kaggle Data Visualization", "url": "https://www.kaggle.com/learn/data-visualization", "level": "beginner"},
    ],
    "Scikit-learn": [
        {"title": "Scikit-learn Tutorials", "url": "https://scikit-learn.org/stable/tutorial/", "level": "beginner"},
        {"title": "Kaggle Intro to ML", "url": "https://www.kaggle.com/learn/intro-to-machine-learning", "level": "beginner"},
    ],
    "Tableau": [
        {"title": "Tableau Public Training", "url": "https://public.tableau.com/en-us/s/resources", "level": "beginner"},
        {"title": "Tableau eLearning", "url": "https://www.tableau.com/learn/training", "level": "intermediate"},
    ],
    "Power BI": [
        {"title": "Microsoft Power BI Learning", "url": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi", "level": "beginner"},
        {"title": "Guy in a Cube YouTube", "url": "https://www.youtube.com/@GuyInACube", "level": "beginner"},
    ],
    "Excel": [
        {"title": "Microsoft Excel Training", "url": "https://support.microsoft.com/en-us/excel", "level": "beginner"},
        {"title": "ExcelJet", "url": "https://exceljet.net/", "level": "intermediate"},
    ],
    "PostgreSQL": [
        {"title": "PostgreSQL Tutorial", "url": "https://www.postgresqltutorial.com/", "level": "beginner"},
        {"title": "PG Exercises", "url": "https://pgexercises.com/", "level": "intermediate"},
    ],
    "MongoDB": [
        {"title": "MongoDB University", "url": "https://learn.mongodb.com/", "level": "beginner"},
        {"title": "MongoDB Docs", "url": "https://www.mongodb.com/docs/manual/tutorial/", "level": "intermediate"},
    ],
    "REST API": [
        {"title": "REST API Tutorial", "url": "https://restfulapi.net/", "level": "beginner"},
        {"title": "Postman Learning Center", "url": "https://learning.postman.com/", "level": "beginner"},
    ],
    "GraphQL": [
        {"title": "GraphQL.org Learn", "url": "https://graphql.org/learn/", "level": "beginner"},
        {"title": "How to GraphQL", "url": "https://www.howtographql.com/", "level": "intermediate"},
    ],
    "Agile": [
        {"title": "Atlassian Agile Coach", "url": "https://www.atlassian.com/agile", "level": "beginner"},
        {"title": "Scrum.org", "url": "https://www.scrum.org/resources/what-scrum-module", "level": "beginner"},
    ],
    "Communication": [
        {"title": "Coursera Communication Skills", "url": "https://www.coursera.org/courses?query=communication%20skills", "level": "beginner"},
        {"title": "Toastmasters", "url": "https://www.toastmasters.org/", "level": "beginner"},
    ],
    "Leadership": [
        {"title": "CCL Leadership Resources", "url": "https://www.ccl.org/articles/leading-effectively-articles/", "level": "intermediate"},
        {"title": "MindTools Leadership", "url": "https://www.mindtools.com/leadership-skills", "level": "beginner"},
    ],
    "Teamwork": [
        {"title": "MindTools Teamwork", "url": "https://www.mindtools.com/pages/article/newTMM_53.htm", "level": "beginner"},
        {"title": "Coursera Teamwork Skills", "url": "https://www.coursera.org/courses?query=teamwork", "level": "beginner"},
    ],
    "Problem Solving": [
        {"title": "Brilliant.org", "url": "https://brilliant.org/", "level": "intermediate"},
        {"title": "Project Euler", "url": "https://projecteuler.net/", "level": "advanced"},
    ],
    "Time Management": [
        {"title": "MindTools Time Management", "url": "https://www.mindtools.com/pages/article/newHTE_00.htm", "level": "beginner"},
        {"title": "Todoist Productivity Guide", "url": "https://todoist.com/productivity-methods", "level": "beginner"},
    ],
    "Critical Thinking": [
        {"title": "Foundation for Critical Thinking", "url": "https://www.criticalthinking.org/pages/college-and-university-students/799", "level": "beginner"},
        {"title": "Coursera Critical Thinking", "url": "https://www.coursera.org/courses?query=critical%20thinking", "level": "beginner"},
    ],
    "Adaptability": [
        {"title": "MindTools Adaptability", "url": "https://www.mindtools.com/pages/article/adaptability.htm", "level": "beginner"},
        {"title": "LinkedIn Learning", "url": "https://www.linkedin.com/learning/topics/adaptability", "level": "beginner"},
    ],
    "Attention to Detail": [
        {"title": "Coursera Detail Courses", "url": "https://www.coursera.org/courses?query=attention+to+detail", "level": "beginner"},
        {"title": "MindTools", "url": "https://www.mindtools.com/", "level": "beginner"},
    ],
    # Programming Languages (missing)
    "C#": [
        {"title": "Microsoft C# Docs", "url": "https://learn.microsoft.com/en-us/dotnet/csharp/", "level": "beginner"},
        {"title": "C# Yellow Book", "url": "https://www.robmiles.com/c-yellow-book/", "level": "beginner"},
    ],
    "Golang": [
        {"title": "Go Tour", "url": "https://go.dev/tour/", "level": "beginner"},
        {"title": "Go by Example", "url": "https://gobyexample.com/", "level": "intermediate"},
    ],
    "Scala": [
        {"title": "Scala Docs", "url": "https://docs.scala-lang.org/", "level": "beginner"},
        {"title": "Scala Exercises", "url": "https://www.scala-exercises.org/", "level": "beginner"},
    ],
    "MATLAB": [
        {"title": "MATLAB Onramp", "url": "https://matlabacademy.mathworks.com/", "level": "beginner"},
        {"title": "MATLAB Tutorials", "url": "https://www.tutorialspoint.com/matlab/index.htm", "level": "beginner"},
    ],
    "Julia": [
        {"title": "Julia Docs", "url": "https://docs.julialang.org/en/v1/", "level": "beginner"},
        {"title": "Julia Academy", "url": "https://juliaacademy.com/", "level": "beginner"},
    ],
    "Ruby": [
        {"title": "The Odin Project Ruby", "url": "https://www.theodinproject.com/paths/full-stack-ruby-on-rails", "level": "beginner"},
        {"title": "Ruby Docs", "url": "https://www.ruby-lang.org/en/documentation/", "level": "beginner"},
    ],
    "PHP": [
        {"title": "PHP Manual", "url": "https://www.php.net/manual/en/", "level": "beginner"},
        {"title": "Laracasts PHP", "url": "https://laracasts.com/series/php-for-beginners-2023-edition", "level": "beginner"},
    ],
    "Shell": [
        {"title": "Bash Scripting Tutorial", "url": "https://linuxconfig.org/bash-scripting-tutorial-for-beginners", "level": "beginner"},
        {"title": "ShellCheck", "url": "https://www.shellcheck.net/", "level": "intermediate"},
    ],
    "Perl": [
        {"title": "Perl.org Learn", "url": "https://www.perl.org/learn.html", "level": "beginner"},
        {"title": "Modern Perl Book", "url": "http://modernperlbooks.com/books/modern_perl_2016/", "level": "intermediate"},
    ],
    "SAS": [
        {"title": "SAS Free Training", "url": "https://www.sas.com/en_us/learn/academic-programs/resources/free-sas-e-learning.html", "level": "beginner"},
        {"title": "SAS Tutorials Point", "url": "https://www.tutorialspoint.com/sas/index.htm", "level": "beginner"},
    ],
    "SPSS": [
        {"title": "IBM SPSS Tutorials", "url": "https://www.ibm.com/training/spss", "level": "beginner"},
        {"title": "Kent SPSS Tutorials", "url": "https://www.kent.ac.uk/is/docs/spss/", "level": "beginner"},
    ],
    "R programming": [
        {"title": "R for Data Science", "url": "https://r4ds.had.co.nz/", "level": "beginner"},
        {"title": "Swirl (learn R in R)", "url": "https://swirlstats.com/", "level": "beginner"},
    ],
    "C programming": [
        {"title": "CS50x (Harvard)", "url": "https://cs50.harvard.edu/x/", "level": "beginner"},
        {"title": "LearnC.org", "url": "https://www.learn-c.org/", "level": "beginner"},
    ],

    # Web Development (missing)
    "HTML": [
        {"title": "MDN HTML Docs", "url": "https://developer.mozilla.org/en-US/docs/Learn/HTML", "level": "beginner"},
        {"title": "The Odin Project HTML", "url": "https://www.theodinproject.com/", "level": "beginner"},
    ],
    "CSS": [
        {"title": "MDN CSS Docs", "url": "https://developer.mozilla.org/en-US/docs/Learn/CSS", "level": "beginner"},
        {"title": "CSS Tricks", "url": "https://css-tricks.com/", "level": "intermediate"},
    ],
    "Angular": [
        {"title": "Angular Docs", "url": "https://angular.io/docs", "level": "intermediate"},
        {"title": "Tour of Heroes (Angular)", "url": "https://angular.io/tutorial/tour-of-heroes", "level": "beginner"},
    ],
    "Vue.js": [
        {"title": "Vue.js Docs", "url": "https://vuejs.org/guide/introduction.html", "level": "beginner"},
        {"title": "Vue Mastery", "url": "https://www.vuemastery.com/courses/intro-to-vue-3/intro-to-vue3/", "level": "beginner"},
    ],
    "Spring Boot": [
        {"title": "Spring Guides", "url": "https://spring.io/guides", "level": "intermediate"},
        {"title": "Baeldung Spring Boot", "url": "https://www.baeldung.com/spring-boot", "level": "intermediate"},
    ],
    "Next.js": [
        {"title": "Next.js Docs", "url": "https://nextjs.org/docs", "level": "intermediate"},
        {"title": "Next.js Learn", "url": "https://nextjs.org/learn", "level": "beginner"},
    ],
    "Express.js": [
        {"title": "Express.js Docs", "url": "https://expressjs.com/en/starter/installing.html", "level": "beginner"},
        {"title": "MDN Express Tutorial", "url": "https://developer.mozilla.org/en-US/docs/Learn/Server-side/Express_Nodejs", "level": "beginner"},
    ],
    "Bootstrap": [
        {"title": "Bootstrap Docs", "url": "https://getbootstrap.com/docs/", "level": "beginner"},
        {"title": "Bootstrap Tutorial W3", "url": "https://www.w3schools.com/bootstrap5/", "level": "beginner"},
    ],
    "Tailwind CSS": [
        {"title": "Tailwind Docs", "url": "https://tailwindcss.com/docs/installation", "level": "beginner"},
        {"title": "Tailwind UI Components", "url": "https://tailwindui.com/", "level": "intermediate"},
    ],
    "WordPress": [
        {"title": "WordPress.org Learn", "url": "https://learn.wordpress.org/", "level": "beginner"},
        {"title": "WP Beginner", "url": "https://www.wpbeginner.com/", "level": "beginner"},
    ],

    # Data Science & ML (missing)
    "Natural Language Processing": [
        {"title": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course/", "level": "intermediate"},
        {"title": "Stanford CS224N", "url": "https://web.stanford.edu/class/cs224n/", "level": "advanced"},
    ],
    "Computer Vision": [
        {"title": "OpenCV Tutorials", "url": "https://docs.opencv.org/4.x/d9/df8/tutorial_root.html", "level": "intermediate"},
        {"title": "Fast.ai Computer Vision", "url": "https://course.fast.ai/", "level": "intermediate"},
    ],
    "Reinforcement Learning": [
        {"title": "Spinning Up in RL (OpenAI)", "url": "https://spinningup.openai.com/en/latest/", "level": "advanced"},
        {"title": "David Silver RL Course", "url": "https://www.davidsilver.uk/teaching/", "level": "advanced"},
    ],
    "Statistical Modeling": [
        {"title": "StatQuest YouTube", "url": "https://www.youtube.com/@statquest", "level": "beginner"},
        {"title": "Penn State STAT 501", "url": "https://online.stat.psu.edu/stat501/", "level": "intermediate"},
    ],
    "Predictive Modeling": [
        {"title": "Kaggle Intro to ML", "url": "https://www.kaggle.com/learn/intro-to-machine-learning", "level": "beginner"},
        {"title": "Coursera ML Specialization", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "level": "intermediate"},
    ],
    "Feature Engineering": [
        {"title": "Kaggle Feature Engineering", "url": "https://www.kaggle.com/learn/feature-engineering", "level": "intermediate"},
        {"title": "Feature Engineering for ML (book)", "url": "https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/", "level": "advanced"},
    ],
    "Data Mining": [
        {"title": "Coursera Data Mining Specialization", "url": "https://www.coursera.org/specializations/data-mining", "level": "intermediate"},
        {"title": "UIUC Data Mining Course", "url": "https://www.coursera.org/learn/data-visualization", "level": "beginner"},
    ],
    "Time Series Analysis": [
        {"title": "Kaggle Time Series Course", "url": "https://www.kaggle.com/learn/time-series", "level": "intermediate"},
        {"title": "Forecasting: Principles & Practice", "url": "https://otexts.com/fpp3/", "level": "intermediate"},
    ],
    "A/B Testing": [
        {"title": "Udacity A/B Testing Course", "url": "https://www.udacity.com/course/ab-testing--ud979", "level": "intermediate"},
        {"title": "Trustworthy Online Experiments (book)", "url": "https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59", "level": "advanced"},
    ],
    "Hypothesis Testing": [
        {"title": "Khan Academy Statistics", "url": "https://www.khanacademy.org/math/statistics-probability", "level": "beginner"},
        {"title": "StatQuest Hypothesis Testing", "url": "https://www.youtube.com/@statquest", "level": "beginner"},
    ],
    "Regression": [
        {"title": "StatQuest Regression", "url": "https://www.youtube.com/@statquest", "level": "beginner"},
        {"title": "Penn State STAT 501", "url": "https://online.stat.psu.edu/stat501/", "level": "intermediate"},
    ],
    "Classification": [
        {"title": "Scikit-learn Classification", "url": "https://scikit-learn.org/stable/supervised_learning.html", "level": "intermediate"},
        {"title": "Kaggle Intro to ML", "url": "https://www.kaggle.com/learn/intro-to-machine-learning", "level": "beginner"},
    ],
    "Clustering": [
        {"title": "Scikit-learn Clustering", "url": "https://scikit-learn.org/stable/modules/clustering.html", "level": "intermediate"},
        {"title": "Kaggle Clustering Tutorial", "url": "https://www.kaggle.com/learn/intro-to-machine-learning", "level": "beginner"},
    ],
    "Neural Networks": [
        {"title": "3Blue1Brown Neural Networks", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", "level": "beginner"},
        {"title": "fast.ai Deep Learning", "url": "https://course.fast.ai/", "level": "intermediate"},
    ],
    "Transformer Models": [
        {"title": "Hugging Face Transformers Course", "url": "https://huggingface.co/learn/nlp-course/", "level": "intermediate"},
        {"title": "Illustrated Transformer (blog)", "url": "https://jalammar.github.io/illustrated-transformer/", "level": "intermediate"},
    ],
    "LLMs": [
        {"title": "Hugging Face LLM Course", "url": "https://huggingface.co/learn/nlp-course/", "level": "intermediate"},
        {"title": "Andrej Karpathy — Let's build GPT", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY", "level": "advanced"},
    ],
    "Generative AI": [
        {"title": "Google Generative AI Learning Path", "url": "https://cloudskillsboost.google/paths/118", "level": "beginner"},
        {"title": "DeepLearning.AI Short Courses", "url": "https://www.deeplearning.ai/short-courses/", "level": "beginner"},
    ],
    "Prompt Engineering": [
        {"title": "DeepLearning.AI Prompt Engineering", "url": "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/", "level": "beginner"},
        {"title": "Anthropic Prompt Engineering Guide", "url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview", "level": "intermediate"},
    ],

    # ML Frameworks (missing)
    "Keras": [
        {"title": "Keras Docs", "url": "https://keras.io/getting_started/", "level": "beginner"},
        {"title": "DeepLearning.AI TF+Keras", "url": "https://www.coursera.org/professional-certificates/tensorflow-in-practice", "level": "intermediate"},
    ],
    "XGBoost": [
        {"title": "XGBoost Docs", "url": "https://xgboost.readthedocs.io/en/stable/tutorials/model.html", "level": "intermediate"},
        {"title": "Kaggle XGBoost Tutorial", "url": "https://www.kaggle.com/learn/intermediate-machine-learning", "level": "intermediate"},
    ],
    "LightGBM": [
        {"title": "LightGBM Docs", "url": "https://lightgbm.readthedocs.io/en/stable/", "level": "intermediate"},
        {"title": "Kaggle LightGBM Guide", "url": "https://www.kaggle.com/learn/intermediate-machine-learning", "level": "intermediate"},
    ],
    "CatBoost": [
        {"title": "CatBoost Docs", "url": "https://catboost.ai/en/docs/", "level": "intermediate"},
        {"title": "CatBoost Tutorial", "url": "https://catboost.ai/en/docs/concepts/tutorials", "level": "intermediate"},
    ],
    "Hugging Face": [
        {"title": "Hugging Face Course", "url": "https://huggingface.co/learn/nlp-course/", "level": "intermediate"},
        {"title": "Hugging Face Docs", "url": "https://huggingface.co/docs", "level": "intermediate"},
    ],
    "spaCy": [
        {"title": "spaCy Course", "url": "https://course.spacy.io/en/", "level": "intermediate"},
        {"title": "spaCy Docs", "url": "https://spacy.io/usage", "level": "intermediate"},
    ],
    "NLTK": [
        {"title": "NLTK Book (free)", "url": "https://www.nltk.org/book/", "level": "beginner"},
        {"title": "NLTK Docs", "url": "https://www.nltk.org/", "level": "beginner"},
    ],
    "OpenCV": [
        {"title": "OpenCV Tutorials", "url": "https://docs.opencv.org/4.x/d9/df8/tutorial_root.html", "level": "intermediate"},
        {"title": "PyImageSearch OpenCV", "url": "https://pyimagesearch.com/start-here/", "level": "beginner"},
    ],
    "SciPy": [
        {"title": "SciPy Docs", "url": "https://docs.scipy.org/doc/scipy/tutorial/index.html", "level": "intermediate"},
        {"title": "SciPy Lecture Notes", "url": "https://scipy-lectures.org/", "level": "intermediate"},
    ],
    "Seaborn": [
        {"title": "Seaborn Docs", "url": "https://seaborn.pydata.org/tutorial.html", "level": "beginner"},
        {"title": "Kaggle Data Visualization", "url": "https://www.kaggle.com/learn/data-visualization", "level": "beginner"},
    ],
    "Plotly": [
        {"title": "Plotly Docs", "url": "https://plotly.com/python/getting-started/", "level": "beginner"},
        {"title": "Plotly Express Tutorial", "url": "https://plotly.com/python/plotly-express/", "level": "beginner"},
    ],
    "Statsmodels": [
        {"title": "Statsmodels Docs", "url": "https://www.statsmodels.org/stable/gettingstarted.html", "level": "intermediate"},
        {"title": "Statsmodels Examples", "url": "https://www.statsmodels.org/stable/examples/", "level": "intermediate"},
    ],

    # Data Engineering (missing)
    "ETL": [
        {"title": "Fundamentals of Data Engineering (book)", "url": "https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/", "level": "intermediate"},
        {"title": "Coursera ETL and Data Pipelines", "url": "https://www.coursera.org/learn/etl-and-data-pipelines-shell-airflow-kafka", "level": "intermediate"},
    ],
    "Data Pipelines": [
        {"title": "Coursera ETL and Data Pipelines", "url": "https://www.coursera.org/learn/etl-and-data-pipelines-shell-airflow-kafka", "level": "intermediate"},
        {"title": "Prefect Docs", "url": "https://docs.prefect.io/", "level": "intermediate"},
    ],
    "Apache Spark": [
        {"title": "Spark Docs", "url": "https://spark.apache.org/docs/latest/", "level": "intermediate"},
        {"title": "Databricks Spark Training", "url": "https://www.databricks.com/learn/training/home", "level": "intermediate"},
    ],
    "Apache Kafka": [
        {"title": "Kafka Docs", "url": "https://kafka.apache.org/documentation/", "level": "intermediate"},
        {"title": "Confluent Kafka Tutorial", "url": "https://developer.confluent.io/learn-kafka/", "level": "beginner"},
    ],
    "Hadoop": [
        {"title": "Hadoop Docs", "url": "https://hadoop.apache.org/docs/current/", "level": "intermediate"},
        {"title": "Udemy Hadoop Course", "url": "https://www.udemy.com/course/the-ultimate-hands-on-hadoop-tame-your-big-data/", "level": "beginner"},
    ],
    "Airflow": [
        {"title": "Apache Airflow Docs", "url": "https://airflow.apache.org/docs/", "level": "intermediate"},
        {"title": "Astronomer Airflow Tutorial", "url": "https://www.astronomer.io/docs/learn/", "level": "beginner"},
    ],
    "dbt": [
        {"title": "dbt Docs", "url": "https://docs.getdbt.com/docs/introduction", "level": "intermediate"},
        {"title": "dbt Learn", "url": "https://courses.getdbt.com/", "level": "beginner"},
    ],
    "Snowflake": [
        {"title": "Snowflake Docs", "url": "https://docs.snowflake.com/en/user-guide-getting-started", "level": "intermediate"},
        {"title": "Snowflake Hands-on Labs", "url": "https://quickstarts.snowflake.com/", "level": "beginner"},
    ],
    "Databricks": [
        {"title": "Databricks Academy", "url": "https://www.databricks.com/learn/training/home", "level": "intermediate"},
        {"title": "Databricks Docs", "url": "https://docs.databricks.com/", "level": "intermediate"},
    ],
    "PySpark": [
        {"title": "PySpark Docs", "url": "https://spark.apache.org/docs/latest/api/python/", "level": "intermediate"},
        {"title": "Databricks PySpark Tutorial", "url": "https://www.databricks.com/learn/training/home", "level": "beginner"},
    ],
    "Data Warehousing": [
        {"title": "Snowflake Quickstarts", "url": "https://quickstarts.snowflake.com/", "level": "beginner"},
        {"title": "Google Cloud Data Warehousing", "url": "https://cloud.google.com/bigquery/docs/introduction", "level": "intermediate"},
    ],
    "Data Modeling": [
        {"title": "dbt Learn Data Modeling", "url": "https://courses.getdbt.com/", "level": "intermediate"},
        {"title": "Kimball Group Resources", "url": "https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/", "level": "advanced"},
    ],
    "Data Cleaning": [
        {"title": "Kaggle Data Cleaning Course", "url": "https://www.kaggle.com/learn/data-cleaning", "level": "beginner"},
        {"title": "Pandas Docs", "url": "https://pandas.pydata.org/docs/getting_started/", "level": "beginner"},
    ],
    "Data Wrangling": [
        {"title": "Kaggle Pandas Course", "url": "https://www.kaggle.com/learn/pandas", "level": "beginner"},
        {"title": "Pandas Docs", "url": "https://pandas.pydata.org/docs/getting_started/", "level": "beginner"},
    ],
    "Web Scraping": [
        {"title": "Real Python Web Scraping", "url": "https://realpython.com/python-web-scraping-practical-introduction/", "level": "beginner"},
        {"title": "Scrapy Docs", "url": "https://docs.scrapy.org/en/latest/intro/tutorial.html", "level": "intermediate"},
    ],
    "BeautifulSoup": [
        {"title": "BeautifulSoup Docs", "url": "https://www.crummy.com/software/BeautifulSoup/bs4/doc/", "level": "beginner"},
        {"title": "Real Python BeautifulSoup", "url": "https://realpython.com/beautiful-soup-web-scraper-python/", "level": "beginner"},
    ],
    "Scrapy": [
        {"title": "Scrapy Docs", "url": "https://docs.scrapy.org/en/latest/intro/tutorial.html", "level": "intermediate"},
        {"title": "Scrapy Tutorial", "url": "https://docs.scrapy.org/en/latest/intro/tutorial.html", "level": "beginner"},
    ],

    # Databases (missing)
    "MySQL": [
        {"title": "MySQL Tutorial", "url": "https://www.mysqltutorial.org/", "level": "beginner"},
        {"title": "MySQL Docs", "url": "https://dev.mysql.com/doc/", "level": "intermediate"},
    ],
    "SQLite": [
        {"title": "SQLite Tutorial", "url": "https://www.sqlitetutorial.net/", "level": "beginner"},
        {"title": "SQLite Docs", "url": "https://www.sqlite.org/docs.html", "level": "intermediate"},
    ],
    "Redis": [
        {"title": "Redis University", "url": "https://university.redis.com/", "level": "beginner"},
        {"title": "Redis Docs", "url": "https://redis.io/docs/", "level": "intermediate"},
    ],
    "Cassandra": [
        {"title": "DataStax Cassandra Course", "url": "https://www.datastax.com/learn/cassandra-fundamentals", "level": "intermediate"},
        {"title": "Apache Cassandra Docs", "url": "https://cassandra.apache.org/doc/latest/", "level": "intermediate"},
    ],
    "DynamoDB": [
        {"title": "AWS DynamoDB Docs", "url": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html", "level": "intermediate"},
        {"title": "DynamoDB Tutorial", "url": "https://aws.amazon.com/dynamodb/getting-started/", "level": "beginner"},
    ],
    "Firebase": [
        {"title": "Firebase Docs", "url": "https://firebase.google.com/docs", "level": "beginner"},
        {"title": "Firebase Codelabs", "url": "https://firebase.google.com/codelabs/firebase-web", "level": "beginner"},
    ],
    "Oracle": [
        {"title": "Oracle Live SQL", "url": "https://livesql.oracle.com/", "level": "beginner"},
        {"title": "Oracle Docs", "url": "https://docs.oracle.com/en/database/", "level": "intermediate"},
    ],
    "Microsoft SQL Server": [
        {"title": "Microsoft SQL Docs", "url": "https://learn.microsoft.com/en-us/sql/sql-server/", "level": "beginner"},
        {"title": "SQL Server Tutorial", "url": "https://www.sqlservertutorial.net/", "level": "beginner"},
    ],
    "NoSQL": [
        {"title": "MongoDB University", "url": "https://learn.mongodb.com/", "level": "beginner"},
        {"title": "NoSQL Distilled (book)", "url": "https://www.oreilly.com/library/view/nosql-distilled-a/9780133036138/", "level": "intermediate"},
    ],
    "Elasticsearch": [
        {"title": "Elastic Docs", "url": "https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html", "level": "intermediate"},
        {"title": "Elastic Getting Started", "url": "https://www.elastic.co/training/free", "level": "beginner"},
    ],
    "Neo4j": [
        {"title": "Neo4j GraphAcademy", "url": "https://graphacademy.neo4j.com/", "level": "beginner"},
        {"title": "Neo4j Docs", "url": "https://neo4j.com/docs/", "level": "intermediate"},
    ],

    # Cloud & DevOps (missing)
    "Google Cloud": [
        {"title": "Google Cloud Skills Boost", "url": "https://cloudskillsboost.google/", "level": "beginner"},
        {"title": "GCP Free Tier", "url": "https://cloud.google.com/free", "level": "beginner"},
    ],
    "Terraform": [
        {"title": "Terraform Docs", "url": "https://developer.hashicorp.com/terraform/docs", "level": "intermediate"},
        {"title": "HashiCorp Learn Terraform", "url": "https://developer.hashicorp.com/terraform/tutorials", "level": "beginner"},
    ],
    "CI/CD": [
        {"title": "GitHub Actions Docs", "url": "https://docs.github.com/en/actions", "level": "beginner"},
        {"title": "GitLab CI/CD Docs", "url": "https://docs.gitlab.com/ee/ci/", "level": "intermediate"},
    ],
    "Jenkins": [
        {"title": "Jenkins Docs", "url": "https://www.jenkins.io/doc/", "level": "intermediate"},
        {"title": "Jenkins Tutorial", "url": "https://www.jenkins.io/doc/tutorials/", "level": "beginner"},
    ],
    "GitHub Actions": [
        {"title": "GitHub Actions Docs", "url": "https://docs.github.com/en/actions", "level": "beginner"},
        {"title": "GitHub Actions Quickstart", "url": "https://docs.github.com/en/actions/quickstart", "level": "beginner"},
    ],
    "Unix": [
        {"title": "Linux Journey", "url": "https://linuxjourney.com/", "level": "beginner"},
        {"title": "The Linux Command Line Book", "url": "https://linuxcommand.org/tlcl.php", "level": "beginner"},
    ],
    "Bash Scripting": [
        {"title": "Bash Scripting Tutorial", "url": "https://linuxconfig.org/bash-scripting-tutorial-for-beginners", "level": "beginner"},
        {"title": "ShellCheck", "url": "https://www.shellcheck.net/", "level": "intermediate"},
    ],
    "Ansible": [
        {"title": "Ansible Docs", "url": "https://docs.ansible.com/ansible/latest/getting_started/index.html", "level": "intermediate"},
        {"title": "Red Hat Ansible Tutorial", "url": "https://www.redhat.com/en/technologies/management/ansible/learn", "level": "beginner"},
    ],
    "Helm": [
        {"title": "Helm Docs", "url": "https://helm.sh/docs/", "level": "intermediate"},
        {"title": "Helm Quickstart", "url": "https://helm.sh/docs/intro/quickstart/", "level": "beginner"},
    ],
    "Microservices": [
        {"title": "Martin Fowler Microservices", "url": "https://martinfowler.com/articles/microservices.html", "level": "intermediate"},
        {"title": "Microservices.io Patterns", "url": "https://microservices.io/patterns/index.html", "level": "advanced"},
    ],
    "Serverless": [
        {"title": "Serverless Framework Docs", "url": "https://www.serverless.com/framework/docs/", "level": "intermediate"},
        {"title": "AWS Lambda Getting Started", "url": "https://docs.aws.amazon.com/lambda/latest/dg/getting-started.html", "level": "beginner"},
    ],
    "CloudFormation": [
        {"title": "AWS CloudFormation Docs", "url": "https://docs.aws.amazon.com/cloudformation/", "level": "intermediate"},
        {"title": "CloudFormation Getting Started", "url": "https://aws.amazon.com/cloudformation/getting-started/", "level": "beginner"},
    ],
    "EC2": [
        {"title": "AWS EC2 Docs", "url": "https://docs.aws.amazon.com/ec2/", "level": "beginner"},
        {"title": "AWS EC2 Getting Started", "url": "https://aws.amazon.com/ec2/getting-started/", "level": "beginner"},
    ],
    "S3": [
        {"title": "AWS S3 Docs", "url": "https://docs.aws.amazon.com/s3/", "level": "beginner"},
        {"title": "AWS S3 Getting Started", "url": "https://aws.amazon.com/s3/getting-started/", "level": "beginner"},
    ],
    "Lambda": [
        {"title": "AWS Lambda Docs", "url": "https://docs.aws.amazon.com/lambda/latest/dg/getting-started.html", "level": "beginner"},
        {"title": "Serverless Framework", "url": "https://www.serverless.com/framework/docs/", "level": "intermediate"},
    ],

    # BI & Visualization (missing)
    "Looker": [
        {"title": "Looker Docs", "url": "https://cloud.google.com/looker/docs", "level": "intermediate"},
        {"title": "Google Looker Training", "url": "https://cloudskillsboost.google/catalog?keywords=looker", "level": "beginner"},
    ],
    "Google Sheets": [
        {"title": "Google Sheets Training", "url": "https://support.google.com/docs/answer/6000292", "level": "beginner"},
        {"title": "Spreadsheet.dev", "url": "https://spreadsheet.dev/", "level": "intermediate"},
    ],
    "D3.js": [
        {"title": "D3.js Docs", "url": "https://d3js.org/getting-started", "level": "advanced"},
        {"title": "Observable D3 Tutorials", "url": "https://observablehq.com/@d3/learn-d3", "level": "intermediate"},
    ],
    "Grafana": [
        {"title": "Grafana Docs", "url": "https://grafana.com/docs/grafana/latest/getting-started/", "level": "beginner"},
        {"title": "Grafana Tutorials", "url": "https://grafana.com/tutorials/", "level": "beginner"},
    ],
    "Kibana": [
        {"title": "Kibana Docs", "url": "https://www.elastic.co/guide/en/kibana/current/index.html", "level": "intermediate"},
        {"title": "Elastic Training", "url": "https://www.elastic.co/training/free", "level": "beginner"},
    ],
    "QlikView": [
        {"title": "Qlik Learning", "url": "https://learning.qlik.com/", "level": "beginner"},
        {"title": "Qlik Docs", "url": "https://help.qlik.com/", "level": "intermediate"},
    ],

    # Version Control & Collaboration (missing)
    "GitHub": [
        {"title": "GitHub Skills", "url": "https://skills.github.com/", "level": "beginner"},
        {"title": "GitHub Docs", "url": "https://docs.github.com/en/get-started", "level": "beginner"},
    ],
    "GitLab": [
        {"title": "GitLab Docs", "url": "https://docs.gitlab.com/ee/tutorials/", "level": "beginner"},
        {"title": "GitLab University", "url": "https://university.gitlab.com/", "level": "beginner"},
    ],
    "Bitbucket": [
        {"title": "Bitbucket Docs", "url": "https://support.atlassian.com/bitbucket-cloud/docs/", "level": "beginner"},
        {"title": "Atlassian Git Tutorials", "url": "https://www.atlassian.com/git/tutorials", "level": "beginner"},
    ],
    "Jira": [
        {"title": "Atlassian Jira Training", "url": "https://university.atlassian.com/student/catalog/list?category_ids=862438-jira", "level": "beginner"},
        {"title": "Jira Docs", "url": "https://support.atlassian.com/jira-software-cloud/", "level": "beginner"},
    ],
    "Confluence": [
        {"title": "Confluence Docs", "url": "https://support.atlassian.com/confluence-cloud/", "level": "beginner"},
        {"title": "Atlassian Confluence Training", "url": "https://university.atlassian.com/student/catalog/list?category_ids=862440-confluence", "level": "beginner"},
    ],
    "Trello": [
        {"title": "Trello Getting Started", "url": "https://trello.com/guide", "level": "beginner"},
        {"title": "Atlassian Trello Docs", "url": "https://support.atlassian.com/trello/", "level": "beginner"},
    ],
    "Asana": [
        {"title": "Asana Academy", "url": "https://academy.asana.com/", "level": "beginner"},
        {"title": "Asana Guide", "url": "https://asana.com/guide", "level": "beginner"},
    ],
    "Notion": [
        {"title": "Notion Guides", "url": "https://www.notion.so/help/guides", "level": "beginner"},
        {"title": "Notion Academy", "url": "https://www.notion.so/learn", "level": "beginner"},
    ],

    # Software Engineering Practices (missing)
    "Scrum": [
        {"title": "Scrum.org", "url": "https://www.scrum.org/resources/what-scrum-module", "level": "beginner"},
        {"title": "Atlassian Scrum Guide", "url": "https://www.atlassian.com/agile/scrum", "level": "beginner"},
    ],
    "Kanban": [
        {"title": "Atlassian Kanban Guide", "url": "https://www.atlassian.com/agile/kanban", "level": "beginner"},
        {"title": "Kanbanize Learning Center", "url": "https://kanbanize.com/kanban-resources", "level": "beginner"},
    ],
    "DevOps": [
        {"title": "Google DevOps Guide", "url": "https://cloud.google.com/devops", "level": "intermediate"},
        {"title": "Microsoft DevOps Learn", "url": "https://learn.microsoft.com/en-us/devops/", "level": "beginner"},
    ],
    "Test-Driven Development": [
        {"title": "TDD by Example (book)", "url": "https://www.oreilly.com/library/view/test-driven-development/0321146530/", "level": "intermediate"},
        {"title": "Real Python TDD", "url": "https://realpython.com/python-testing/", "level": "beginner"},
    ],
    "Unit Testing": [
        {"title": "Real Python Testing", "url": "https://realpython.com/python-testing/", "level": "beginner"},
        {"title": "pytest Docs", "url": "https://docs.pytest.org/en/stable/", "level": "beginner"},
    ],
    "Integration Testing": [
        {"title": "Martin Fowler Integration Testing", "url": "https://martinfowler.com/bliki/IntegrationTest.html", "level": "intermediate"},
        {"title": "Real Python Testing", "url": "https://realpython.com/python-testing/", "level": "beginner"},
    ],
    "Object-Oriented Programming": [
        {"title": "Real Python OOP", "url": "https://realpython.com/python3-object-oriented-programming/", "level": "beginner"},
        {"title": "Coursera OOP in Java", "url": "https://www.coursera.org/specializations/object-oriented-programming", "level": "beginner"},
    ],
    "Functional Programming": [
        {"title": "Mostly Adequate Guide (JS FP)", "url": "https://github.com/MostlyAdequate/mostly-adequate-guide", "level": "intermediate"},
        {"title": "Real Python Functional Programming", "url": "https://realpython.com/python-functional-programming/", "level": "intermediate"},
    ],
    "Design Patterns": [
        {"title": "Refactoring.Guru", "url": "https://refactoring.guru/design-patterns", "level": "intermediate"},
        {"title": "Head First Design Patterns (book)", "url": "https://www.oreilly.com/library/view/head-first-design/9781492077992/", "level": "intermediate"},
    ],
    "System Design": [
        {"title": "System Design Primer (GitHub)", "url": "https://github.com/donnemartin/system-design-primer", "level": "intermediate"},
        {"title": "Grokking System Design", "url": "https://www.educative.io/courses/grokking-the-system-design-interview", "level": "advanced"},
    ],
    "API Development": [
        {"title": "REST API Tutorial", "url": "https://restfulapi.net/", "level": "beginner"},
        {"title": "FastAPI Docs", "url": "https://fastapi.tiangolo.com/tutorial/", "level": "beginner"},
    ],
    "Code Review": [
        {"title": "Google Code Review Guide", "url": "https://google.github.io/eng-practices/review/", "level": "intermediate"},
        {"title": "Best Practices for Code Review", "url": "https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/", "level": "beginner"},
    ],

    # Cybersecurity (missing)
    "Network Security": [
        {"title": "Cybrary Network Security", "url": "https://www.cybrary.it/course/comptia-security-plus/", "level": "intermediate"},
        {"title": "CompTIA Security+ Guide", "url": "https://www.comptia.org/certifications/security", "level": "intermediate"},
    ],
    "Penetration Testing": [
        {"title": "TryHackMe", "url": "https://tryhackme.com/", "level": "beginner"},
        {"title": "Hack The Box", "url": "https://www.hackthebox.com/", "level": "intermediate"},
    ],
    "Ethical Hacking": [
        {"title": "TryHackMe", "url": "https://tryhackme.com/", "level": "beginner"},
        {"title": "CEH Study Guide", "url": "https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/", "level": "advanced"},
    ],
    "Vulnerability Assessment": [
        {"title": "OWASP Testing Guide", "url": "https://owasp.org/www-project-testing-guide/", "level": "intermediate"},
        {"title": "Nessus Essentials", "url": "https://www.tenable.com/products/nessus/nessus-essentials", "level": "intermediate"},
    ],
    "Cryptography": [
        {"title": "Cryptography I (Coursera)", "url": "https://www.coursera.org/learn/crypto", "level": "advanced"},
        {"title": "Crypto101 (free book)", "url": "https://www.crypto101.io/", "level": "intermediate"},
    ],
    "Firewalls": [
        {"title": "Cybrary Firewalls Course", "url": "https://www.cybrary.it/", "level": "intermediate"},
        {"title": "CompTIA Network+ Guide", "url": "https://www.comptia.org/certifications/network", "level": "beginner"},
    ],
    "IAM": [
        {"title": "AWS IAM Docs", "url": "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html", "level": "intermediate"},
        {"title": "Google Cloud IAM", "url": "https://cloud.google.com/iam/docs/overview", "level": "intermediate"},
    ],
    "OWASP": [
        {"title": "OWASP Top 10", "url": "https://owasp.org/www-project-top-ten/", "level": "intermediate"},
        {"title": "OWASP Testing Guide", "url": "https://owasp.org/www-project-testing-guide/", "level": "advanced"},
    ],
    "Zero Trust": [
        {"title": "NIST Zero Trust Architecture", "url": "https://www.nist.gov/publications/zero-trust-architecture", "level": "advanced"},
        {"title": "Cloudflare Zero Trust Docs", "url": "https://developers.cloudflare.com/cloudflare-one/", "level": "intermediate"},
    ],
    "SOC": [
        {"title": "Cybrary SOC Analyst Course", "url": "https://www.cybrary.it/course/soc-analyst/", "level": "intermediate"},
        {"title": "TryHackMe SOC Level 1", "url": "https://tryhackme.com/path/outline/soclevel1", "level": "beginner"},
    ],
    "SIEM": [
        {"title": "Splunk Free Training", "url": "https://www.splunk.com/en_us/training/free-courses/overview.html", "level": "beginner"},
        {"title": "Microsoft Sentinel Docs", "url": "https://learn.microsoft.com/en-us/azure/sentinel/", "level": "intermediate"},
    ],

    # Mobile Development (missing)
    "Android": [
        {"title": "Android Developers Docs", "url": "https://developer.android.com/courses", "level": "beginner"},
        {"title": "Udacity Android Basics", "url": "https://www.udacity.com/course/android-basics-nanodegree-by-google--nd803", "level": "beginner"},
    ],
    "iOS": [
        {"title": "Apple Developer Tutorials", "url": "https://developer.apple.com/tutorials/swiftui", "level": "beginner"},
        {"title": "Hacking with Swift", "url": "https://www.hackingwithswift.com/", "level": "beginner"},
    ],
    "React Native": [
        {"title": "React Native Docs", "url": "https://reactnative.dev/docs/getting-started", "level": "intermediate"},
        {"title": "Expo Go Tutorial", "url": "https://docs.expo.dev/tutorial/introduction/", "level": "beginner"},
    ],
    "Flutter": [
        {"title": "Flutter Docs", "url": "https://docs.flutter.dev/get-started/codelab", "level": "beginner"},
        {"title": "Flutter Codelabs", "url": "https://docs.flutter.dev/codelabs", "level": "beginner"},
    ],
    "Xcode": [
        {"title": "Apple Xcode Docs", "url": "https://developer.apple.com/documentation/xcode", "level": "beginner"},
        {"title": "Hacking with Swift Xcode", "url": "https://www.hackingwithswift.com/read/0/3/choosing-your-xcode-version", "level": "beginner"},
    ],
    "Android Studio": [
        {"title": "Android Studio Docs", "url": "https://developer.android.com/studio/intro", "level": "beginner"},
        {"title": "Android Developers Codelabs", "url": "https://developer.android.com/courses", "level": "beginner"},
    ],

    # Mathematics & Statistics (missing)
    "Linear Algebra": [
        {"title": "3Blue1Brown Linear Algebra", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab", "level": "beginner"},
        {"title": "MIT 18.06 Linear Algebra", "url": "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", "level": "intermediate"},
    ],
    "Calculus": [
        {"title": "Khan Academy Calculus", "url": "https://www.khanacademy.org/math/calculus-1", "level": "beginner"},
        {"title": "3Blue1Brown Calculus", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr", "level": "beginner"},
    ],
    "Probability": [
        {"title": "Khan Academy Probability", "url": "https://www.khanacademy.org/math/statistics-probability", "level": "beginner"},
        {"title": "Introduction to Probability (Blitzstein)", "url": "https://projects.iq.harvard.edu/stat110/home", "level": "intermediate"},
    ],
    "Statistics": [
        {"title": "Khan Academy Statistics", "url": "https://www.khanacademy.org/math/statistics-probability", "level": "beginner"},
        {"title": "StatQuest YouTube", "url": "https://www.youtube.com/@statquest", "level": "beginner"},
    ],
    "Bayesian Inference": [
        {"title": "Bayesian Methods for Hackers", "url": "https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers", "level": "intermediate"},
        {"title": "Statistical Rethinking (book)", "url": "https://xcelab.net/rm/statistical-rethinking/", "level": "advanced"},
    ],
    "Optimization": [
        {"title": "Convex Optimization (Stanford)", "url": "https://www.edx.org/course/convex-optimization", "level": "advanced"},
        {"title": "Scipy Optimization Docs", "url": "https://docs.scipy.org/doc/scipy/tutorial/optimize.html", "level": "intermediate"},
    ],
    "Graph Theory": [
        {"title": "Coursera Graph Theory", "url": "https://www.coursera.org/learn/graphs", "level": "intermediate"},
        {"title": "Brilliant Graph Theory", "url": "https://brilliant.org/courses/graph-theory/", "level": "intermediate"},
    ],
    "Discrete Mathematics": [
        {"title": "MIT 6.042J Discrete Math", "url": "https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/", "level": "intermediate"},
        {"title": "Coursera Discrete Math", "url": "https://www.coursera.org/learn/discrete-mathematics", "level": "beginner"},
    ],
    "Numerical Methods": [
        {"title": "Numerical Methods for Engineers (Coursera)", "url": "https://www.coursera.org/learn/numerical-methods-engineers", "level": "intermediate"},
        {"title": "SciPy Numerical Methods", "url": "https://docs.scipy.org/doc/scipy/tutorial/index.html", "level": "intermediate"},
    ],

    # Business & Soft Skills (missing)
    "Project Management": [
        {"title": "Google Project Management Certificate", "url": "https://www.coursera.org/professional-certificates/google-project-management", "level": "beginner"},
        {"title": "PMI Resources", "url": "https://www.pmi.org/learning/library", "level": "intermediate"},
    ],
    "Presentation Skills": [
        {"title": "Coursera Presentation Skills", "url": "https://www.coursera.org/courses?query=presentation%20skills", "level": "beginner"},
        {"title": "Toastmasters", "url": "https://www.toastmasters.org/", "level": "beginner"},
    ],
    "Stakeholder Management": [
        {"title": "PMI Stakeholder Management", "url": "https://www.pmi.org/learning/library/stakeholder-management-fundamentals-7736", "level": "intermediate"},
        {"title": "Coursera Project Management", "url": "https://www.coursera.org/professional-certificates/google-project-management", "level": "beginner"},
    ],
    "Cross-functional Collaboration": [
        {"title": "Coursera Teamwork Skills", "url": "https://www.coursera.org/courses?query=teamwork", "level": "beginner"},
        {"title": "MindTools Collaboration", "url": "https://www.mindtools.com/pages/article/newTMM_53.htm", "level": "beginner"},
    ],

    # Domain Specific (missing)
    "Financial Modeling": [
        {"title": "CFI Financial Modeling", "url": "https://corporatefinanceinstitute.com/resources/financial-modeling/", "level": "intermediate"},
        {"title": "Wall Street Prep", "url": "https://www.wallstreetprep.com/knowledge/financial-modeling/", "level": "advanced"},
    ],
    "Accounting": [
        {"title": "Khan Academy Finance & Accounting", "url": "https://www.khanacademy.org/economics-finance-domain/core-finance", "level": "beginner"},
        {"title": "Coursera Accounting", "url": "https://www.coursera.org/courses?query=accounting", "level": "beginner"},
    ],
    "Supply Chain": [
        {"title": "Coursera Supply Chain Specialization", "url": "https://www.coursera.org/specializations/supply-chain-management", "level": "beginner"},
        {"title": "MITx Supply Chain MicroMasters", "url": "https://micromasters.mit.edu/scm/", "level": "advanced"},
    ],
    "Operations Research": [
        {"title": "MIT 15.053 Optimization Methods", "url": "https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/", "level": "advanced"},
        {"title": "Coursera Operations Research", "url": "https://www.coursera.org/courses?query=operations+research", "level": "intermediate"},
    ],
    "Marketing Analytics": [
        {"title": "Google Analytics Academy", "url": "https://analytics.google.com/analytics/academy/", "level": "beginner"},
        {"title": "Coursera Marketing Analytics", "url": "https://www.coursera.org/learn/marketing-analytics", "level": "intermediate"},
    ],
    "SEO": [
        {"title": "Google Search Central", "url": "https://developers.google.com/search/docs", "level": "beginner"},
        {"title": "Moz Beginner's Guide to SEO", "url": "https://moz.com/beginners-guide-to-seo", "level": "beginner"},
    ],
    "CRM": [
        {"title": "HubSpot Academy", "url": "https://academy.hubspot.com/", "level": "beginner"},
        {"title": "Salesforce Trailhead", "url": "https://trailhead.salesforce.com/", "level": "beginner"},
    ],
    "Salesforce": [
        {"title": "Salesforce Trailhead", "url": "https://trailhead.salesforce.com/", "level": "beginner"},
        {"title": "Salesforce Docs", "url": "https://help.salesforce.com/", "level": "intermediate"},
    ],
    "SAP": [
        {"title": "SAP Learning Hub", "url": "https://learning.sap.com/", "level": "beginner"},
        {"title": "SAP Training Catalog", "url": "https://training.sap.com/", "level": "intermediate"},
    ],
    "AutoCAD": [
        {"title": "Autodesk AutoCAD Tutorials", "url": "https://www.autodesk.com/learning/online-classes", "level": "beginner"},
        {"title": "LinkedIn Learning AutoCAD", "url": "https://www.linkedin.com/learning/topics/autocad", "level": "beginner"},
    ],
    "SolidWorks": [
        {"title": "SolidWorks Tutorials", "url": "https://www.solidworks.com/sw/resources/getting-started.htm", "level": "beginner"},
        {"title": "Dassault SolidWorks Training", "url": "https://www.solidworks.com/sw/education/student-software-3d-mcad.htm", "level": "beginner"},
    ],
    "ANSYS": [
        {"title": "ANSYS Learning Hub", "url": "https://www.ansys.com/academic/free-student-products/support-resources", "level": "intermediate"},
        {"title": "ANSYS How To Videos", "url": "https://www.youtube.com/@ANSYSInc", "level": "beginner"},
    ],
    "Simulink": [
        {"title": "MATLAB/Simulink Onramp", "url": "https://matlabacademy.mathworks.com/", "level": "beginner"},
        {"title": "Simulink Docs", "url": "https://www.mathworks.com/help/simulink/getting-started-with-simulink.html", "level": "intermediate"},
    ],
    "Embedded Systems": [
        {"title": "Coursera Embedded Systems Specialization", "url": "https://www.coursera.org/specializations/embedded-systems-security", "level": "intermediate"},
        {"title": "edX Embedded Systems", "url": "https://www.edx.org/learn/embedded-systems", "level": "intermediate"},
    ],
    "FPGA": [
        {"title": "Xilinx FPGA Tutorials", "url": "https://www.xilinx.com/support/university/students.html", "level": "advanced"},
        {"title": "nandland FPGA Guide", "url": "https://nandland.com/", "level": "beginner"},
    ],
    "IoT": [
        {"title": "Coursera IoT Specialization", "url": "https://www.coursera.org/specializations/iot", "level": "beginner"},
        {"title": "Arduino Getting Started", "url": "https://www.arduino.cc/en/Guide", "level": "beginner"},
    ],
    "Bioinformatics": [
        {"title": "Rosalind Bioinformatics Problems", "url": "https://rosalind.info/", "level": "intermediate"},
        {"title": "Coursera Bioinformatics Specialization", "url": "https://www.coursera.org/specializations/bioinformatics", "level": "intermediate"},
    ],

    # Certifications (missing)
    "AWS Certified": [
        {"title": "AWS Skill Builder", "url": "https://skillbuilder.aws/", "level": "intermediate"},
        {"title": "A Cloud Guru AWS Cert Prep", "url": "https://acloudguru.com/aws-cloud-training", "level": "intermediate"},
    ],
    "Google Analytics": [
        {"title": "Google Analytics Academy", "url": "https://analytics.google.com/analytics/academy/", "level": "beginner"},
        {"title": "GA4 Certification", "url": "https://skillshop.exceedlms.com/student/catalog/list?category_ids=6431-google-analytics", "level": "beginner"},
    ],
    "PMP": [
        {"title": "PMI PMP Certification", "url": "https://www.pmi.org/certifications/project-management-pmp", "level": "advanced"},
        {"title": "Google Project Management Certificate", "url": "https://www.coursera.org/professional-certificates/google-project-management", "level": "beginner"},
    ],
    "Tableau Certified": [
        {"title": "Tableau Certification", "url": "https://www.tableau.com/learn/certification", "level": "intermediate"},
        {"title": "Tableau Public Training", "url": "https://public.tableau.com/en-us/s/resources", "level": "beginner"},
    ],
    "Microsoft Certified": [
        {"title": "Microsoft Learn Certifications", "url": "https://learn.microsoft.com/en-us/certifications/", "level": "intermediate"},
        {"title": "AZ-900 Study Guide", "url": "https://learn.microsoft.com/en-us/certifications/exams/az-900", "level": "beginner"},
    ],
    "CompTIA": [
        {"title": "CompTIA CertMaster Learn", "url": "https://www.comptia.org/training/certmaster-learn", "level": "intermediate"},
        {"title": "Professor Messer CompTIA", "url": "https://www.professormesser.com/", "level": "beginner"},
    ],
    "CFA": [
        {"title": "CFA Institute Study Materials", "url": "https://www.cfainstitute.org/en/programs/cfa/exam", "level": "advanced"},
        {"title": "Investopedia CFA Guide", "url": "https://www.investopedia.com/cfa-4689854", "level": "intermediate"},
    ],
}

AVAILABLE_DEGREES = sorted(DEGREE_ROLE_MAP.keys())

# ══════════════════════════════════════════════════════════════════════════════
#  RESUME PARSER
# ══════════════════════════════════════════════════════════════════════════════

import fitz          # pymupdf
from docx import Document as DocxDocument

def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF resume."""
    try:
        doc  = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"  ⚠️  Could not read PDF: {e}")
        return ""


def extract_text_from_docx(path: str) -> str:
    """Extract raw text from a Word (.docx) resume."""
    try:
        doc   = DocxDocument(path)
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"  ⚠️  Could not read DOCX: {e}")
        return ""


def extract_text_from_txt(path: str) -> str:
    """Extract raw text from a plain text resume."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"  ⚠️  Could not read TXT: {e}")
        return ""


def load_resume(path: str) -> str:
    """Load resume text from PDF, DOCX, or TXT."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        print(f"  ⚠️  Unsupported file type: {ext}")
        return ""


def extract_name_from_resume(text: str) -> str:
    """
    Attempt to extract the candidate's name from the top of the resume.
    Falls back to prompting the user if not found.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Name is usually one of the first 3 non-empty lines,
    # is title-cased, has 2-4 words, and contains no digits
    for line in lines[:5]:
        words = line.split()
        if (2 <= len(words) <= 4
                and all(w[0].isupper() for w in words if w)
                and not any(c.isdigit() for c in line)
                and "@" not in line
                and "http" not in line.lower()):
            return line

    return ""


def extract_skills_from_resume(resume_text: str) -> dict:
    """
    Run skill extraction against resume text and split into
    technical and soft skill buckets.
    """
    normalized = normalize_text(resume_text)
    text_lower = normalized.lower()

    found = []
    for skill in SKILLS:
        pattern = rf"\b{re.escape(skill.lower())}\b"
        match   = re.search(pattern, text_lower)
        if match:
            # Get surrounding context to verify it's a real match
            start   = max(0, match.start() - 30)
            end     = min(len(text_lower), match.end() + 30)
            context = text_lower[start:end].replace("\n", " ")
            found.append((skill, context))

    # Deduplicate
    seen       = set()
    found_unique = []
    for skill, context in found:
        if skill not in seen:
            seen.add(skill)
            found_unique.append((skill, context))

    # Split into technical and soft
    technical = [(s, c) for s, c in found_unique if s not in SOFT_SKILLS]
    soft      = [(s, c) for s, c in found_unique if s in SOFT_SKILLS]

    # Print with context so you can verify matches
    print(f"\n  📋 Resume skill extraction results:")
    print(f"  {'─' * 50}")
    print(f"  🔧 Technical ({len(technical)}):")
    for skill, ctx in technical:
        print(f"     ✓ {skill:<30} → ...{ctx}...")

    print(f"\n  🤝 Soft ({len(soft)}):")
    for skill, ctx in soft:
        print(f"     ✓ {skill:<30} → ...{ctx}...")

    # Flag any suspicious single-letter or very short matches
    suspicious = [s for s, _ in found_unique if len(s) <= 2]
    if suspicious:
        print(f"\n  ⚠️  Short skill matches (verify these): {suspicious}")

    all_skills = [s for s, _ in found_unique]
    return {
        "technical": sorted(s for s, _ in technical),
        "soft":      sorted(s for s, _ in soft),
        "all":       all_skills,
        "raw_text":  resume_text,
    }


def parse_resume(path: str) -> dict:
    """
    Full resume parsing pipeline.
    Returns a dict with name, technical skills, soft skills.
    """
    print(f"\n📄 Parsing resume: {path}")

    if not os.path.exists(path):
        print(f"  ❌ File not found: {path}")
        return {}

    text = load_resume(path)
    if not text.strip():
        print("  ❌ Could not extract any text from resume.")
        return {}

    print(f"  ✅ Extracted {len(text):,} characters from resume")

    # Extract name
    name = extract_name_from_resume(text)
    if name:
        print(f"  👤 Detected name: {name}")
    else:
        print("  ⚠️  Could not detect name automatically")

    # Extract skills
    skills = extract_skills_from_resume(text)
    print(f"\n  🔧 Technical skills found ({len(skills['technical'])}):")
    print(f"     {', '.join(skills['technical']) if skills['technical'] else 'None detected'}")
    print(f"\n  🤝 Soft skills found ({len(skills['soft'])}):")
    print(f"     {', '.join(skills['soft']) if skills['soft'] else 'None detected'}")

    return {
        "name":             name,
        "technical_skills": skills["technical"],
        "soft_skills":      skills["soft"],
        "all_skills":       skills["all"],
        "resume_text":      text,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATED DATA COLLECTION  (replaces collect_jobs.py)
# ══════════════════════════════════════════════════════════════════════════════

import requests
import time

API_KEY   = os.getenv("RAPIDAPI_KEY")
NUM_PAGES = 8     # ~80 jobs per run


def build_search_query(season: str) -> str:
    year = datetime.now().year
    if season == "Summer":
        query_year = year if datetime.now().month <= 8 else year + 1
    elif season == "Fall":
        query_year = year if datetime.now().month <= 12 else year + 1
    else:
        query_year = year + 1 if datetime.now().month >= 9 else year
    query = f"{season} Co-op OR Intern Canada {query_year}"
    print(f"🔍 Search query: \"{query}\"")
    return query


def fetch_jobs(query: str) -> list[dict]:
    url     = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key":  API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    all_jobs = []

    for page in range(1, NUM_PAGES + 1):
        print(f"  Fetching page {page}/{NUM_PAGES}...", end=" ")
        params = {
            "query":      query,
            "page":       str(page),
            "num_pages":  "1",
            "date_posted": "all",
        }

        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers,
                                        params=params, timeout=20)
                break
            except requests.exceptions.Timeout:
                if attempt < 2:
                    print(f"\n  ⏱  Timeout, retrying "
                          f"({attempt + 2}/3)...", end=" ")
                    time.sleep(2)
                else:
                    print(f"\n  ⚠️  Page {page} failed — skipping")
                    response = None
            except requests.exceptions.RequestException as e:
                print(f"\n  ⚠️  Network error: {e}")
                response = None
                break

        if response is None:
            continue
        if response.status_code != 200:
            print(f"\n  ⚠️  Error {response.status_code}")
            continue

        jobs = response.json().get("data", [])
        print(f"got {len(jobs)} jobs")
        all_jobs.extend(jobs)
        time.sleep(0.6)

    return all_jobs


def build_dataframe(jobs: list[dict], season: str) -> pd.DataFrame:
    records = []
    for job in jobs:
        records.append({
            "title":           job.get("job_title"),
            "company":         job.get("employer_name"),
            "location_city":   job.get("job_city"),
            "location_state":  job.get("job_state"),
            "is_remote":       job.get("job_is_remote"),
            "employment_type": job.get("job_employment_type"),
            "description":     job.get("job_description"),
            "apply_link":      job.get("job_apply_link"),
            "source":          job.get("job_publisher"),
            "posted_at":       job.get("job_posted_at_datetime_utc"),
            "highlights":      str(job.get("job_highlights", {})),
            "season":          season,
            "scraped_date":    datetime.now().date().isoformat(),
        })

    df = pd.DataFrame(records)
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce", utc=True)
    df["is_remote"] = df["is_remote"].astype(bool)
    df["location_city"]  = df["location_city"].fillna("Unknown")
    df["location_state"] = df["location_state"].fillna("Unknown")
    df["employment_type"] = df["is_remote"].apply(
        lambda x: "Remote" if x else "In person"
    )
    before = len(df)
    df = df.drop_duplicates(subset=["title", "company"],
                             keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  🧹 Removed {removed} duplicate postings")
    return df


def should_refresh(parquet_path: str) -> bool:
    """Check if we already scraped today for this season."""
    if not os.path.exists(parquet_path):
        return True
    today     = datetime.now().date().isoformat()
    lock_file = parquet_path.replace(".parquet", ".lock")

    # Check lock file first — simplest and most reliable
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                last_scraped = f.read().strip()
            if last_scraped == today:
                print(f"\n✅ Already scraped today ({today}) — "
                      f"loading cached data, skipping API call.")
                return False
            else:
                print(f"  📅 Last scraped: {last_scraped} → "
                      f"new day detected, refreshing...")
                return True
        except Exception:
            pass

    # No lock file — first time ever running for this season
    if not os.path.exists(parquet_path):
        print(f"  📭 No data found for this term — fetching now...")
        return True

    # Parquet exists but no lock file — scrape to be safe
    print(f"  ⚠️  Lock file missing — re-fetching to ensure freshness...")
    return True


def write_lock(parquet_path: str):
    """Write today's date to a lock file after successful scrape."""
    today     = datetime.now().date().isoformat()
    lock_file = parquet_path.replace(".parquet", ".lock")
    with open(lock_file, "w") as f:
        f.write(today)
    print(f"  🔒 Lock written: {lock_file} ({today})")

def trim_old_data(parquet_path: str, keep_days: int = 90):
    """
    Keep only the last N days of data in the parquet.
    Prevents unbounded growth while preserving enough
    history for meaningful trend analysis.
    """
    if not os.path.exists(parquet_path):
        return

    try:
        df       = pd.read_parquet(parquet_path)
        before   = len(df)

        if "scraped_date" not in df.columns:
            return

        cutoff   = (datetime.now().date()
                    - pd.Timedelta(days=keep_days)).isoformat()
        df       = df[df["scraped_date"] >= cutoff]
        after    = len(df)
        removed  = before - after

        if removed > 0:
            df.to_parquet(parquet_path, index=False)
            print(f"  🧹 Trimmed {removed} rows older than "
                  f"{keep_days} days ({before} → {after})")
        else:
            print(f"  ✅ No old data to trim ({after} rows kept)")

    except Exception as e:
        print(f"  ⚠️  Could not trim data: {e}")


def collect_and_save(season: str) -> str:
    """
    Scrapes only once per day per season.
    Uses a .lock file to guarantee no repeat fetching.
    """
    parquet_path = f"jobs_{season.lower()}.parquet"

    # ── Gate — exit immediately if already done today ──────────────────────
    if not should_refresh(parquet_path):
        return parquet_path          # ← no API call, returns instantly

    # ── Scrape ─────────────────────────────────────────────────────────────
    print(f"\n📡 Fetching fresh {season} co-op data "
          f"({NUM_PAGES} pages)...\n")
    query = build_search_query(season)
    jobs  = fetch_jobs(query)

    if not jobs:
        if os.path.exists(parquet_path):
            print("⚠️  Fetch failed — using last saved data.")
            write_lock(parquet_path)   # don't retry again today
            return parquet_path
        print("❌ No jobs returned and no cached data. "
              "Check your API key.")
        sys.exit(1)

    print(f"\n📦 Building dataframe from {len(jobs)} raw results...")
    new_df = build_dataframe(jobs, season)

    # Merge with historical data from previous days
    if os.path.exists(parquet_path):
        try:
            existing = pd.read_parquet(parquet_path)
            today    = datetime.now().date().isoformat()
            if "scraped_date" in existing.columns:
                existing = existing[existing["scraped_date"] != today]
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["title", "company"], keep="last"
            ).reset_index(drop=True)
            print(f"  📊 Historical: {len(existing)} | "
                  f"Today: {len(new_df)} | "
                  f"Total: {len(combined)}")
            new_df = combined
        except Exception as e:
            print(f"  ⚠️  Could not merge historical data: {e}")

    # Save parquet + write lock atomically
    new_df.to_parquet(parquet_path, index=False)
    write_lock(parquet_path)

    # Trim after saving — keeps last 90 days only
    trim_old_data(parquet_path, keep_days=90)

    print(f"✅ Saved {len(new_df)} jobs → '{parquet_path}'\n")
    return parquet_path

# ══════════════════════════════════════════════════════════════════════════════
#  USER INPUT
# ══════════════════════════════════════════════════════════════════════════════

def choose_season() -> str:
    seasons = {"1": "Summer", "2": "Fall", "3": "Winter"}
    print("\n  Which co-op season should the report cover?")
    print("  1 — Summer   2 — Fall   3 — Winter")
    while True:
        c = input("  Enter 1, 2, or 3: ").strip()
        if c in seasons:
            return seasons[c]
        print("  ⚠️  Enter 1, 2, or 3.")


def get_student_profile() -> dict:
    """Collect student profile — parse resume or enter skills manually."""
    print("\n" + "=" * 55)
    print("   Co-op Readiness Report — Student Profile")
    print("=" * 55)

    # ── Resume or manual? ─────────────────────────────────────────────────────
    print("\n  How would you like to enter your skills?")
    print("  1 — Upload a resume (PDF, DOCX, or TXT) — recommended")
    print("  2 — Enter skills manually")

    while True:
        mode = input("\n  Enter 1 or 2: ").strip()
        if mode in ["1", "2"]:
            break
        print("  ⚠️  Enter 1 or 2.")

    # ── Resume path ───────────────────────────────────────────────────────────
    parsed_resume = {}
    if mode == "1":
        while True:
            resume_path = input(
                "\n  Enter the full path to your resume\n"
                "  (e.g. /Users/you/Documents/resume.pdf): "
            ).strip()

            # Clean up all common copy-paste artifacts
            resume_path = resume_path.strip("'\"")          # remove surrounding quotes
            resume_path = resume_path.replace("\\ ", " ")   # unescape spaces
            resume_path = resume_path.replace("\\(", "(")   # unescape (
            resume_path = resume_path.replace("\\)", ")")   # unescape )
            resume_path = resume_path.replace("\\[", "[")   # unescape [
            resume_path = resume_path.replace("\\]", "]")   # unescape ]
            resume_path = resume_path.replace("\\&", "&")   # unescape &
            resume_path = resume_path.replace("\\!", "!")   # unescape !
            resume_path = resume_path.strip()

            parsed_resume = parse_resume(resume_path)
            if parsed_resume:
                break
            retry = input("\n  Try a different path? (y/n): ").strip().lower()
            if retry != "y":
                print("  ⚠️  Falling back to manual entry.")
                mode = "2"
                break

    # ── Name ──────────────────────────────────────────────────────────────────
    if parsed_resume.get("name"):
        name_input = input(
            f"\n  Detected name: {parsed_resume['name']}\n"
            f"  Press Enter to confirm or type a correction: "
        ).strip()
        name = name_input if name_input else parsed_resume["name"]
    else:
        name = input("\n  Your full name: ").strip()

    # ── University ────────────────────────────────────────────────────────────
    print("\n  Canadian universities (sample):")
    for i, u in enumerate(CANADIAN_UNIVERSITIES[:10], 1):
        print(f"    {i:2}. {u}")
    print("  (or type your university name directly)")
    university = input("\n  Your university: ").strip()

    # ── Degree ────────────────────────────────────────────────────────────────
    print("\n  Available degrees:")
    for i, d in enumerate(AVAILABLE_DEGREES, 1):
        print(f"    {i:2}. {d}")
    while True:
        d_input = input("\n  Enter degree number or type it: ").strip()
        if d_input.isdigit() and 1 <= int(d_input) <= len(AVAILABLE_DEGREES):
            degree = AVAILABLE_DEGREES[int(d_input) - 1]
            break
        elif d_input in AVAILABLE_DEGREES:
            degree = d_input
            break
        else:
            print("  ⚠️  Not recognised — try the number or exact name.")

    # ── Year of study ─────────────────────────────────────────────────────────
    while True:
        yr = input("\n  Year of study (1–4): ").strip()
        if yr in ["1", "2", "3", "4"]:
            year_of_study = int(yr)
            break
        print("  ⚠️  Enter 1, 2, 3, or 4.")

    # ── Skills ────────────────────────────────────────────────────────────────
    if mode == "1" and parsed_resume:
        technical_skills = parsed_resume["technical_skills"]
        soft_skills      = parsed_resume["soft_skills"]

        # Let student review and add anything that was missed
        print(f"\n  ── Extracted Technical Skills ({len(technical_skills)}) ──")
        print(f"  {', '.join(technical_skills) if technical_skills else 'None'}")
        add_tech = input(
            "\n  Add any missing technical skills (comma separated),\n"
            "  or press Enter to continue: "
        ).strip()
        if add_tech:
            extras = [s.strip() for s in add_tech.split(",") if s.strip()]
            technical_skills = list(set(technical_skills + extras))

        print(f"\n  ── Extracted Soft Skills ({len(soft_skills)}) ──")
        print(f"  {', '.join(soft_skills) if soft_skills else 'None'}")
        add_soft = input(
            "\n  Add any missing soft skills (comma separated),\n"
            "  or press Enter to continue: "
        ).strip()
        if add_soft:
            extras = [s.strip() for s in add_soft.split(",") if s.strip()]
            soft_skills = list(set(soft_skills + extras))

    else:
        # Manual entry
        print("\n  ── Technical Skills ──────────────────────────────────")
        print("  Examples: Python, SQL, Git, Excel, Machine Learning")
        raw_tech         = input("  Your technical skills (comma separated): ")
        technical_skills = [s.strip() for s in raw_tech.split(",")
                            if s.strip()]

        print("\n  ── Soft Skills ───────────────────────────────────────")
        print("  Examples: Communication, Leadership, Teamwork")
        raw_soft    = input("  Your soft skills (comma separated): ")
        soft_skills = [s.strip() for s in raw_soft.split(",") if s.strip()]

    all_skills = list(set(technical_skills + soft_skills))

    print(f"\n  ✅ Profile ready for {name}")
    print(f"     Technical skills: {len(technical_skills)}")
    print(f"     Soft skills:      {len(soft_skills)}")

    return {
        "name":              name,
        "university":        university,
        "degree":            degree,
        "year_of_study":     year_of_study,
        "skills":            all_skills,
        "technical_skills":  technical_skills,
        "soft_skills":       soft_skills,
        "resume_parsed":     mode == "1",
    }

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(season: str) -> pd.DataFrame:
    path = f"jobs_{season.lower()}.parquet"
    if not os.path.exists(path):
        print(f"\n❌ File not found: '{path}'")
        print(f"   Run  python collect_jobs.py  first and choose '{season}'.")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"\n📂 Loaded {len(df)} jobs from '{path}'")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["posted_at"]      = pd.to_datetime(df.get("posted_at"), utc=True,
                                           errors="coerce")
    df["posted_at"]      = df["posted_at"].fillna(
                               pd.Timestamp.now(tz="UTC"))
    df["location_city"]  = df["location_city"].fillna("Unknown")
    df["location_state"] = df["location_state"].fillna("Unknown")
    df["full_text"]      = (df["description"].fillna("") + " "
                            + df["highlights"].fillna(""))
    return df


def normalize_text(t: str) -> str:
    for alias, full in SKILL_ALIASES.items():
        t = re.sub(rf"\b{alias}\b", full, str(t))
    return t


def extract_skills(t: str) -> list:
    t = normalize_text(t).lower()
    return list({s for s in SKILLS
                 if re.search(rf"\b{re.escape(s.lower())}\b", t)})


def clean_text_for_tfidf(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_experience_level(t: str) -> str:
    t = str(t).lower()
    if any(x in t for x in ["0-1 year", "no experience", "entry level",
                              "entry-level", "1st year", "first year"]):
        return "Entry"
    if any(x in t for x in ["1-2 year", "2 year", "some experience",
                              "2nd year", "second year"]):
        return "Junior"
    if any(x in t for x in ["3-4 year", "3 year", "intermediate",
                              "3rd year", "third year"]):
        return "Intermediate"
    if any(x in t for x in ["5+ year", "senior", "advanced",
                              "4th year", "fourth year"]):
        return "Senior"
    return "Any"


def classify_role(title: str) -> str:
    t = str(title).lower()
    if any(x in t for x in ["data scientist", "machine learning",
                              "ml engineer", "ai engineer", "data science"]):
        return "Data Science"
    if any(x in t for x in ["data analyst", "business analyst", "analytics",
                              "business intelligence", "reporting analyst",
                              "business system analyst",
                              "technology business management",
                              "financial analyst", "product analyst",
                              "research analyst", "management reporting",
                              "sales analysis", "geospatial"]):
        return "Data Analytics"
    if any(x in t for x in ["software", "developer", "swe", "backend",
                              "frontend", "full stack", "fullstack",
                              "web developer", "mobile developer",
                              "test automation", "automation engineer",
                              "digital verification", "game designer",
                              "ux", "product design",
                              "digital transformation", "data engineer"]):
        return "Software Engineering"
    if any(x in t for x in ["devops", "cloud", "infrastructure", "platform",
                              "sre", "site reliability", "systems engineer",
                              "network engineer", "automation practice",
                              "ip routing"]):
        return "DevOps/Cloud"
    if any(x in t for x in ["cyber", "security", "governance", "risk",
                              "compliance", "grc", "information security",
                              "soc analyst"]):
        return "Cybersecurity"
    if any(x in t for x in ["product manager", "product management",
                              "program management", "project manager",
                              "business program", "business architecture"]):
        return "Product/Project Management"
    if any(x in t for x in ["finance", "financial", "accounting", "audit",
                              "tax", "investment", "wealth management",
                              "treasury", "actuarial", "investor", "banking",
                              "commercial banking", "credit", "capital",
                              "fund", "insurance", "p&l", "expense"]):
        return "Finance/Accounting"
    if any(x in t for x in ["consulting", "consultant", "advisory",
                              "assurance", "management consulting",
                              "strategy"]):
        return "Consulting"
    if any(x in t for x in ["operations", "supply chain", "logistics",
                              "procurement", "administration",
                              "business development", "entrepreneur",
                              "startup", "account manager", "sales",
                              "customer"]):
        return "Operations/Business"
    if any(x in t for x in ["electrical", "mechanical", "civil", "structural",
                              "chemical", "materials", "aerospace",
                              "manufacturing", "construction",
                              "bim technician", "field technician",
                              "environment", "health & safety", "ehs"]):
        return "Engineering (Non-CS)"
    if any(x in t for x in ["research", "scientist", "lab", "clinical",
                              "biology", "chemistry", "physics",
                              "agriculture", "sustainability"]):
        return "Research/Science"
    if any(x in t for x in ["marketing", "seo", "content", "brand",
                              "growth", "communications", "advancement",
                              "engagement"]):
        return "Marketing"
    return "Other"


def build_features(df: pd.DataFrame):
    """Run full NLP + TF-IDF pipeline. Returns enriched df + tfidf artifacts."""
    print("\n🔤 Extracting skills from job descriptions...")
    df["full_text"]        = df["full_text"].apply(normalize_text)
    df["skills_found"]     = df["full_text"].apply(extract_skills)
    df["skill_count"]      = df["skills_found"].apply(len)
    df["clean_text"]       = df["full_text"].apply(clean_text_for_tfidf)
    df["experience_level"] = df["full_text"].apply(extract_experience_level)
    df["role_category"]    = df["title"].apply(classify_role)

    print(f"   Role distribution:\n"
          f"{df['role_category'].value_counts().to_string()}\n")

    # TF-IDF
    custom_sw = [
        "position", "candidate", "candidates", "role", "opportunity",
        "apply", "application", "team", "work", "working", "experience",
        "develop", "development", "learn", "learning", "term", "co",
        "op", "coop", "internship", "intern", "job", "company",
        "organization", "employer", "employee", "responsibilities",
        "requirements", "qualifications", "preferred", "required",
        "ability", "strong", "excellent", "good", "great", "looking",
        "seeking", "join", "help", "support", "make", "use", "using",
        "knowledge", "understanding", "skill", "skills", "technical",
        "canada", "ontario", "toronto", "2026", "2025",
        "including", "related", "etc", "new", "day", "time", "year",
    ]
    all_sw = list(sk_text.ENGLISH_STOP_WORDS.union(custom_sw))

    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words=all_sw,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )
    tfidf_matrix = tfidf.fit_transform(df["clean_text"])

    return df, tfidf, tfidf_matrix


def build_skill_freq(df: pd.DataFrame) -> pd.DataFrame:
    all_skills = [s for row in df["skills_found"] for s in row]
    freq       = Counter(all_skills)
    skill_df   = pd.DataFrame(freq.most_common(),
                               columns=["skill", "count"])
    skill_df["demand_pct"] = (skill_df["count"] / len(df) * 100).round(1)
    skill_df["type"]       = skill_df["skill"].apply(
        lambda s: "Soft" if s in SOFT_SKILLS else "Technical"
    )
    return skill_df


def build_skill_trends(season: str, window_days: int = 30) -> pd.DataFrame:
    parquet_path = f"jobs_{season.lower()}.parquet"

    if not os.path.exists(parquet_path):
        return pd.DataFrame()

    try:
        df_all = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  Could not load historical data: {e}")
        return pd.DataFrame()

    if "scraped_date" not in df_all.columns:
        return pd.DataFrame()

    # ── Filter to this season only ─────────────────────────────────────────
    if "season" in df_all.columns:
        df_all = df_all[df_all["season"].str.lower() == season.lower()]

    if df_all.empty:
        print(f"  No data found for season: {season}")
        return pd.DataFrame()

    # Only look at last N days
    today  = datetime.now().date()
    cutoff = (today - pd.Timedelta(days=window_days)).isoformat()
    df_all = df_all[df_all["scraped_date"] >= cutoff]

    dates = sorted(df_all["scraped_date"].unique())

    if len(dates) < 2:
        print(f"  Not enough historical data for {season} trends "
              f"(need 2+ days, have {len(dates)})")
        return pd.DataFrame()

    # Extract skills on the fly if not already present
    if "skills_found" not in df_all.columns:
        print(f"  Extracting skills from {season} historical data...")
        df_all["full_text"]    = (df_all["description"].fillna("") + " "
                                   + df_all["highlights"].fillna(""))
        df_all["full_text"]    = df_all["full_text"].apply(normalize_text)
        df_all["skills_found"] = df_all["full_text"].apply(extract_skills)

    # Compute skill demand % per day
    daily_freq = {}
    for date in dates:
        day_df     = df_all[df_all["scraped_date"] == date]
        all_skills = [s for row in day_df["skills_found"] for s in row]
        freq       = Counter(all_skills)
        total_jobs = len(day_df)
        daily_freq[date] = {
            skill: round(count / total_jobs * 100, 1)
            for skill, count in freq.items()
        }

    # Compare earliest vs latest
    earliest = daily_freq[dates[0]]
    latest   = daily_freq[dates[-1]]

    all_skills_seen = set(earliest) | set(latest)
    trends = []
    for skill in all_skills_seen:
        old_pct = earliest.get(skill, 0)
        new_pct = latest.get(skill, 0)
        change  = round(new_pct - old_pct, 1)

        if new_pct == 0 and old_pct == 0:
            continue

        trends.append({
            "skill":     skill,
            "old_pct":   old_pct,
            "new_pct":   new_pct,
            "change":    change,
            "direction": "up" if change > 0 else "down" if change < 0 else "flat",
            "type":      "Soft" if skill in SOFT_SKILLS else "Technical",
        })

    trend_df = pd.DataFrame(trends)
    if trend_df.empty:
        return trend_df

    trend_df = trend_df.sort_values(
        "change", ascending=False, key=abs
    ).reset_index(drop=True)

    print(f"  Trends computed for {season}: "
          f"{len(trend_df)} skills across {len(dates)} days "
          f"({dates[0]} to {dates[-1]})")

    return trend_df

def save_trend_chart(trend_df: pd.DataFrame, season: str) -> str:
    """Save a horizontal bar chart of top rising and falling skills."""
    if trend_df.empty:
        return ""

    rising  = trend_df[trend_df["change"] > 0].head(8)
    falling = trend_df[trend_df["change"] < 0].tail(8)
    combined = pd.concat([rising, falling]).drop_duplicates()
    combined = combined.sort_values("change", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [
        "#2e7d32" if c > 0 else "#c62828"
        for c in combined["change"]
    ]
    bars = ax.barh(combined["skill"], combined["change"],
                    color=bar_colors)

    # Add value labels
    for bar, val in zip(bars, combined["change"]):
        label = f"+{val}%" if val > 0 else f"{val}%"
        x_pos = bar.get_width() + 0.3 if val >= 0 else bar.get_width() - 0.3
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, ha=ha,
                color="#2e7d32" if val > 0 else "#c62828",
                fontweight="bold")

    ax.axvline(x=0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Change in Demand % vs 30 Days Ago")
    ax.set_title(f"Trending Skills — {season} Co-ops (Last 30 Days)",
                  fontweight="bold", fontsize=13)
    ax.set_xlim(combined["change"].min() - 5,
                 combined["change"].max() + 8)
    plt.tight_layout()

    path = f"skill_trends_{season.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Trend chart saved → {path}")
    return path

#SKILL TREND CREATOR

def normalize_skill(skill: str) -> str:
    """Resolve aliases before lookup."""
    skill = skill.lower()
    return SKILL_ALIASES.get(skill, skill)


def get_resources_for_skill(skill: str) -> list[dict]:
    """Return learning resources for a skill - exact match first, then word-boundary fuzzy."""
    skill_lower = skill.lower()
    
    # 1. Exact match first
    for key, resources in SKILL_RESOURCES.items():
        if key.lower() == skill_lower:
            return resources
    
    # 2. Only fuzzy match if the full skill string is contained in the key or vice versa
    #    AND the shorter string is at least 4 chars (avoids "C", "R" false matches)
    for key, resources in SKILL_RESOURCES.items():
        key_lower = key.lower()
        shorter = min(len(skill_lower), len(key_lower))
        if shorter >= 4 and (key_lower in skill_lower or skill_lower in key_lower):
            return resources
    
    return []


def get_resources_for_skills(skills: list[str]) -> dict:
    """Batch lookup — returns a dict of skill -> resources for all matched skills."""
    result = {}
    for skill in skills:
        res = get_resources_for_skill(skill)
        if res:
            result[skill] = res
    return result


def build_rising_resources(trend_df) -> list[dict]:
    """Return rising skills paired with their learning resources."""
    if trend_df.empty:
        return []
    rising = trend_df[trend_df["change"] > 0].head(10)
    result = []
    for _, row in rising.iterrows():
        resources = get_resources_for_skill(row["skill"])
        if resources:
            result.append({
                "skill": row["skill"],
                "change": row["change"],
                "type": row["type"],
                "resources": resources,
            })
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(tfidf_matrix, df: pd.DataFrame):
    le = LabelEncoder()
    y  = le.fit_transform(df["role_category"])
    X  = tfidf_matrix

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred         = rf.predict(X_test)
    present_labels = sorted(set(y_test))
    present_names  = le.classes_[present_labels]

    print("\n📊 Random Forest — Classification Report:")
    print(classification_report(y_test, y_pred,
                                 labels=present_labels,
                                 target_names=present_names,
                                 zero_division=0))
    return rf, le


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENT SCORING
# ══════════════════════════════════════════════════════════════════════════════

def get_year_benchmarks(year: int) -> dict:
    return {
        1: {"strong": 40, "good": 25, "partial": 10},
        2: {"strong": 55, "good": 35, "partial": 20},
        3: {"strong": 70, "good": 50, "partial": 30},
        4: {"strong": 80, "good": 60, "partial": 40},
    }.get(year, {"strong": 55, "good": 35, "partial": 20})


def fit_tier_by_year(score: float, year: int) -> str:
    b = get_year_benchmarks(year)
    if score >= b["strong"]:    return "🟢 Strong Match"
    if score >= b["good"]:      return "🟡 Good Match"
    if score >= b["partial"]:   return "🟠 Partial Match"
    return "🔴 Low Match"


def score_student_against_all_jobs(student_skills: list,
                                    year_of_study: int,
                                    degree: str,
                                    df: pd.DataFrame) -> pd.DataFrame:
    baseline        = DEGREE_BASELINE_SKILLS.get(degree, [])
    combined_skills = list(set(student_skills + baseline))
    combined_lower  = {s.lower() for s in combined_skills}

    relevant_roles  = DEGREE_ROLE_MAP.get(degree)
    eligible_levels = YEAR_TO_LEVEL.get(year_of_study, ["Any"])

    # ── Filter by experience level ────────────────────────────────────────────
    filtered = df[df["experience_level"].isin(eligible_levels)].copy()

    # If experience filter is too strict, fall back to all jobs
    if len(filtered) < 5:
        print(f"  ⚠️  Only {len(filtered)} jobs match experience level "
              f"{eligible_levels} — relaxing filter to all jobs")
        filtered = df.copy()

    # ── Filter by degree/role ─────────────────────────────────────────────────
    if relevant_roles:
        role_filtered = filtered[filtered["role_category"].isin(relevant_roles)]

        # If role filter is too strict, fall back to unfiltered
        if len(role_filtered) < 5:
            print(f"  ⚠️  Only {len(role_filtered)} jobs match roles "
                  f"{relevant_roles} — relaxing role filter")
        else:
            filtered = role_filtered

    print(f"\n🎯 Scoring against {len(filtered)} eligible jobs "
          f"({degree}, Year {year_of_study})...")

    results     = []
    seen_titles = set()

    for _, row in filtered.iterrows():
        job_skills       = set(row["skills_found"])
        job_skills_lower = {s.lower() for s in job_skills}

        if len(job_skills) < 3:
            continue
        key = f"{row['title']}_{row['company']}"
        if key in seen_titles:
            continue
        seen_titles.add(key)

        matched = {s for s in job_skills
                   if s.lower() in combined_lower}
        missing = {s for s in job_skills
                   if s.lower() not in combined_lower}

        baseline_lower = {s.lower() for s in baseline}
        hard_missing   = [s for s in missing
                          if s.lower() not in baseline_lower]
        soft_missing   = [s for s in missing
                          if s.lower() in baseline_lower]

        raw_score       = len(matched) / len(job_skills) * 100
        depth_bonus     = min(len(matched) * 2, 10)
        relevance_bonus = 5 if (relevant_roles and
                                 row["role_category"] in relevant_roles) else 0
        fit_score       = min(raw_score + depth_bonus + relevance_bonus, 100)

        results.append({
            "job_title":        row["title"],
            "company":          row["company"],
            "location":         row["location_city"],
            "is_remote":        row["is_remote"],
            "apply_link":       row["apply_link"],
            "experience_level": row["experience_level"],
            "role_category":    row["role_category"],
            "skills_required":  list(job_skills),
            "skills_matched":   list(matched),
            "skills_missing":   hard_missing,
            "skills_exposure":  soft_missing,
            "fit_score":        round(fit_score, 1),
            "match_count":      f"{len(matched)}/{len(job_skills)}",
            "tier":             fit_tier_by_year(fit_score, year_of_study),
        })

    if not results:
        print("⚠️  No matches found even after relaxing filters.")
        return pd.DataFrame()

    out = (pd.DataFrame(results)
             .sort_values(["fit_score", "match_count"], ascending=False)
             .reset_index(drop=True))

    # Relax the minimum match count if too strict
    out_filtered = out[out["skills_matched"].apply(len) >= 2].reset_index(drop=True)
    if len(out_filtered) < 3:
        print(f"  ⚠️  Only {len(out_filtered)} jobs had 2+ skill matches "
              f"— relaxing to 1+ match")
        out_filtered = out[out["skills_matched"].apply(len) >= 1].reset_index(drop=True)

    return out_filtered


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def save_skill_charts(skill_df: pd.DataFrame, season: str):
    tech_df = skill_df[skill_df["type"] == "Technical"].head(15)
    soft_df = skill_df[skill_df["type"] == "Soft"].head(10)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.barplot(data=tech_df, x="demand_pct", y="skill",
                hue="skill", palette="Blues_r",
                legend=False, ax=axes[0])
    axes[0].set_title(f"Top Technical Skills — {season} Co-ops")
    axes[0].set_xlabel("% of Job Postings")
    axes[0].set_ylabel("")

    sns.barplot(data=soft_df, x="demand_pct", y="skill",
                hue="skill", palette="Greens_r",
                legend=False, ax=axes[1])
    axes[1].set_title(f"Top Soft Skills — {season} Co-ops")
    axes[1].set_xlabel("% of Job Postings")
    axes[1].set_ylabel("")

    plt.suptitle(f"Co-op Skill Demand — {season}", fontsize=14,
                  fontweight="bold")
    plt.tight_layout()

    chart_path = f"skill_demand_{season.lower()}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Chart saved → {chart_path}")
    return chart_path


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(student: dict, season: str,
                         ranked_jobs: pd.DataFrame,
                         skill_df: pd.DataFrame,
                         trend_df: pd.DataFrame,
                         df: pd.DataFrame,
                         output_path: str = "coop_readiness_report.pdf"):
    student_name      = student["name"]
    student_skills    = student["skills"]
    technical_skills  = student.get("technical_skills", student_skills)
    soft_skills_input = student.get("soft_skills", [])
    year_of_study     = student["year_of_study"]
    degree            = student["degree"]
    university        = student["university"]
    baseline          = DEGREE_BASELINE_SKILLS.get(degree, [])

    doc    = SimpleDocTemplate(output_path, pagesize=letter,
                                rightMargin=0.75 * inch,
                                leftMargin=0.75 * inch,
                                topMargin=0.75 * inch,
                                bottomMargin=0.75 * inch)
    story  = []
    styles = getSampleStyleSheet()
    # Fetch learning resources (automated via Claude API)
    # ── Styles ────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "RTitle", parent=styles["Title"], fontSize=22,
        textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=6, alignment=TA_CENTER)
    sub_style = ParagraphStyle(
        "RSub", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4, alignment=TA_CENTER)
    section_style = ParagraphStyle(
        "RSec", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#1a3a5c"),
        spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle(
        "RBody", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#333333"),
        spaceAfter=4, leading=14)
    tag_style = ParagraphStyle(
        "RTag", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#2e6da4"), spaceAfter=2)
    link_style = ParagraphStyle(
        "RLink", parent=styles["Normal"], fontSize=7,
        textColor=colors.HexColor("#2e6da4"),
        spaceAfter=0, leading=10)
    footer_style = ParagraphStyle(
        "RFoot", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#999999"), alignment=TA_CENTER)
    page_title_style = ParagraphStyle(
        "PTitle", parent=styles["Title"], fontSize=18,
        textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=8, alignment=TA_CENTER)
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#444444"),
        spaceAfter=2, leading=11)

    def hr(thick=0.5):
        return HRFlowable(width="100%", thickness=thick,
                           color=colors.HexColor("#cccccc"), spaceAfter=8)

    def footer():
        return [
            Spacer(1, 20),
            HRFlowable(width="100%", thickness=1,
                        color=colors.HexColor("#1a3a5c"), spaceAfter=6),
            Paragraph(
                "Generated by Co-op Skill Gap Analyzer  |  "
                "Data sourced from live job postings",
                footer_style),
        ]
    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 1+  — READINESS REPORT
    # ══════════════════════════════════════════════════════════════════════════

    # ── Header ────────────────────────────────────────────────────────────────
    story += [
        Paragraph("Co-op Readiness Report", title_style),
        Paragraph(f"Prepared for: {student_name}", sub_style),
        Paragraph(
            f"{degree}  |  {university}  |  Year {year_of_study}  |  "
            f"{season} Season  |  "
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            sub_style),
        HRFlowable(width="100%", thickness=2,
                    color=colors.HexColor("#1a3a5c"), spaceAfter=12),
    ]
    # ── Section 1: Student Profile ────────────────────────────────────────────
    story.append(Paragraph("Student Profile", section_style))

    profile_data = [
        ["Name",       student_name],
        ["University", university],
        ["Degree",     degree],
        ["Season",     f"{season} Co-op"],
        ["Year",       f"Year {year_of_study} of Study"],
    ]
    pt = Table(profile_data, colWidths=[1.5 * inch, 5.5 * inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT",  (0, 0), (-1, -1), 20),
        ("TEXTCOLOR",  (0, 0), (0, -1),  colors.HexColor("#1a3a5c")),
    ]))
    story += [pt, Spacer(1, 8)]

    # ── Skills side by side table ──────────────────────────────────────────────
    story.append(Paragraph("Your Skills", section_style))

    max_rows   = max(len(technical_skills), len(soft_skills_input), 1)
    skill_rows = [["Technical Skills", "Soft Skills"]]
    for i in range(max_rows):
        tech = technical_skills[i] if i < len(technical_skills) else ""
        soft = soft_skills_input[i] if i < len(soft_skills_input) else ""
        skill_rows.append([tech, soft])

    skills_table = Table(skill_rows, colWidths=[3.5 * inch, 3.5 * inch])
    skills_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0f4f8"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ALIGN",          (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT",      (0, 0), (-1, -1), 16),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
    ]))
    story += [skills_table, Spacer(1, 8)]

    # Degree exposure note
    extra = [s for s in baseline if s not in student_skills]
    if extra:
        story.append(Paragraph(
            f"<b>Degree Exposure:</b> Based on your {degree} degree, "
            f"you likely have exposure to: {', '.join(extra[:6])}",
            tag_style))
    story.append(Spacer(1, 6))

    # ── Section 2: Market Skill Demand ────────────────────────────────────────
    story += [hr(), Paragraph("Top In-Demand Skills (Market Overview)",
                               section_style)]

    def skill_table(sdf, student_has: list):
        data = [["Skill", "Demand %", "Status"]]
        for _, row in sdf.iterrows():
            has_it = row["skill"] in student_has or row["skill"] in baseline
            data.append([row["skill"], f"{row['demand_pct']}%",
                         "✓ Have it" if has_it else "✗ Missing"])
        t = Table(data, colWidths=[2.8 * inch, 1.5 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f0f4f8"), colors.white]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT",      (0, 0), (-1, -1), 18),
        ]))
        return t

    tech_skill_df = skill_df[skill_df["type"] == "Technical"].head(10)
    soft_skill_df = skill_df[skill_df["type"] == "Soft"].head(6)

    story += [
        Paragraph("<b>Technical Skills in Demand</b>", body_style),
        skill_table(tech_skill_df, technical_skills),
        Spacer(1, 8),
        Paragraph("<b>Soft Skills in Demand</b>", body_style),
        skill_table(soft_skill_df, soft_skills_input),
        Spacer(1, 10),
    ]

    # Embed skill demand chart if it exists
    chart_path = f"skill_demand_{season.lower()}.png"
    if os.path.exists(chart_path):
        story += [
            Paragraph("<b>Skill Demand Visualized</b>", body_style),
            Spacer(1, 6),
            Image(chart_path, width=6.5 * inch, height=2.8 * inch),
            Spacer(1, 10),
        ]
    
    # ── Section 2b: Trending Skills ───────────────────────────────────────────
    if not trend_df.empty:
        story += [hr(), Paragraph(
            f"Trending Skills — Last 30 Days ({season})",
            section_style)]

        story.append(Paragraph(
            "Skills rising or falling in demand based on your "
            "accumulated daily job data. "
            "Use this to prioritise what to learn next.",
            body_style))
        story.append(Spacer(1, 6))

        # Split into rising and falling
        rising  = trend_df[trend_df["change"] > 0].head(8)
        falling = trend_df[trend_df["change"] < 0].head(8)
        stable  = trend_df[trend_df["change"] == 0].head(5)

        def trend_row_color(change):
            if change > 0:   return colors.HexColor("#e8f5e9")  # green tint
            if change < 0:   return colors.HexColor("#ffebee")  # red tint
            return colors.HexColor("#f5f5f5")                   # grey

        trend_table_data = [["Skill", "Type",
                              "30d Ago", "Now", "Change"]]

        for _, row in pd.concat([rising, falling,
                                  stable]).iterrows():
            change_str = (f"+{row['change']}%"
                          if row["change"] > 0
                          else f"{row['change']}%")
            direction_color = (
                colors.HexColor("#2e7d32") if row["change"] > 0
                else colors.HexColor("#c62828") if row["change"] < 0
                else colors.HexColor("#555555")
            )
            trend_table_data.append([
                row["skill"],
                row["type"],
                f"{row['old_pct']}%",
                f"{row['new_pct']}%",
                Paragraph(
                    f"<b>{row['direction']} {change_str}</b>",
                    ParagraphStyle("TrendVal",
                                    parent=styles["Normal"],
                                    fontSize=9,
                                    textColor=direction_color,
                                    fontName="Helvetica-Bold")),
            ])

        trend_table = Table(
            trend_table_data,
            colWidths=[2.2 * inch, 1.0 * inch,
                        0.9 * inch, 0.9 * inch, 1.5 * inch])
        trend_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),
             colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("GRID",           (0, 0), (-1, -1), 0.5,
             colors.HexColor("#dddddd")),
            ("ALIGN",          (2, 0), (-1, -1), "CENTER"),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT",      (0, 0), (-1, -1), 18),
            ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ]))

        # Apply row colours based on trend direction
        for i, (_, row) in enumerate(
                pd.concat([rising, falling, stable]).iterrows(), start=1):
            trend_table.setStyle(TableStyle([
                ("BACKGROUND", (0, i), (-1, i),
                 trend_row_color(row["change"]))
            ]))

        story += [trend_table, Spacer(1, 8)]

        # Embed trend chart if it exists
        trend_chart = f"skill_trends_{season.lower()}.png"
        if os.path.exists(trend_chart):
            story += [
                Spacer(1, 4),
                Image(trend_chart, width=6.5 * inch,
                       height=3.2 * inch),
                Spacer(1, 10),
            ]
    else:
        story += [
            hr(),
            Paragraph("Trending Skills", section_style),
            Paragraph(
                "⏳ Not enough historical data yet to show trends. "
                "Run this tool daily and trends will appear "
                "automatically after 2+ days of data.",
                body_style),
            Spacer(1, 10),
        ]
    # ── Section 2c: Learning Resources for Rising Skills ──────────────────────
    rising_resources = build_rising_resources(trend_df)
    if rising_resources:
        story += [hr(), Paragraph("Learning Resources — Rising Skills", section_style)]
        story.append(Paragraph(
            "These skills are gaining demand in the job market. "
            "Here are curated resources to help you build them:",
            body_style))
        story.append(Spacer(1, 8))
        for item in rising_resources:
            story.append(Paragraph(
                f"<b>{item['skill']}</b> "
                f"<font color='#2e7d32'>▲ +{item['change']}% demand</font> "
                f"<font color='#888888'>({item['type']})</font>",
                ParagraphStyle("ResSkill", parent=styles["Normal"],
                            fontSize=10,
                            textColor=colors.HexColor("#1a3a5c"),
                            fontName="Helvetica-Bold",
                            spaceBefore=6, spaceAfter=3)))
            for res in item["resources"]:
                story.append(Paragraph(
                    f' • <link href="{res["url"]}"><u>{res["title"]}</u></link>'
                    f' — <font color="#555555">{res["url"]}</font>'
                    f' <font color="#888888">[{res["level"]}]</font>',
                    ParagraphStyle("ResLink", parent=styles["Normal"],
                                fontSize=8,
                                textColor=colors.HexColor("#2e6da4"),
                                leading=13, leftIndent=12)))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))
    else:
        story += [
            hr(),
            Paragraph("Learning Resources", section_style),
            Paragraph(
                "Learning resources will appear here once trend data "
                "is available (requires 2+ days of scrapes).",
                body_style),
            Spacer(1, 10),
        ]
    # ── Section 3: Skill Gap ──────────────────────────────────────────────────
    story += [hr(), Paragraph("Skill Gap Analysis", section_style)]

    top_demanded         = set(skill_df.head(20)["skill"])
    student_skills_lower = {s.lower() for s in student_skills}
    baseline_lower       = {s.lower() for s in baseline}
    all_have_lower       = student_skills_lower | baseline_lower

    tech_gaps = sorted(
        s for s in top_demanded
        if s not in SOFT_SKILLS
        and s.lower() not in all_have_lower
    )
    soft_gaps = sorted(
        s for s in top_demanded
        if s in SOFT_SKILLS
        and s.lower() not in all_have_lower
    )

    story.append(Paragraph(
        f"Based on your <b>{degree}</b> degree at <b>{university}</b> "
        f"and your submitted skills, here is your gap against the "
        f"top 20 most demanded {season} co-op skills:",
        body_style))
    story.append(Spacer(1, 6))

    def gap_cell(skill_list):
        if not skill_list:
            return Paragraph(
                "✓ No gaps — great coverage!",
                ParagraphStyle("GapGood", parent=styles["Normal"],
                               fontSize=9,
                               textColor=colors.HexColor("#2e7d32"),
                               leading=14))
        bullets = "<br/>".join(f"• {s}" for s in skill_list)
        return Paragraph(
            bullets,
            ParagraphStyle("GapItem", parent=styles["Normal"],
                           fontSize=9,
                           textColor=colors.HexColor("#333333"),
                           leading=14, leftIndent=4))

    gap_data = [
        [
            Paragraph("<b>Technical Gaps</b>",
                      ParagraphStyle("GapH", parent=styles["Normal"],
                                     fontSize=10,
                                     textColor=colors.white,
                                     fontName="Helvetica-Bold")),
            Paragraph("<b>Soft Skill Gaps</b>",
                      ParagraphStyle("GapH2", parent=styles["Normal"],
                                     fontSize=10,
                                     textColor=colors.white,
                                     fontName="Helvetica-Bold")),
        ],
        [gap_cell(tech_gaps), gap_cell(soft_gaps)],
    ]

    gap_table = Table(gap_data, colWidths=[3.5 * inch, 3.5 * inch])
    gap_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1a3a5c")),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f0f4f8")),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, 0),  6),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  6),
        ("TOPPADDING",    (0, 1), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
    ]))
    story += [gap_table, Spacer(1, 10)]

    # ── Section 4: Top Job Matches ────────────────────────────────────────────
    story += [
        hr(),
        Paragraph(f"Top Job Matches — {degree}, Year {year_of_study},"
                   f" {season}", section_style),
    ]
    relevant_roles = DEGREE_ROLE_MAP.get(degree, [])
    story.append(Paragraph(
        f"Jobs filtered for: "
        f"<b>{', '.join(relevant_roles) if relevant_roles else 'All roles'}</b>",
        body_style))

    job_table_data = [["#", "Job Title & Apply Link",
                        "Company", "Fit Score", "Tier"]]
    for i, row in ranked_jobs.head(10).iterrows():
        title_text = (row["job_title"][:38] + "..."
                      if len(row["job_title"]) > 38 else row["job_title"])
        link_url   = str(row["apply_link"])
        short_url  = link_url[:45] + "..." if len(link_url) > 45 else link_url
        title_cell = [
            Paragraph(title_text, body_style),
            Paragraph(f'<link href="{link_url}"><u>{short_url}</u></link>',
                       link_style),
        ]
        job_table_data.append([
            str(i + 1),
            title_cell,
            row["company"][:18],
            f"{row['fit_score']}%",
            row["tier"].split(" ", 1)[-1],
        ])

    jt = Table(job_table_data,
                colWidths=[0.3 * inch, 3.0 * inch,
                            1.3 * inch, 0.7 * inch, 1.2 * inch])
    jt.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0f4f8"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",          (1, 0), (1, -1),  "LEFT"),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("ROWHEIGHT",      (0, 0), (-1, -1), 30),
        ("TOPPADDING",     (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 1), (-1, -1), 5),
    ]))
    story += [jt, Spacer(1, 10)]

    # Embed job match chart if it exists
    job_chart_path = f"job_matches_{season.lower()}.png"
    if os.path.exists(job_chart_path):
        story += [
            Paragraph("<b>Fit Score Breakdown</b>", body_style),
            Spacer(1, 6),
            Image(job_chart_path, width=6.5 * inch, height=3.5 * inch),
            Spacer(1, 10),
        ]

    # ── Section 5: Recommendations ───────────────────────────────────────────
    story += [hr(), Paragraph("Recommendations", section_style)]
    benchmarks = get_year_benchmarks(year_of_study)
    avg_score  = ranked_jobs["fit_score"].mean() if len(ranked_jobs) else 0
    story.append(Paragraph(
        f"Your average fit score across matched jobs is "
        f"<b>{avg_score:.1f}%</b>. For a Year {year_of_study} "
        f"{degree} student, a strong match is "
        f"<b>{benchmarks['strong']}%+</b>.",
        body_style))
    story.append(Spacer(1, 6))

    # Top technical skills to learn
    top_missing_tech = [
        s for _, r in ranked_jobs.head(20).iterrows()
        for s in r["skills_missing"] if s not in SOFT_SKILLS
    ]
    top_missing_soft = [
        s for _, r in ranked_jobs.head(20).iterrows()
        for s in r["skills_missing"] if s in SOFT_SKILLS
    ]

    tech_freq = Counter(top_missing_tech)
    soft_freq = Counter(top_missing_soft)
    top3_tech = [s for s, _ in tech_freq.most_common(3)]
    top3_soft = [s for s, _ in soft_freq.most_common(3)]

    story.append(Paragraph(
        f"<b>Top technical skills to learn for {season} {degree} co-ops:</b>",
        body_style))
    for i, skill in enumerate(top3_tech, 1):
        story.append(Paragraph(
            f"{i}. <b>{skill}</b> — missing in "
            f"{tech_freq[skill]} of your top matched jobs", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Top soft skills to develop:</b>", body_style))
    for i, skill in enumerate(top3_soft, 1):
        story.append(Paragraph(
            f"{i}. <b>{skill}</b> — missing in "
            f"{soft_freq[skill]} of your top matched jobs", body_style))

    story += footer()


    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL PAGE — ALL JOBS FOUND FOR THIS TERM
    # ══════════════════════════════════════════════════════════════════════════

    from reportlab.platypus import PageBreak

    story.append(PageBreak())

    story += [
        Paragraph(f"All {season} 2026 Co-op Postings Found",
                   page_title_style),
        Paragraph(
            f"Complete list of {len(df)} job postings scraped for the "
            f"{season} 2026 co-op term. Click any link to apply.",
            sub_style),
        HRFlowable(width="100%", thickness=2,
                    color=colors.HexColor("#1a3a5c"), spaceAfter=12),
    ]

    # Build full jobs table — all rows
    all_jobs_data = [["#", "Job Title & Link", "Company",
                       "Location", "Remote"]]

    for i, row in df.iterrows():
        title_text = (str(row["title"])[:42] + "..."
                      if len(str(row["title"])) > 42
                      else str(row["title"]))
        link_url   = str(row.get("apply_link", ""))
        short_url  = link_url[:40] + "..." if len(link_url) > 40 else link_url

        title_cell = [
            Paragraph(title_text, small_style),
            Paragraph(
                f'<link href="{link_url}"><u>{short_url}</u></link>'
                if link_url else "N/A",
                link_style),
        ]

        city    = str(row.get("location_city", "Unknown"))
        remote  = "Yes" if row.get("is_remote", False) else "No"
        company = str(row.get("company", ""))[:20]

        all_jobs_data.append([
            str(i + 1),
            title_cell,
            company,
            city[:18],
            remote,
        ])

    all_jobs_table = Table(
        all_jobs_data,
        colWidths=[0.3 * inch, 3.2 * inch, 1.4 * inch,
                    1.1 * inch, 0.5 * inch],
        repeatRows=1,       # repeat header row on each page
    )
    all_jobs_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f7fa"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ALIGN",          (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",          (4, 0), (4, -1),  "CENTER"),
        ("ALIGN",          (1, 0), (1, -1),  "LEFT"),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",     (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 1), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
    ]))

    story += [all_jobs_table]
    story += footer()

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"\n✅ PDF report saved → {output_path}")
    return output_path

def save_job_match_chart(ranked_jobs: pd.DataFrame,
                          student_name: str, season: str) -> str:
    top20 = ranked_jobs.head(20).copy()
    top20["label"] = (top20["job_title"].str[:35] + "..."
                      ).where(top20["job_title"].str.len() > 35,
                               top20["job_title"]) + "\n@ " + top20["company"]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top20["label"], top20["fit_score"],
                    color=plt.cm.Blues(
                        [s / 100 for s in top20["fit_score"]]
                    ))

    # Add score labels on bars
    for bar, score in zip(bars, top20["fit_score"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score}%", va="center", fontsize=8)

    ax.axvline(x=50, color="red", linestyle="--",
                alpha=0.4, label="50% threshold")
    ax.set_xlabel("Fit Score (%)")
    ax.set_title(f"Top Job Matches — {student_name} ({season})",
                  fontweight="bold")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()

    path = f"job_matches_{season.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Job match chart saved → {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 55)
    print("   Co-op Job Market — Report Generator")
    print("=" * 55)

    # 1. Choose season
    season = choose_season()

    # 2. Auto-collect data (scrapes fresh if needed, otherwise loads cache)
    collect_and_save(season)

    # 3. Load parquet
    df = load_data(season)
    df = clean_dataframe(df)
    df, tfidf, tfidf_matrix = build_features(df)

    # 4. Skill frequency table
    skill_df = build_skill_freq(df)
    trend_df  = build_skill_trends(season)
    # Save trend chart
    if not trend_df.empty:
        save_trend_chart(trend_df, season)  
    print(f"\n🏆 Top 10 skills in {season} postings:")
    print(skill_df.head(10)[["skill", "demand_pct"]].to_string(index=False))

    # ── Section 4b: Job Match Chart ───────────────────────────────────────────
    job_chart_path = f"job_matches_{season.lower()}.png"
    if os.path.exists(job_chart_path):
        story += [
            Paragraph("<b>Fit Score Breakdown</b>", body_style),
            Spacer(1, 6),
            Image(job_chart_path, width=6.5 * inch, height=3.5 * inch),
            Spacer(1, 10),
        ]

    # 5. Train Random Forest
    train_random_forest(tfidf_matrix, df)

    # 6. Save skill demand charts
    save_skill_charts(skill_df, season)

    # 7. Collect student profile
    student = get_student_profile()

    # 8. Score student against all jobs
    ranked_jobs = score_student_against_all_jobs(
        student_skills=student["skills"],
        year_of_study=student["year_of_study"],
        degree=student["degree"],
        df=df,
    )

    if ranked_jobs.empty:
        print("\n⚠️  No ranked jobs — PDF will be skipped.")
        sys.exit(0)

    print(f"\n📋 Top 5 matches for {student['name']}:")
    print(ranked_jobs[["job_title", "company",
                         "fit_score", "tier"]].head(5).to_string(index=False))

    # 9. Generate PDF
    safe_name   = student["name"].replace(" ", "_").lower()
    output_path = f"coop_report_{safe_name}_{season.lower()}.pdf"
    generate_pdf_report(
        student=student,
        season=season,
        ranked_jobs=ranked_jobs,
        skill_df=skill_df,
        trend_df=trend_df,
        df=df,
        output_path=output_path,
    )

    print(f"\n🎉 Done!  Open  {output_path}  to view your report.\n")


if __name__ == "__main__":
    main()
