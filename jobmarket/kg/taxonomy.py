"""Keyword taxonomies used to derive Role and Skill nodes from free text."""

import re


def compile_alternation(aliases: list[str]) -> re.Pattern:
    """One compiled pattern matching any alias, with symbol-aware boundaries
    (so it works for 'C++', 'CI/CD', 'Node.js', not just plain words)."""
    body = "|".join(re.escape(a) for a in aliases)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE)


# Ordered most-specific-first: the first canonical role whose pattern matches wins.
ROLE_TAXONOMY = [
    ("Machine Learning Engineer", ["machine learning engineer", "ml engineer", "applied ai engineer", "ai engineer"]),
    ("Data Scientist", ["data scientist"]),
    ("Data Engineer", ["data engineer"]),
    ("Data Analyst", ["data analyst"]),
    ("DevOps Engineer", ["devops engineer", "site reliability engineer", "sre"]),
    ("Cloud Engineer", ["cloud engineer"]),
    ("Embedded Software Engineer", ["embedded"]),
    ("QA / Test Engineer", ["qa engineer", "quality assurance", "software engineer in test", "test engineer"]),
    ("Full Stack Engineer", ["full stack", "full-stack", "fullstack"]),
    ("Software Engineer", ["software engineer", "software developer"]),
]
ROLE_PATTERNS = [(role, compile_alternation(aliases)) for role, aliases in ROLE_TAXONOMY]


def classify_role(title: str) -> str:
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(title):
            return role
    return "Other"


# Curated tech/data skill vocabulary. Deliberately excludes bare single-word tokens with high
# false-positive risk in prose (e.g. "R", "Go", "node") — only unambiguous aliases are listed.
SKILL_VOCABULARY: dict[str, list[str]] = {
    "Python": ["python"],
    "SQL": ["sql"],
    "Java": ["java"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "Scala": ["scala"],
    "C++": ["c++"],
    "C#": ["c#"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Terraform": ["terraform"],
    "Linux": ["linux"],
    "Git": ["git"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous deployment"],
    "Spark": ["spark"],
    "Hadoop": ["hadoop"],
    "Kafka": ["kafka"],
    "Airflow": ["airflow"],
    "dbt": ["dbt"],
    "ETL": ["etl"],
    "Snowflake": ["snowflake"],
    "Redshift": ["redshift"],
    "BigQuery": ["bigquery"],
    "Data Warehousing": ["data warehouse", "data warehousing"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "NoSQL": ["nosql"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Natural Language Processing": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Large Language Models": ["large language model", "large language models", "llm", "llms"],
    "Generative AI": ["generative ai", "genai", "gen ai"],
    "Retrieval-Augmented Generation": ["retrieval-augmented generation", "retrieval augmented generation", "rag"],
    "Vector Database": ["vector database", "vector store", "vector search"],
    "Prompt Engineering": ["prompt engineering"],
    "React": ["react", "react.js", "reactjs"],
    "Node.js": ["node.js", "nodejs"],
    "REST API": ["rest api", "restful"],
    "GraphQL": ["graphql"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Excel": ["excel"],
    "Statistics": ["statistics", "statistical modeling"],
    "A/B Testing": ["a/b testing", "ab testing"],
    "Agile": ["agile", "scrum"],
    "Salesforce": ["salesforce"],
}
SKILL_PATTERNS = [(skill, compile_alternation(aliases)) for skill, aliases in SKILL_VOCABULARY.items()]


def extract_skills(text: str) -> set[str]:
    return {skill for skill, pattern in SKILL_PATTERNS if pattern.search(text)}
