"""Rebuild the deterministic synthetic demo dataset using only the standard library."""
import csv
from pathlib import Path


# Each role has two plausible work areas and several optional requirements.
ROLE_PROFILES = [
    ("AI Analyst", ["Python", "SQL", "Pandas", "Machine Learning"],
     ["Excel", "Power BI", "Scikit-learn", "NumPy"],
     ["evaluate prediction quality and explain AI-assisted business insights",
      "prepare labeled datasets and investigate errors in classification models"]),
    ("Data Analyst", ["SQL", "Excel", "Power BI"],
     ["Python", "Pandas", "Tableau", "Statistics"],
     ["investigate reporting discrepancies and build operational dashboards",
      "clean transaction data and present weekly performance trends"]),
    ("Business Analyst", ["Excel", "SQL", "Requirements Analysis"],
     ["Power BI", "Tableau", "Agile", "Data Visualization"],
     ["translate stakeholder needs into acceptance criteria and process reports",
      "map business processes and measure the impact of product changes"]),
    ("Machine Learning Engineer", ["Python", "Machine Learning", "Scikit-learn", "NumPy"],
     ["Pandas", "Docker", "AWS", "FastAPI"],
     ["build reproducible training pipelines and monitor model performance",
      "deploy prediction services and validate models against held-out data"]),
    ("Data Scientist", ["Python", "Pandas", "Machine Learning", "Statistics"],
     ["SQL", "NumPy", "Scikit-learn", "Data Visualization"],
     ["design experiments and develop forecasting models with measurable baselines",
      "explore customer behavior and communicate predictive modeling results"]),
    ("Software Engineer", ["Java", "SQL", "Git"],
     ["REST APIs", "Spring Boot", "Docker", "Unit Testing"],
     ["implement application features and review tested service code",
      "resolve production defects and improve database-backed applications"]),
    ("Frontend Developer", ["React", "JavaScript", "HTML", "CSS"],
     ["TypeScript", "Git", "REST APIs", "Unit Testing"],
     ["build accessible responsive interfaces and reusable UI components",
      "integrate browser interfaces with APIs and optimize page performance"]),
    ("Backend Developer", ["Node.js", "JavaScript", "REST APIs", "PostgreSQL"],
     ["Docker", "AWS", "Express.js", "MongoDB"],
     ["design authenticated service endpoints and optimize database queries",
      "build reliable asynchronous services and maintain integration tests"]),
    ("Full Stack Developer", ["React", "Node.js", "JavaScript", "REST APIs"],
     ["MongoDB", "PostgreSQL", "Git", "Docker"],
     ["deliver web features across user interfaces and backend services",
      "build internal web applications with secure data access"]),
    ("Java Developer", ["Java", "SQL", "Spring Boot"],
     ["REST APIs", "PostgreSQL", "Git", "Unit Testing"],
     ["implement transactional services and maintain enterprise integrations",
      "develop enterprise APIs and troubleshoot service performance"]),
    ("Python Developer", ["Python", "SQL", "REST APIs"],
     ["FastAPI", "PostgreSQL", "Git", "Docker"],
     ["build data processing services and automate recurring operational tasks",
      "implement tested APIs and maintain scheduled ingestion jobs"]),
    ("Cloud Engineer", ["AWS", "Linux", "Terraform"],
     ["Docker", "Python", "Kubernetes", "Git"],
     ["provision cloud infrastructure and monitor availability and cost",
      "configure secure cloud networks and automate environment setup"]),
    ("DevOps Engineer", ["AWS", "Docker", "CI/CD", "Linux"],
     ["Kubernetes", "Terraform", "Git", "Python"],
     ["maintain release pipelines and improve deployment reliability",
      "automate container deployments and investigate infrastructure incidents"]),
]

LOCATIONS = ["Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi",
             "Noida", "Chennai", "Gurugram", "Kolkata"]
COMPANIES = ["AsterByte Labs", "Cedar Data Systems", "Nila Analytics",
             "OrbitLeaf Technologies", "PrismBridge Software", "MapleGrid Digital",
             "RiverPeak Solutions", "CoralStack Systems", "SilverFern Cloud",
             "MangoTree Insights", "QuartzPath Technologies", "BlueKite Digital",
             "CopperLane Labs", "LotusWave Software", "AmberNest Analytics",
             "SaffronLoop Systems", "IvoryArc Digital", "PebbleCloud Labs"]
DOMAINS = ["retail inventory", "logistics operations", "education platforms",
           "subscription billing", "travel bookings", "manufacturing quality",
           "customer support", "energy usage", "insurance operations",
           "warehouse planning", "supplier management", "online marketplaces"]


def generate_jobs():
    rows = []
    for role_index, (title, core, optional, responsibilities) in enumerate(ROLE_PROFILES):
        # Unequal role counts make role and skill demand charts more informative.
        for variant in range(8 + role_index % 5):
            extras = [optional[variant % len(optional)]]
            if variant % 3:
                extras.append(optional[(variant + 1) % len(optional)])
            skills = core + extras
            domain = DOMAINS[(role_index + variant) % len(DOMAINS)]
            description = (
                f"Join our {domain} team to {responsibilities[variant % 2]}. "
                f"Use {', '.join(skills)} in day-to-day delivery. "
                f"Collaborate with product and engineering teams, document decisions, "
                f"and validate work before release. "
                f"Suitable for candidates with {1 + variant % 4} to {3 + variant % 4} "
                f"years of relevant experience."
            )
            rows.append({
                "job_title": title,
                "company": COMPANIES[(role_index * 3 + variant) % len(COMPANIES)],
                "location": LOCATIONS[(role_index + variant) % len(LOCATIONS)],
                "job_description": description,
                "skills": ", ".join(skills),
            })
    return rows


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "jobs.csv"
    rows = generate_jobs()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic jobs to {output}")
