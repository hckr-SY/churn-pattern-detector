import subprocess
import json

def generate_insights_with_ollama(df, similarity_methods):
    """
    Given a dataframe of churned rows, generate overall insights using Ollama local LLM.
    This calls Ollama CLI with a prompt built from churn data.
    """

    # Build prompt text for LLM
    prompt_text = f"Analyze the following churn data and provide insights on patterns:\n\n"
    prompt_text = (
    "You are a data analyst. Analyze the following dataset comparing old and new customer records. "
    "Each row represents a customer with potential changes in attributes such as address, city, and pincode. "
    "The columns ending in `_similarity` represent column-wise similarity scores, and `overall_similarity` represents the weighted sum.\n\n"
    "The `churn_flag` column indicates whether a significant change (i.e., churn) was detected.\n\n"
    "Your task:\n"
    "1. Identify the most common types of changes (e.g., address tweaks, city changes).\n"
    "2. Highlight any geographic patterns (e.g., customers moving from specific cities or pin codes).\n"
    "3. Suggest what these patterns might indicate (e.g., relocation, service dissatisfaction).\n"
    "4. Mention if changes appear minor (like abbreviations) or major (like different regions).\n"
    "5. Recommend what kind of features or alerts a business should monitor to detect future churn.\n\n"
    "Dataset:\n"
)


   

    prompt_text += df.to_string(index=False)
    # prompt_text += "\n\nConsider similarity methods: " + ", ".join(similarity_methods)

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],  
            input=prompt_text.encode('utf-8'),
            capture_output=True,
            check=True
        )
        insights = result.stdout.decode("utf-8").strip()
    except Exception as e:
        insights = f"Error calling Ollama LLM: {str(e)}"

    return insights

