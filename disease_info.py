from dotenv import load_dotenv
load_dotenv()  # loads .env

import os
from openai import OpenAI

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def fetch_description(disease_name, client=None):
    if client is None:
        client = get_openai_client()
    if client is None:
        return "OpenAI API key missing; cannot fetch description."

    # Make the disease label more readable
    readable_name = disease_name.replace("___", " ").replace("_", " ")

    prompt = (
        f"You are an expert plant pathologist. Give a short, clear description of the plant disease "
        f"'{readable_name}', including typical symptoms and prevention tips. Keep it concise (3–4 sentences)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in plant diseases and crop health."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=180,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Failed to fetch description from OpenAI: {e})"
