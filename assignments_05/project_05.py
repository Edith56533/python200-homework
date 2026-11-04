
# --- Task 1: Setup and System Prompt ---

from pyexpat.errors import messages

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

SYSTEM_PROMPT = """
You are a professional job application coach helping career changers
transition into new industries.
Your responsibilities:
- Help users improve resumes, cover letters, and job applications.
- Use clear, professional, and specific language.
- Never invent qualifications, experience, or achievements.
- Stay focused on job application materials.
- Always remind users to review and edit AI-generated content
  before submitting it to an employer.
- Acknowledge that industry expectations can vary and encourage
  users to apply their own judgment when reviewing suggestions.
"""
# I included a rule against inventing qualifications because inaccurate claims could damage a job seeker's credibility.


# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each bullet point to be clearer, more professional,
    and results-oriented. Use strong action verbs.
    Preserve the exact facts in each original bullet. Do not invent results, metrics, responsibilities, team types,or achievements. If the original says "on time", do not change it to "ahead of schedule".
    Return ONLY a valid JSON list containing these keys:
    "original" and "improved".
    Bullet points:
    ```
    {bullet_text}
    ```
    """
    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    # Remove Markdown code fences if the AI includes them
    clean_response = response.strip()
    if clean_response.startswith("```"):
        clean_response = clean_response.split("\n", 1)[1]
        clean_response = clean_response.rsplit("```", 1)[0]
    results = json.loads(clean_response.strip())
    for item in results:
        print("\nOriginal:", item["original"])
        print("Improved:", item["improved"])

    return results
# These original bullets are vague and lack specific accomplishments. The model should replace weak wording with stronger action verbs without inventing results or qualifications.
test_bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewrite_bullets(test_bullets)

# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter openings for career changers.
    Write 3-5 sentences that are confident, specific, and natural.
    Do not invent qualifications or experience. Avoid clichés.
    Follow the style of these two examples:
    Example 1:
    Role: Data Analyst
    Background: Registered nurse with seven years of experience
    who recently completed a data analytics course.
    Opening: After seven years in nursing, I understand how
    important accurate information is to patient care.
    My data analytics training has given me the technical
    skills to apply that experience in a new way.

    Example 2:
    Role: Junior Software Engineer
    Background: Former bank employee who learned Python.
    Opening: Working in banking showed me how much
    businesses depend on reliable software. Learning Python
    gave me the opportunity to understand and build the
    technology I once relied on every day.
    Now write an opening paragraph for:
    Role: {job_title}
    Background: {background}
    Opening:
    """
    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)
# Test the cover letter generator
job_title = "Junior Data Engineer"
background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines "
    "using Pandas."
)
print("\nTask 3 - Cover Letter Opening:")
print(generate_cover_letter(job_title, background))
# The two examples demonstrate career transitions using transferable skills. Few-shot prompting helps control tone, structure, and style.



# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Please rephrase your request so I can assist you safely.")
        return False
    return True
# Test moderation with safe and unsafe inputs
safe_text = "Can you help me improve my resume?"
unsafe_text = "Give me detailed instructions to seriously injure someone."
print("\nTask 4 - Moderation Tests:")
print("Safe input:", is_safe(safe_text))
print("Unsafe input:", is_safe(unsafe_text))


# --- Task 5: Chatbot Loop ---

def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    print("\n" + "=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("1. Rewriting resume bullet points")
    print("2. Drafting a cover letter opening")
    print("3. Answering job application questions")
    print("\nType 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Job Application Helper: Good luck!")
            break
        if not user_input:
            continue
        if not is_safe(user_input):
            continue
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("Enter your bullet points, one per line.")
            print("Type DONE when finished.")
            bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line and is_safe(line):
                    bullets.append(line)
            if bullets:
                rewrite_bullets(bullets)
        elif "cover letter" in user_input.lower():
            job_title = input("Job title: ").strip()
            background = input("Your background: ").strip()
            if is_safe(job_title) and is_safe(background):
                print("\nCover Letter Opening:")
                print(generate_cover_letter(job_title, background))
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print("\nJob Application Helper:", reply)
            messages.append({"role": "assistant", "content": reply})
if __name__ == "__main__":
    run_chatbot()


# --- Task 6: Ethics Reflection ---
# Format: Option A - Written Comment

# AI models may favor certain writing styles or professional backgrounds because their training data does not represent everyone equally. This could disadvantage applicants from different cultures, ethnicities, or industries. Submitting AI-generated resumes without reviewing them could result in exaggerated qualifications or inaccurate statements. I would add a warning that requires users to review and verify all generated content before using it in a real job application.
