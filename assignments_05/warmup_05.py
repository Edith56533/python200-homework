
# API Q1: First OpenAI API call

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)


# API Q2: Temperature experiments

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\nTemperature {temp}:")
    print(response.choices[0].message.content)
# Higher temperatures tend to produce more varied and creative answers. Temperature 0 is best when consistency is important, but it does not guarantee identical outputs every time.


# API Q3: Multiple completions in one request

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices, start=1):
    print(f"\nPanda Fact {i}:")
    print(choice.message.content)


# API Q4: Limiting response length

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work in detail."
        }
    ],
    max_tokens=15
)
print("\nAPI Q4 - Limited Response:")
print(response.choices[0].message.content)
# The response is cut short because the model is limited to 15 output tokens.Token limits help control API costs and keep responses short.


# --- System Messages and Personas ---
# System Q1: Comparing two AI personalities

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. Explain things simply and end with encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what tokenization is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nPatient Tutor:")
print(response.choices[0].message.content)

# Change the system message to a completely different personality

messages[0]["content"] = (
    "You are a sarcastic, impatient programming instructor. "
    "Keep explanations short, blunt, and slightly humorous."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print("\nSarcastic Instructor:")
print(response.choices[0].message.content)
# The system message changes the model's tone and teaching style, even though the user's question stays exactly the same.


# System Q2: Conversation history

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "My name is Karen and I'm learning Python."
    },
    {
        "role": "assistant",
        "content": "Nice to meet you, Karen! Python is a great choice. What would you like to work on?"
    },
    {
        "role": "user",
        "content": "Can you remind me what my name is?"
    }
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print("\nSystem Q2 - Conversation Memory:")
print(response.choices[0].message.content)
# The model knows my name (Karen) because the previous messages were included in the request, not because the API stores memories.



#  Prompt Engineering 
# Prompt Q1: Zero-shot sentiment classification

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this review as positive, negative, or mixed. "
                    f"Reply with only the sentiment.\nReview: {review}"
                )
            }
        ]
    )
    print(f"\nReview {i}:")
    print(response.choices[0].message.content)


# Prompt Q2: One-shot sentiment classification

for i, review in enumerate(reviews, start=1):
    prompt = f"""
    Classify the review as positive, negative, or mixed.
    Follow the format shown in this example.
    Example:
    Review: "Fast shipping but the item arrived damaged."
    Sentiment: mixed
    Review: "{review}"
    Sentiment:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\nOne-Shot Review {i}:")
    print(response.choices[0].message.content)

# The example helps demonstrate the expected format. Zero-shot worked well, but one-shot provides a clearer pattern to follow.


# Prompt Q3: Few-shot sentiment classification

for i, review in enumerate(reviews, start=1):
    prompt = f"""
    Classify each review as positive, negative, or mixed.
    Follow the examples below.
    Example 1:
    Review: "Excellent service and friendly staff."
    Sentiment: positive
    Example 2:
    Review: "The product broke after one day."
    Sentiment: negative
    Example 3:
    Review: "Affordable price, but delivery was slow."
    Sentiment: mixed
    Review: "{review}"
    Sentiment:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\nFew-Shot Review {i}:")
    print(response.choices[0].message.content)

# Zero-shot is useful for simple tasks that need no examples.One-shot helps establish the desired format. Few-shot is useful when several examples can clarify the categories. 


# Prompt Q4: Chain of thought

prompt = """
A data engineer earns $85,000 per year.
She gets a 12% raise, then 6 months later takes a new job
that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
Show the calculation steps clearly and label the final answer.
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("\nPrompt Q4 - Salary Calculation:")
print(response.choices[0].message.content)
# Breaking a problem into calculation steps can reduce errors and makes the final answer easier to verify.


# Prompt Q5: Structured JSON output

import json
review = (
    "I've been using this tool for three months. "
    "It handles large datasets well, but the UI is clunky "
    "and the export options are limited."
)
prompt = f"""
Analyze the following review.
Return ONLY valid JSON with these three keys:
- sentiment: positive, negative, or mixed
- confidence: a number between 0 and 1
- reason: one sentence explaining the classification
Review: {review}
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
raw_response = response.choices[0].message.content
print("\nPrompt Q5 - Raw JSON:")
print(raw_response)
try:
    data = json.loads(raw_response)
    print("\nParsed JSON:")
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
except json.JSONDecodeError:
    print("Invalid JSON received:")
    print(raw_response)


# Prompt Q6: Using delimiters

user_text = (
    "First boil a pot of water. Once boiling, add a handful "
    "of salt and the pasta. Cook for 8-10 minutes until "
    "al dente. Drain and toss with your sauce of choice."
)
def format_instructions(text):
    prompt = f"""
    You will be given text inside triple backticks.
    If it contains step-by-step instructions, rewrite them
    as a numbered list.
    If it does not contain instructions, respond with exactly:
    "No steps provided."
    ```{text}```
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
print("\nPrompt Q6 - Instructions:")
print(format_instructions(user_text))
regular_text = (
    "The weather was beautiful yesterday. "
    "The birds were singing in the trees."
)
print("\nPrompt Q6 - Regular Text:")
print(format_instructions(regular_text))
# Delimiters distinguish input data from instructions. They help reduce confusion and prompt injection risks, although they can't prevent every injection attempt.



# --- Local Models with Ollama ---
# Ollama Q1: Local model versus OpenAI

# Ollama output: """ A large language model is a type of artificial intelligence that can understand, generate, and process natural language, allowing it to perform tasks like writing, answering questions, or creating content. It's built with massive datasets and trained on vast amounts of text to improve accuracy and adaptability across various domains. """
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)
print("\nOllama Q1 - OpenAI Response:")
print(response.choices[0].message.content)
# Both models explain the same basic concept, but their wording may differ. One advantage of Ollama is that it can run locally without API charges. One disadvantage is that local models use computer resources and smaller models may produce less accurate or detailed answers.
