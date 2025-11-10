from __future__ import annotations

import os

import google.generativeai as genai


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    embed_model = "models/gemini-embedding-001"
    generator_model = "models/gemini-2.5-flash"
    judge_model = "models/gemini-2.5-flash"

    genai.configure(api_key=api_key)

    print("Testing embed model:", embed_model)
    embed_request = genai.embed_content(model=embed_model, content="hello world")
    vector = embed_request["embedding"]
    print("Embedding length:", len(vector))

    generator = genai.GenerativeModel(generator_model)
    answer = generator.generate_content("Say hello in a sentence.")
    print("Generator output:", answer.text.strip() if hasattr(answer, "text") else answer)

    judge = genai.GenerativeModel(judge_model)
    judge_prompt = (
        "Respond with JSON {\"score\": 1}.\n\nQuestion: What is 2+2?\n"
        "Context: 2+2 equals 4.\nAnswer: 4"
    )
    judge_response = judge.generate_content(judge_prompt)
    print("Judge output:", judge_response.text.strip() if hasattr(judge_response, "text") else judge_response)


if __name__ == "__main__":
    main()

