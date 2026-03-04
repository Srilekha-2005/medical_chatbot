import os
import json
import re

input_dir = "datasets/raw/medquad/MedQuAD-master"
output_file = "datasets/processed/medical_knowledge.json"

qa_pairs = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

                questions = re.findall(r"<Question>(.*?)</Question>", content, re.DOTALL)
                answers = re.findall(r"<Answer>(.*?)</Answer>", content, re.DOTALL)

                for q, a in zip(questions, answers):
                    qa_pairs.append({
                        "question": q.strip(),
                        "answer": a.strip()
                    })

        except:
            pass

print("Total pairs extracted:", len(qa_pairs))

os.makedirs("datasets/processed", exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(qa_pairs[:500], f, indent=2)

print("Saved to:", output_file)
