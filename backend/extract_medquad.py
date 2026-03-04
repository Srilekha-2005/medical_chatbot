import os
import json
import xml.etree.ElementTree as ET

input_dir = "datasets/raw/medquad/MedQuAD-master"
output_file = "datasets/processed/medical_knowledge.json"

qa_pairs = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".xml"):
            file_path = os.path.join(root, file)

            try:
                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                for qa in root_xml.iter("QAPair"):
                    question = qa.find("Question")
                    answer = qa.find("Answer")

                    if question is not None and answer is not None:
                        qa_pairs.append({
                            "question": question.text.strip(),
                            "answer": answer.text.strip()
                        })

            except:
                pass

print("Total pairs:", len(qa_pairs))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(qa_pairs[:500], f, indent=2)