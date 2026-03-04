import os
import json
import xml.etree.ElementTree as ET

input_dir = "datasets/raw/medquad/MedQuAD-master"
output_file = "datasets/processed/medical_knowledge.json"

qa_pairs = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            tree = ET.parse(file_path)
            root_xml = tree.getroot()

            for qa in root_xml.iter("QAPair"):
                q = qa.find("Question")
                a = qa.find("Answer")

                if q is not None and a is not None:
                    qa_pairs.append({
                        "question": q.text.strip(),
                        "answer": a.text.strip()
                    })

        except:
            pass

print("Total pairs extracted:", len(qa_pairs))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(qa_pairs[:500], f, indent=2)

print("Saved to:", output_file)
