import os
from openai import OpenAI
from pypdf import PdfReader

# --- SET YOUR API KEY ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_all_pdfs(folder_path):
    """
    Reads all PDF files inside a folder and concatenates their text.
    """
    full_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            print(f"Reading: {filename}")
            reader = PdfReader(path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n\n"
    return full_text

def split_into_three_parts(text, max_chars=700000):
    parts = []
    for i in range(0, len(text), max_chars):
        parts.append(text[i:i+max_chars])
    print("Total parts:", len(parts))
    return parts

def process_part_with_gpt5mini(part_text):
    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Below is part of a medical ECG textbook. 

Lightly compress redundant lines while KEEPING ALL medical meaning.
DO NOT summarize. DO NOT remove any clinical content.

Text:
{part_text}
"""
    )
    return response.output_text

def generate_ecg_guide(combined_text):
    prompt = f"""
You are an expert cardiologist and medical educator. Using the content below
(from multiple ECG textbooks):

{combined_text}

Create a comprehensive **ECG Interpretation Guide** in English, using correct
and professional medical terminology.

Requirements:
- Structure the guide into clear sections.
- Include the physiology behind ECG waves.
- Explain systematic ECG interpretation:
  * Rate
  * Rhythm
  * P wave morphology
  * PR interval
  * QRS duration & morphology
  * QT/QTc interval
  * ST segment abnormalities
  * T wave abnormalities
  * U waves
- Include diagnostic criteria for:
  * Chamber enlargement (LAE, RAE, LVH, RVH)
  * Bundle branch blocks (RBBB, LBBB)
  * Fascicular blocks (LAFB, LPFB)
  * Supraventricular arrhythmias (AF, AFL, SVT)
  * Ventricular arrhythmias (VT, VF, PVCs)
  * AV blocks (1st, 2nd Mobitz I/II, 3rd degree)
- Provide a detailed section on **Acute Myocardial Infarction (STEMI/NSTEMI)**:
  * STEMI localization by leads
  * Reciprocal changes
  * Evolution stages (hyperacute, acute, subacute, chronic)
- Include quick-reference tables & bullet-point algorithms.
- Format as a clean, structured, medical-style teaching document.

Output ONLY the final ECG Guide.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text

# Step 1: Read PDFs
raw_text = read_all_pdfs("Books")

# Step 2: Split into 3 large but safe parts
parts = split_into_three_parts(raw_text)

# Step 3: Light compress each part (NO summarization)
print("\nProcessing part 1...")
p1 = process_part_with_gpt5mini(parts[0])

print("\nProcessing part 2...")
p2 = process_part_with_gpt5mini(parts[1])

print("\nProcessing part 3...")
p3 = process_part_with_gpt5mini(parts[2])

# Step 4: Combine processed parts
combined_text = p1 + "\n\n" + p2 + "\n\n" + p3

# Step 5: Final ECG Interpretation Guide
print("\nGenerating FINAL ECG GUIDE...")
final_guide = generate_ecg_guide(combined_text)

# Step 6: Save to file
with open("ECG_Interpretation_Guide.txt", "w", encoding="utf-8") as f:
    f.write(final_guide)

print("Saved ECG_Interpretation_Guide.txt")
