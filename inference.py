import torch
from transformers import BartTokenizer, BartForConditionalGeneration

model_path = "clinical_summarizer_model"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def summarize_clinical_note(note_text):
    inputs = tokenizer(
        note_text,
        return_tensors="pt",
        truncation=True,
        max_length=384
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=90,
            min_length=40,
            num_beams=5,
            no_repeat_ngram_size=3,
            length_penalty=1.3,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    while True:
        note = input("\nEnter clinical note (type 'exit' to stop):\n")
        if note.lower() == "exit":
            break
        print("\nSummary:\n")
        print(summarize_clinical_note(note))