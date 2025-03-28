
from datasets import load_dataset
import re

ds = load_dataset("refugee-law-lab/canadian-legal-data", "default")

"""

   :param text: Text from unofficial_text category in dataset
   :return: cleaned text

   Main idea: All Canadian legal documents have this feature where they list the
   """

def clean_text_between_citation_and_signature(text):
    citation_pattern = r'(\d{4}\s+CHRT\s+\d{1,2})\s*\n(\d{4}/\d{2}/\d{2})'
    signature_pattern = r'Signed by\s+[A-Za-z\s\-éçàèùâêîôûëïüÿ]+(?:")?\s*\n[A-Za-z\s]+,\s+[A-Za-z\s]+\s*\n(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'

    citation_match = re.search(citation_pattern, text, re.MULTILINE)
    signature_match = re.search(signature_pattern, text, re.MULTILINE)

    if citation_match and signature_match:
        start_pos = citation_match.start()
        end_pos = signature_match.end()
        cleaned_text = text[start_pos:end_pos]
    elif citation_match:
        start_pos = citation_match.start()
        cleaned_text = text[start_pos:]
    elif signature_match:
        end_pos = signature_match.end()
        cleaned_text = text[:end_pos]
    else:
        cleaned_text = text

    return cleaned_text

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Loading dataset...")

    # Load dataset

    print("Cleaning text...")


    # Use map to update the dataset
    def clean_text(example):
        example['unofficial_text'] = clean_text_between_citation_and_signature(example['unofficial_text'])
        return example


    ds = ds.map(clean_text)

    # Save the cleaned dataset
    ds.save_to_disk("cleaned_data.hf")
    print("Dataset saved to 'cleaned_data.hf'")

