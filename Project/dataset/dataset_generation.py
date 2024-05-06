import os
import requests
from bs4 import BeautifulSoup
import pdfkit
from tqdm import tqdm
import re

# URL list
urls = [
    "https://velocityhospital.com/scope-of-services/",
    "https://velocityhospital.com/patient-rights-responsibilities-velocity-hospital/",
    "https://velocityhospital.com/visiting-consultants/",
    "https://velocityhospital.com/contact-velocity-hospital/",
    "https://velocityhospital.com/facilities/",
    "https://velocityhospital.com/health-checkup/",
    "https://velocityhospital.com/gastrointestinal-surgery-velocity-hospital/",
    "https://velocityhospital.com/general-medicine-and-infectious-diseases-velocity-hospital/",
    "https://velocityhospital.com/neurology-velocity-hospital/",
    "https://velocityhospital.com/neurosurgery-velocity-hospital/",
    "https://velocityhospital.com/nutrition-and-dietetics-velocity-hospital/",
    "https://velocityhospital.com/obstetrics-gynecology-infertility-velocity-hospital/",
    "https://velocityhospital.com/oncosurgery-velocity-hospital/",
    "https://velocityhospital.com/orthopedics-velocity-hospital/",
    "https://velocityhospital.com/pediatrics-neonatology-velocity-hospital/",
    "https://velocityhospital.com/physiotherapy-velocity-hospital/",
    "https://velocityhospital.com/plastic-reconstructive-surgery-velocity-hospital/",
    "https://velocityhospital.com/urosurgery-velocity-hospital/",
    "https://velocityhospital.com/arthroscopy-sports-medicine-velocity-hospital/",
    "https://velocityhospital.com/critical-care-emergency-medicine-velocity-hosptal/",
    "https://velocityhospital.com/general-laparoscopic-surgery-velocity-hospital/",
    "https://velocityhospital.com/maxillofacial-surgery-velocity-hospital/",
    "https://velocityhospital.com/doctors-velocity-hospital/"
]

# Create a directory to store PDF files if it doesn't exist
if not os.path.exists("pdf_files"):
    os.makedirs("pdf_files")

# Function to extract text from HTML
def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    # Add punctuation where needed
    text = re.sub(r'(?<=[a-zA-Z0-9])\n', '. ', text)  # Add period after a word if it's followed by a newline
    text = re.sub(r'\n(?=[a-zA-Z0-9])', '. ', text)  # Add period before a word if it's preceded by a newline
    return text

# Function to save HTML content to PDF with progress bar
def save_to_pdf_with_progress(url, pbar):
    response = requests.get(url)
    if response.status_code == 200:
        text = extract_text_from_html(response.content)
        filename = f"pdf_files/{url.split('/')[-2]}.pdf"
        pdfkit.from_string(text, filename)
        # Update progress bar description
        pbar.set_description(f"Saving {filename}")
        # Update progress bar position
        pbar.update(1)

# Main function to loop through URLs and save to PDF
def main():
    # Initialize tqdm progress bar
    with tqdm(total=len(urls)) as pbar:
        # Loop through URLs
        for url in urls:
            # Call save_to_pdf_with_progress function
            save_to_pdf_with_progress(url, pbar)

if __name__ == "__main__":
    main()
