import os
import time
import pandas as pd
from google.cloud import translate_v2 as translate
from google.auth.exceptions import GoogleAuthError
import requests

# Set the environment variable for Google credentials (replace with your JSON key file)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Siddhi\Desktop\kau\hallowed-digit-439812-g7-be1a8cffe16f.json"

# Initialize the translation client
translate_client = translate.Client()

def translate_text(text, target_language="mr"):
    """
    Translate text to the target language (default: Marathi).
    Retries up to 2000 times with a 45-second wait between retries in case of network issues.
    """
    retries = 2000
    wait_time = 45  # seconds
   
    for attempt in range(retries):
        try:
            if isinstance(text, bytes):
                text = text.decode('utf-8')
           
            # Perform the translation
            result = translate_client.translate(text, target_language=target_language)
            return result['translatedText']
        except (GoogleAuthError, requests.exceptions.RequestException) as e:
            print(f"Error occurred: {e}. Retrying... Attempt {attempt+1}/{retries}")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:
            print(f"Failed to translate: {e}")
            return None  # Return None if translation fails

    print("Max retries exceeded. Translation failed.")
    return None

def process_file(input_file, output_file, target_language="mr"):
    """
    Read the input CSV file, translate the 'input_text' column, and save the results to the output CSV file.
    Updates progress every 1000 rows.
    """
    try:
        # Load the CSV file with proper encoding
        df = pd.read_csv(input_file, encoding='utf-8')  
        print(f"Loaded {len(df)} rows from {input_file}")

        translated_texts = []
        translated_count = 0

        for index, row in df.iterrows():
            text_to_translate = row['input_text']
            translated_text = translate_text(text_to_translate, target_language)

            # Check if translation succeeded and append to list
            if translated_text:
                translated_texts.append(translated_text)
                translated_count += 1

            # Print progress after every 1000 rows
            if (index + 1) % 1000 == 0:
                print(f"Progress: {index + 1} rows translated.")

        # Add translated texts to the DataFrame
        df['translated_text'] = translated_texts

        # Save the translated DataFrame to a new CSV file
        df.to_csv(output_file, index=False, encoding='utf-8-sig')  
        print(f"Translation completed. Translated data saved to {output_file}.")
        print(f"Total rows translated: {translated_count}/{len(df)}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Update input and output file paths
input_file = r"C:\Users\Siddhi\Desktop\kau\t5_prepared_data.csv"
output_file = r"C:\Users\Siddhi\Desktop\kau\translated_t5_prepared_data.csv"

process_file(input_file, output_file)
