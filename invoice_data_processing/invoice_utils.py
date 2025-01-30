import json
import logging
import os
import re
import typing
from dateutil.parser import parse

import cv2
import fitz
import pdfplumber
import pytesseract
import numpy as np
from openai import OpenAI
from anthropic import Anthropic
from fuzzywuzzy import fuzz, process

#SS mod
#from db_utils import minio_utils
#from invoice_regex import patterns

#SS mod
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

logger = logging.getLogger(__name__)

TEMPERATURE = 0

#SS mod
import re

patterns = {
    "issue_date": re.compile(r'((?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{2,4})|(?:\d{2,4}[-/]\d{1,2}[-/]\d{1,2}))',re.IGNORECASE),
    "amount": re.compile(r"[a-zA-Z£$R,\s/]",re.IGNORECASE),
    "vendor_registration_number": re.compile(r'[\d/]'),
    "vendor_tax_id": re.compile(r'([0-9]+)'),
    "bank_branch_code": re.compile(r'(\d{6})'),
    "purchase_order_number": re.compile(r'(\d{10})'),
    "bank_account_number": re.compile(r'(\d{8,16})')
    }


# Functions
def clean_json_string(messy_string: str) -> str:
    """
    Cleans a JSON string by removing newline characters and extra spaces.

    Args:
        messy_string (str): The JSON string to be cleaned.

    Returns:
        str: The cleaned JSON string.

    """
    cleaned_string = messy_string.replace('\n', '').replace('    ', '')
    return cleaned_string

def find_brackets(string: str) -> str:
    """
    Find the content enclosed in curly brackets ('{}') in the given string.

    Args:
        string (str): The input string.

    Returns:
        str: The content enclosed in brackets.
    """
    opening_bracket_index: int = string.find('{')
    closing_bracket_index: int = string.rfind('}')

    string: str = string[opening_bracket_index:closing_bracket_index + 1]
    string = clean_json_string(string)

    return string

def pull_invoice_pdf(inv_name, bucket):
    """
    Pulls an invoice PDF from a specified bucket and saves it locally.

    Args:
        inv_name (str): The name of the invoice file in the bucket.
        bucket (str): The name of the bucket where the invoice file is located.

    Returns:
        str: The local path where the invoice PDF was saved.
    """
    save_path = inv_name.split('/')[-1]
    save_path = os.path.join('pdfs',save_path)
    minio_utils.minio_to_file(filename=save_path,
                                minio_filename_override=inv_name,
                                minio_bucket=bucket
                                )
    return save_path

def pdf_to_text(path):
    """
    Extracts text from a PDF file located at the specified path.

    Args:
        path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        text = ''
        for page in pages:
            text += page.extract_text(layout=True) 
    return text

def convert_pdf_to_images(pdf_path: str) -> typing.List[str]:
    """
    Convert a PDF to a sequence of images and save them in the output directory.
    Returns a list of image file paths.

    Args:
        pdf_path (str): The path to the PDF file.
        output_dir (str): The output directory to save the images.

    Returns:
        List[str]: A list of image file paths.
    """
    dpi: int = 800  # choose desired dpi here
    zoom: float = dpi / 72  # zoom factor, standard: 72 dpi
    magnify: fitz.Matrix = fitz.Matrix(zoom, zoom)
    img_name_list: typing.List[str] = []

    doc: fitz.Document = fitz.open(pdf_path)
    for idx, page in enumerate(doc):
        pix: fitz.Pixmap = page.get_pixmap(matrix=magnify)
        img_name: str = f"img/{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{idx+1}.png"
        pix.save(img_name)
        img_name_list.append(img_name)

    return img_name_list

def extract_text_from_image(pdf_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Returns the extracted text.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text from the image.
    """
    img_name_list = convert_pdf_to_images(pdf_path)
    text = ''
    for image_path in img_name_list:
        img: np.ndarray = cv2.imread(image_path)
        os.remove(image_path)
        img_gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_thresh: np.ndarray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        custom_config: str = r'-l eng --oem 1 --psm 6'
        text += pytesseract.image_to_string(img_thresh, config=custom_config)

    return text.strip()

def text_checks(text: str, val_set: dict) -> list:
    """
    A function that checks the presence of purchase order number and vendor name in the given text.
    
    Args:
        text (str): The text to be checked.
        val_set (dict): A dictionary containing 'purchase_order_number' and 'vendor_name' keys.
        
    Returns:
        list: A list of error codes based on the text checks.
    """
    po_no = val_set['purchase_order_number']
    vendor_name = val_set['vendor_name']
    err_code = []
    if po_no in text:
        logger.info("Text PO check passed!")
        err_code.append('PO_RAW_PASS')
    else:
        logger.error("Text PO check failed!")
        err_code.append('PO_RAW_FAIL')
    if vendor_name in text:
        logger.info("Text Vendor name check passed!")
        err_code.append('VENDOR_RAW_PASS')
    else:
        logger.error("Text Vendor name check failed!")
        err_code.append('VENDOR_RAW_FAIL')
    return err_code

def text_extraction(invoice_path: str) -> str:
    """
    Extracts text from the provided invoice_path by first attempting to extract text from a PDF using pdf_to_text,
    then falling back to extract text from an image using extract_text_from_image if the PDF extraction fails.

    Args:
        invoice_path (str): The path to the invoice file.

    Returns:
        str: The extracted text from the invoice.
    """
    print("text_extraction function running")
    text = pdf_to_text(invoice_path)
    if text == '':
        print("pdf_to_text failed, trying text_from_image")
        text = extract_text_from_image(invoice_path)
        print(text)
        if text == '':
            print("failed to extract text from pdf_to_text or text_from_image")
            #logger.error(f"Failed to extract text from invoice {invoice_path}")
    else:
        print("received text from pdf")
        #logger.error(f"Failed to extract text from invoice {invoice_path}")    
    
    return text

# LLM
def llm_key_extraction(input_text: str, model: str = 'gpt-4o') -> dict:
    # gpt-4o, gpt-4o-mini, gpt-3.5-turbo, claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-haiku-20240307
    # 
    """
    Extracts specific key fields from the given input text using a language model.

    Args:
        input_text (str): The text from which to extract the key fields.

    Returns:
        dict: A dictionary containing the extracted key fields in snake case as keys and their corresponding values as strings.
              The keys are:
              - vendor_invoice_id: The invoice number or document number or tax invoice number.
              - purchase_order_number: Your reference number or purchase order number or external order number.
              - vendor_tax_id: The vendor VAT or tax reference number.
              - vendor_registration_number: The vendor/company registration number.
              - invoice_date: The issue date as a date format.
              - total_amount: The total amount.
              - net_amount: The net amount.
              - tax_amount: The VAT or tax amount.
              - vendor_address: The address of the vendor (excluding city of Cape Town).
              - bank_name: The name of the bank.
              - bank_branch_code: The bank branch code.
              - bank_account_number: The bank account number.
              - sort_code: The bank sort code.
              - vendor_name: The company name or vendor name.

              If the extraction fails, an empty dictionary is returned.

    Raises:
        Exception: If the JSON object creation fails.

    Note:
        The language model used is GPT-4o.
    """

    system_prompt = """
        You are a helpful assistant with expert knowledge of the layout of invoice documents. 
        Your task is extract only the key fields specified by the user, from the invoice text and only output these as a valid JSON object. 
    """

    user_prompt = f"""
    Question:
    Extract the following information as a JSON object with keys in snake case and extracted values as strings for an invoice text document:
        - Invoice number or document number or tax invoice number (vendor_invoice_id): [extracted value]
        - Your reference number or purchase order number or external order number (a 10 digit code that starts with a 450) (purchase_order_number): [extracted value]
        - Vendor VAT or tax reference number (a 10 digit code) (vendor_tax_id): [extracted value]
        - vendor/company registration number (vendor_registration_number): [extracted value]
        - Issue date (invoice_date) as a date fromat: [extracted value]
        - Total (total_amount): [extracted value]
        - Net amout (net_amount): [extracted value]
        - VAT or tax amount (tax_amount): [extracted value]
        - Address: [extracted value] (exclude city of cape town, look for vendor address)
        - Bank name Common banks start with ABSA, FNB, Standard Bank, Nedbank, and Capitec (bank_name): [extracted value]
        - Bank branch code (bank_branch_code): [extracted value (digit)]
        - Bank account number (bank_account_number): [extracted value (digit)]
        - Bank sort code (sort_code): [extracted value (digit)]
        - Company name or vendor name (vendor_name): [extracted value]
    
        from the following invoice text: {input_text}
    """
    
    logger.info("Starting LLM key extraction")


    try:
        if model.startswith('gpt'):
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            client = OpenAI()
            #client = Anthropic() claude-3-5-sonnet-20240620
            response = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                messages=messages
            )

            response = response.choices[0].message.content

        elif model.startswith('claude'):
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            client = Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=5000,
                temperature=TEMPERATURE,
                messages=messages,
                system=system_prompt
            )
            response = response.content[0].text
        
        response = find_brackets(response)
        response = json.loads(response)
        logger.info("Text converted to JSON object")
        return response
    
    except Exception as e:
        logger.error(f'Failed to create JSON object {e}')

# Key and val clean up
def key_clean(input_dict: dict) -> dict:
    """
    Cleans the keys of a dictionary by mapping the keys to a new dictionary with standardized keys.

    Args:
        input_dict (dict): The dictionary to clean.

    Returns:
        dict: The cleaned dictionary.

    """
    new_dict = {
        "vendor_name": "",
        "vendor_invoice_id": "",
        "invoice_date": "",
        "vendor_tax_id": "",
        "vendor_registration_number": "",
        "purchase_order_number": "",
        "bank_name": "",
        "bank_account_number": "",
        "bank_branch_code": "",
        "bank_sort_code": "",
        "account_holder_name": "",
        "net_amount": None,
        "total_amount": None,
        "tax_amount": None,
        "address": "",
        "line_items": []
    } 

    for key, val in input_dict.items():
        match key:
            case _ if 'vendor' in key and 'name' in key:
                new_dict['vendor_name'] = val
            case _ if 'invoice' in key and 'id' in key:
                new_dict['vendor_invoice_id'] = val
            case _ if 'date' in key:
                new_dict['invoice_date'] = val
            case _ if 'vendor' in key and 'tax' in key:
                new_dict['vendor_tax_id'] = val
            case _ if 'vendor' in key and 'registration' in key:
                new_dict['vendor_registration_number'] = val
            case _ if 'purchase' in key or 'order' in key:
                new_dict['purchase_order_number'] = val
            case _ if 'bank' in key and 'name' in key:
                new_dict['bank_name'] = val
            case _ if 'bank' in key and 'account' in key:
                new_dict['bank_account_number'] = val
            case _ if 'bank' in key and 'branch' in key:
                new_dict['bank_branch_code'] = val
            case _ if 'bank' in key and 'sort' in key:
                new_dict['bank_sort_code'] = val
            case _ if 'account' in key and 'name' in key:
                new_dict['account_holder_name'] = val
            case _ if 'net' in key and 'amount' in key:
                new_dict['net_amount'] = val
            case _ if 'total' in key and 'amount' in key:
                new_dict['total_amount'] = val
            case _ if 'tax' in key and 'amount' in key:
                new_dict['tax_amount'] = val
            case _ if 'address' in key:
                new_dict['address'] = val
            case _ if 'items' in key:
                new_dict['line_items'].append(val)

    return new_dict

def val_clean(input_dict: dict) -> dict:
    """
    Cleans and standardizes the values in the input dictionary based on specific keys, returning a new dictionary with cleaned values.
    
    Args:
        input_dict (dict): The input dictionary to be cleaned.
        
    Returns:
        dict: A dictionary with cleaned and standardized values.
    """
    
    new_dict = key_clean(input_dict)

    for key, val in new_dict.items():
        match key:
            case 'vendor_name':
                try:
                    new_dict[key] = val.strip().lower()
                except Exception as e:
                    new_dict[key] = val
                    
            case 'vendor_invoice_id':
                try:
                    val = val.replace(' ','').upper()
                    new_dict[key] = val.strip()
                except Exception as e:
                    new_dict[key] = val

            case 'invoice_date':
                try:
                    val = val.replace('.','-')
                    match = re.search(patterns['issue_date'], val)
                    new_dict[key] = parse(match.group(0))
                    new_dict[key] = new_dict[key].strftime("%Y-%m-%d")
                except Exception as e:
                    new_dict[key] = val          
                
            case 'vendor_tax_id':
                try:
                    val = val.replace(" ",'').replace('-','')
                    match = re.search(patterns['vendor_tax_id'], val)
                    new_dict[key] = match.group(0)
                except Exception as e:
                    new_dict[key] = val
                    
            case 'vendor_registration_number':
                try:
                    val = val.replace(' ','')
                    match = re.findall(patterns['vendor_registration_number'], val)
                    if match:
                        new_dict[key] = ''.join(match)
                except Exception as e:
                    new_dict[key] = val
                    
            case 'purchase_order_number':
                try:
                    val = val.replace(' ','')
                    match = re.search(patterns['purchase_order_number'], val)
                    new_dict[key] = match.group(0)
                except Exception as e:
                    new_dict[key] = val

            case 'bank_name':
                try:
                    new_dict[key] = val.strip().lower()
                except Exception as e:
                    new_dict[key] = val
                    
            case 'bank_account_number':
                try:
                    val = val.replace(" ",'').replace('-','')
                    match = re.search(patterns['bank_account_number'], val)
                    new_dict[key] = match.group(0)
                except Exception as e:
                    new_dict[key] = val
                    
            case 'bank_branch_code':
                try:
                    val = val.replace(" ",'').replace('-','')
                    match = re.search(patterns['bank_branch_code'], val)
                    new_dict[key] = match.group(0)
                except Exception as e:
                    new_dict[key] = val

            case 'bank_sort_code':
                try:
                    new_dict[key] = val.strip()
                except Exception as e:
                    new_dict[key] = val

            case 'account_holder_name':
                try:
                    new_dict[key] = val.strip()
                except Exception as e:
                    new_dict[key] = val
                    
            case 'net_amount':
                try:
                    val = re.sub(patterns['amount'],'',val)
                    val = float(val)
                    new_dict[key] = val
                except Exception as e:
                    new_dict[key] = val
                    
            case 'total_amount':
                try:
                    val = re.sub(patterns['amount'],'',val)
                    val = float(val)
                    new_dict[key] = val
                except Exception as e:
                    new_dict[key] = val
                    
            case 'tax_amount':
                try:
                    val = re.sub(patterns['amount'],'',val)
                    val = float(val)
                    new_dict[key] = val
                except Exception as e:
                    new_dict[key] = val
                    
            case 'address':
                try:
                    new_dict[key] = val.strip().lower()
                except Exception as e:
                    new_dict[key] = val
            
            case 'line_items':
                try:
                    new_dict[key] = val
                except Exception as e:
                    continue

    return new_dict

# Function to compare two strings and get the match score
def get_match(string1: str, string2: str, partial_ratio: float = 0.8) -> bool:
    """
    A function that compares two strings and returns a boolean based on the partial ratio comparison.
    
    Args:
        string1 (str): The first string for comparison.
        string2 (str): The second string for comparison.
        partial_ratio (float): The threshold for the partial ratio.

    Returns:
        bool: True if the partial ratio of string1 and string2 is greater than the provided threshold, False otherwise.
    """
    partial_match = fuzz.partial_ratio(string1, string2)

    if partial_match > partial_ratio:
        return True

    else:
        return False

