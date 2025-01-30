import os
import yaml
import logging
import logging_config 
import functools
from box import Box
from datetime import datetime
from dateutil.parser import parse

# 3rd party
import pandas as pd
import numpy as np

# Internal
from db_utils import minio_utils
import invoice_utils as iu

# functions
@functools.cache
def validation_set() -> pd.DataFrame: 
    """
    Retrieves the validation set from the minio bucket and performs some data cleaning and transformation.
    # maybe an async function can call the table but continue with others while this runs in background
    """
    
    val_df = minio_utils.minio_to_dataframe(
        minio_bucket=config.bucket.validation_data,
        filename_prefix_override="current"
        )
    val_df['invoice_file_name'] = val_df.invoice_file_name.str.split('/').str.get(-1)
    val_df['invoice_file_name'] = val_df.invoice_file_name.str.split('.').str.get(0)
    val_df['invoice_file_name'] = val_df.invoice_file_name.str.lower()
    val_df.head()

    cols = {
        'vendor_invoice_ref':'vendor_invoice_id',
        'invoice_document_date':'invoice_date'
    }
    val_df = val_df.rename(columns=cols).reset_index(drop=True)

    '''
    For this we need to be make sure of 1:1 ratio and check that this is the correct table
    I would suggest that we have this data pre-linked in a bucket and not do a merge like this.
    '''
    bank_df = minio_utils.minio_to_dataframe(
        minio_bucket=config.bucket.vendor_acc,
        filename_prefix_override='current'
    )

    val_df = val_df.merge(bank_df[['vendor_id','bank_name','bank_branch_code','bank_account_number']],how='left',on='vendor_id')
    return val_df

def validate_invoice(llm_clean: dict, val_set: dict) -> dict:
    '''
    We need to check for the following:
    1. PO number
    2. Vat number
    3. Bank details
    4. Vendor name
    5. Invoice date
    '''
    codes = []
    val_set['purchase_order_number'] = val_set['purchase_order_number'].strip().replace(' ','')
    if val_set['purchase_order_number'] == llm_clean['purchase_order_number']:
        codes.append(f'PO_VAL_PASS')
    else:
        codes.append(f'PO_VAL_FAIL')
    
    # banking
    val_set['bank_name'] = val_set['bank_name'].lower().strip().replace(' ','')
    if iu.get_match(llm_clean['bank_name'], val_set['bank_name']):
        codes.append(f'BANK_VAL_PASS')
    else:
        codes.append(f'BANK_VAL_FAIL')
    
    val_set['bank_branch_code'] = val_set['bank_branch_code'].strip().replace(' ','')
    if val_set['bank_branch_code'] == llm_clean['bank_branch_code']:
        codes.append(f'BANK_BRANCH_VAL_PASS')
    else:
        codes.append(f'BANK_BRANCH_VAL_FAIL')
    
    val_set['bank_account_number'] = val_set['bank_account_number'].strip().replace(' ','')
    if val_set['bank_account_number'] == llm_clean['bank_account_number']:
        codes.append(f'BANK_ACCOUNT_VAL_PASS')
    else:
        codes.append(f'BANK_ACCOUNT_VAL_FAIL')
    
    val_set['vendor_name'] = val_set['vendor_name'].lower().strip().replace(' ','')
    if iu.get_match(llm_clean['vendor_name'], val_set['vendor_name']):
        codes.append(f'VENDOR_NAME_VAL_PASS')
    else:
        codes.append(f'VENDOR_NAME_VAL_FAIL')
    
    val_set['invoice_date'] = val_set['invoice_date'].strip().replace(' ','')
    if val_set['invoice_date'] == llm_clean['invoice_date']:
        codes.append(f'INVOICE_DATE_VAL_PASS')
    else:
        codes.append(f'INVOICE_DATE_VAL_FAIL')
    
    return codes


if __name__ == "__main__":
    # Paths
    dir = os.path.dirname(__file__)
    os.chdir(dir)

    # Config
    config = Box.from_yaml(filename="../config/config.yml", Loader=yaml.FullLoader)

    # logging
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists('img'):
        os.makedirs('img')
    
    logger = logging.getLogger(__name__)
    logger = logging.getLogger('invoice_extraction')
    logger.info("Starting extractions!")

    #val_df = validation_set()

    #pdf_path = ''
    #val_set = val_df.query('Some match')

    text = iu.text_extraction(pdf_path)
    #checks = iu.text_checks(text, val_set)
    #if 'PASS' in checks.all():
    llm_raw = iu.llm_key_extraction(text)
    llm_clean = iu.val_clean(llm_raw)

        #codes = validate_invoice(llm_clean, val_set)
        #if 'PASS' in codes.all():
        #    logger.info("Validation passed!")
        #else:
        #    logger.error("Validation failed!")

