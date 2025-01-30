import os
import yaml
import logging
import logging_config
import functools
from datetime import datetime
from dateutil.parser import parse

# 3rd party
from box import Box
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
    
    config = Box.from_yaml(filename="../config/config.yml", Loader=yaml.FullLoader) # to make this allways accessible
    
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
    6. Total amount
    '''
    codes = {}
    codes['object_id'] = llm_clean['object_id']
    val_set['purchase_order_number'] = val_set['purchase_order_number'].strip().replace(' ','')
    if val_set['purchase_order_number'] == llm_clean['purchase_order_number']:
        codes['PO'] = True
    else:
        codes['PO'] = False
    
    # banking
    val_set['bank_name'] = val_set['bank_name'].lower().strip().replace(' ','')
    if iu.get_match(llm_clean['bank_name'], val_set['bank_name']):
        codes['bank_name'] = True
    else:
        codes['bank_name'] = False
    
    val_set['bank_branch_code'] = val_set['bank_branch_code'].strip().replace(' ','')
    if val_set['bank_branch_code'] == llm_clean['bank_branch_code']:
        codes['bank_branch_code'] = True
    else:
        codes['bank_branch_code'] = False
    
    val_set['bank_account_number'] = val_set['bank_account_number'].strip().replace(' ','')
    if val_set['bank_account_number'] == llm_clean['bank_account_number']:
        codes['bank_account_number'] = True
    else:
        codes['bank_account_number'] = False
    
    val_set['vendor_name'] = val_set['vendor_name'].lower().strip().replace(' ','')
    if iu.get_match(llm_clean['vendor_name'], val_set['vendor_name']):
        codes['vendor_name'] = True
    else:
        codes['vendor_name'] = False

    if val_set['invoice_value'] == llm_clean['total_amount']:
        codes['total_amount'] = True
    else:
        codes['total_amount'] = False

    return codes

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    df = validation_set()