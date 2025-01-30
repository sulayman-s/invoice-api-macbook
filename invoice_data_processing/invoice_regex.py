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
