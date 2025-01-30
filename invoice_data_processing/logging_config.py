# logging_config.py
import logging
import logging.config

def setup_logging():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': logging.DEBUG,
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'log/invoice_extract.log',
                'formatter': 'standard',
                'level': logging.DEBUG,
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': logging.DEBUG,
                'propagate': True
            },
        }
    })

setup_logging()
