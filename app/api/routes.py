import logging
import sys
from fastapi import APIRouter

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

router = APIRouter()
