#!/usr/bin/env python3

import logging

# Reset logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create and configure logger
logging.basicConfig(filename = "/home/hargy/Science/Projects/sci-muons_gesher_v2/src/log/log.log",
                    format   = "%(asctime)s %(message)s",
                    filemode = "w"
    )

# Create logging object
logger = logging.getLogger()

# Setting threshold of logger to DEBUG
logger.setLevel(logging.ERROR)

# Test message
"""
logger.debug("Harmless debug Message")
logger.info("Just an information")
logger.warning("Its a Warning")
logger.error("Did you try to divide by zero")
logger.critical("Internet is down")
"""