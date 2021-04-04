import logging
import os
import datetime

''' 
Logger
'''
def setup_logger():
        """Sets up the logger"""
        date_time = datetime.datetime.now()
        filename = "logs/traininglogs/Training_" + date_time.strftime("%Y-%m-%d_%H:%M:%S") + ".log"
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except: pass

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format 
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

#self.logger.info("time:{0:.10f}".format(time()-start))
logger = setup_logger()
