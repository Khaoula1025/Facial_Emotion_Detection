from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 

DATABASE_URL = "postgres://postgres:password@localhost/emotiondb"
sessionLocal=sessionmaker(autocommit=False,autoflush=False)
Base=declarative_base()