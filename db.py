from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class EmbeddingModel(Base):
    __tablename__ = 'embedding_model'

    # Unique ID for each row
    id = Column(Integer, primary_key=True, autoincrement=True)
    # Non-nullable string column for the name
    name = Column(String, nullable=False)
    # Nullable string column for embedding
    embedding = Column(String, nullable=True)


# Create a SQLite database connection
engine = create_engine('sqlite:///embedding_model.db', echo=True)

# Create the table in the database
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
