from dotenv import load_dotenv

load_dotenv()

# reuse original implementations
from reference.LLMSymbMech.datasets.get_vocab import main_


if __name__ == "__main__":
    main_()