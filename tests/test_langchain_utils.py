import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from langchain_utils import setup_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Mock environment variable for OPENAI_API_KEY
@pytest.fixture(scope="module", autouse=True)
def set_env_vars():
    os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
    yield
    del os.environ["OPENAI_API_KEY"]


def test_setup_chain():
    chain = setup_chain()

    # Verify the components of the chain
    assert hasattr(chain, 'first')
    assert hasattr(chain, 'middle')
    assert hasattr(chain, 'last')

    # Verify the types of the chain components
    assert isinstance(chain.first, ChatPromptTemplate)
    assert isinstance(chain.first.messages[0], SystemMessagePromptTemplate)
    assert isinstance(chain.first.messages[1], HumanMessagePromptTemplate)
    assert isinstance(chain.middle[0], ChatOpenAI)
    assert isinstance(chain.last, StrOutputParser)

    # Check if the chain is callable and runs without error
    inputs = {
        "context": "This is some context.",
        "question": "What is the main content?"
    }
    try:
        result = chain.invoke(inputs)
        assert result is not None
    except Exception as e:
        pytest.fail(f"Chain invocation failed with exception: {e}")
