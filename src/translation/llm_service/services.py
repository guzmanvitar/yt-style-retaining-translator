"""LLM Service Abstraction.

This module provides an abstraction layer for interacting with various language models (LLMs).
It allows flexibility in switching between different models.
"""

import json
import os
import pathlib
from abc import ABC, abstractmethod

import torch
import yaml
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import LLM_SERVICE_CONFIG, SECRETS


class LLMService(ABC):
    """
    Abstract base class for language model services.

    Args:
        model (str): Model version used for response generation.
        initial_prompt (str): Initial instructions for the service.
        conversation_history (List[Dict[str, str]]): Conversation context if exists.
    """

    def __init__(
        self,
        model: str,
        initial_prompt: str | None,
        conversation_history: list[dict[str, str]] | None = None,
    ):
        self._model = model
        self.initial_prompt = initial_prompt
        if conversation_history:
            self.conversation_history = conversation_history
        else:
            self.conversation_history = []

    @property
    def model(self):
        """Model name is inmutable."""
        return self._model

    @abstractmethod
    def chat_completion(self) -> str:
        """
        Generates a response from the language model continuing the chat history.

        Returns:
            str: The model-generated response.
        """

    @abstractmethod
    def generate_one_off_response(self, system_prompt: str, user_input: str) -> str:
        """
        Generates a response from the language model without affecting chat continuity.

        Returns:
            str: The model-generated response.
        """

    @abstractmethod
    def generate_formatted_response(self, formatted_prompt: str) -> str:
        """
        Generates a response from a pre-formatted prompt.

        Returns:
            str: The model-generated response.
        """


class OpenAIService(LLMService):
    """
    OpenAI GPT-based language model service.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        initial_prompt: str | None = None,
        temperature: float | None = 0.7,
        embedding_version: str = "text-embedding-ada-002",
    ):
        super().__init__(model, initial_prompt)
        self.temperature = temperature

        # Try fetching API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")

        # Fallback to .secrets file if no environment variable is set
        if not api_key:
            try:
                secrets_path = SECRETS / "openai-creds.json"
                with open(secrets_path, encoding="utf-8") as f:
                    api_key = json.load(f).get("key")
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Error loading OpenAI credentials: {e}") from e

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = OpenAIEmbeddings(
            model=embedding_version, openai_api_key=api_key
        )

    def chat_completion(self):
        """Generates a response using the current conversation history."""
        messages = [
            {"role": "system", "content": self.initial_prompt}
        ] + self.conversation_history

        # Get response from current history
        response = (
            self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
            .choices[0]
            .message.content
        )

        return response

    def generate_one_off_response(self, system_prompt: str, user_input: str):
        """Generates a response without modifying conversation history."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        response = (
            self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
            .choices[0]
            .message.content
        )

        return response

    def generate_formatted_response(self, formatted_prompt: str):
        """Generates a response from a pre-formatted prompt."""
        messages = [{"role": "user", "content": formatted_prompt}]

        response = (
            self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
            .choices[0]
            .message.content
        )

        return response


class SampleService(LLMService):
    """
    Hello world model service for testing conection with the front end.
    """

    def __init__(self, model: str = "sample", initial_prompt: str | None = None):
        super().__init__(model, initial_prompt)

    def chat_completion(self):

        return f"(Local AI) You said: {self.conversation_history[-1]['content']}"

    def generate_one_off_response(self, system_prompt: str, user_input: str):
        return f"(Local AI) System was prompted {system_prompt}, user message was {user_input}"

    def generate_formatted_response(self, formatted_prompt: str):
        """Generates a response from a pre-formatted prompt."""
        return f"(Local AI) Prompt was: '{formatted_prompt}'"


class LLMServiceFactory:
    """
    Factory class to dynamically LLM services based on backend and specific service configuration.

    Args:
        llm_backend (str): Defines base model to use with its configurations.
        service_type (str): Defines type of service called. E.g. dungeon-master, character-creator.
        config_path (pathlib.Path): Path to LLM services config file.
    """

    def __init__(
        self,
        llm_backend: str,
        service_type: str,
        config_path: pathlib.Path = LLM_SERVICE_CONFIG,
    ):
        self.llm_backend = llm_backend
        self.service_type = service_type

        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
            self.backend_config = config["backends"][self.llm_backend]
            self.service_config = config["services"][self.service_type]

    def get_service(self) -> LLMService:
        """
        Returns an instance of the selected LLM service.

        Returns:
            LLMService: An instance of the chosen LLM service.
        """
        initial_prompt = self.service_config.get("initial_prompt", None)

        if self.llm_backend.startswith("gpt"):
            openai_service = OpenAIService(
                model=self.backend_config["model"],
                temperature=self.backend_config["temperature"],
                initial_prompt=initial_prompt,
            )
            return openai_service

        elif self.llm_backend == "samplev1":
            sample_service = SampleService(
                model=self.backend_config["model"],
                initial_prompt=initial_prompt,
            )
            return sample_service
        else:
            raise ValueError(f"Unsupported backend: {self.llm_backend}")
