import os

import backoff
from anthropic import Anthropic
from openai import (
    AzureOpenAI,
    APIConnectionError,
    APIError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)
import requests
from together import Together
from groq import Groq

class LMMEngine:
    pass


class LMMEngineFriendli(LMMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("FRIENDLI_TOKEN")
        if api_key is None:
            raise ValueError(
                "A Friendli token needs to be provided in either the api_key parameter or as an environment variable named FRIENDLI_TOKEN"
            )

        self.base_url = base_url or "https://api.friendli.ai/dedicated/v1"
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        import logging
        logger = logging.getLogger("desktopenv.agent")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            
            content = response.choices[0].message.content
            
            # Only log if there's an issue
            if not content or content.strip() == "":
                logger.error(f"Empty response from Friendli model {self.model}")
                
            return content
            
        except Exception as e:
            logger.error(f"Friendli API call failed: {type(e).__name__}: {str(e)}")
            raise


class LMMEngineHyperbolic(LMMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model
        
        api_key = api_key or os.getenv("HYPERBOLIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Hyperbolic API key needs to be provided in either the api_key parameter or as an environment variable named HYPERBOLIC_API_KEY"
            )

        self.api_key = api_key
        self.base_url = base_url or "https://api.hyperbolic.xyz/v1/chat/completions"
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    @backoff.on_exception(
        backoff.expo, (requests.exceptions.RequestException,), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens if max_new_tokens else 4096,
            **kwargs
        }

        try:
            print(f"🔍 Making Hyperbolic request to: {self.base_url}")
            print(f"🔍 Model: {self.model}")
            print(f"🔍 Message count: {len(messages)}")
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
            
            print(f"🔍 Response status code: {response.status_code}")
            
            response.raise_for_status()
            
            response_data = response.json()
            print(f"🔍 Response keys: {list(response_data.keys())}")
            
            if 'choices' not in response_data or not response_data['choices']:
                print(f"❌ Error: No choices in response: {response_data}")
                raise ValueError(f"No choices in Hyperbolic response: {response_data}")
            
            content = response_data['choices'][0]['message']['content']
            print(f"🔍 Content length: {len(content) if content else 0}")
            print(f"🔍 Content preview: {content[:200] if content else 'EMPTY'}")
            
            if not content:
                print(f"❌ Warning: Empty content from Hyperbolic. Full response: {response_data}")
                raise ValueError("Empty content from Hyperbolic API")
                
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Hyperbolic API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"❌ Error response status: {e.response.status_code}")
                print(f"❌ Error response text: {e.response.text}")
            raise
        except (KeyError, IndexError) as e:
            print(f"❌ Hyperbolic response parsing failed: {e}")
            print(f"❌ Response data: {response_data if 'response_data' in locals() else 'No response data'}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"❌ Raw response text: {response.text}")
            raise

class LMMEngineGroq(LMMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Groq token needs to be provided in either the api_key parameter or as an environment variable named GROQ_API_KEY"
            )

        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = Groq(api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )
         
class LMMEngineTogether(LMMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model
        
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Together token needs to be provided in either the api_key parameter or as an environment variable named TOGETHER_API_KEY"
            )

        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = Together(api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        import logging
        logger = logging.getLogger("desktopenv.agent")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            
            content = response.choices[0].message.content
            
            # Only log if there's an issue
            if not content or content.strip() == "":
                logger.error(f"Empty response from {self.model}")
                
            return content
            
        except Exception as e:
            logger.error(f"API call failed: {type(e).__name__}: {str(e)}")
            raise

class LMMEngineOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        organization=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.organization = organization
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature  # Can force temperature to be the same (in the case of o3 requiring temperature to be 1)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )
        organization = self.organization or os.getenv("OPENAI_ORG_ID")
        if not self.llm_client:
            if not self.base_url:
                self.llm_client = OpenAI(api_key=api_key, organization=organization)
            else:
                self.llm_client = OpenAI(
                    base_url=self.base_url, api_key=api_key, organization=organization
                )
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=(
                    temperature if self.temperature is None else self.temperature
                ),
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineAnthropic(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        thinking=False,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        if not self.llm_client:
            self.llm_client = Anthropic(api_key=api_key)
        # Use the instance temperature if not specified in the call
        temp = self.temperature if temperature is None else temperature
        if self.thinking:
            full_response = self.llm_client.messages.create(
                system=messages[0]["content"][0]["text"],
                model=self.model,
                messages=messages[1:],
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **kwargs,
            )
            thoughts = full_response.content[0].thinking
            return full_response.content[1].text
        return (
            self.llm_client.messages.create(
                system=messages[0]["content"][0]["text"],
                model=self.model,
                messages=messages[1:],
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temp,
                **kwargs,
            )
            .content[0]
            .text
        )

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    # Compatible with Claude-3.7 Sonnet thinking mode
    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        """Generate the next message based on previous messages, and keeps the thinking tokens"""

        full_response = self.llm_client.messages.create(
            system=messages[0]["content"][0]["text"],
            model=self.model,
            messages=messages[1:],
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096},
            **kwargs,
        )

        thoughts = full_response.content[0].thinking
        answer = full_response.content[1].text
        full_response = (
            f"<thoughts>\n{thoughts}\n</thoughts>\n\n<answer>\n{answer}\n</answer>\n"
        )
        return full_response


class LMMEngineGemini(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )
        base_url = self.base_url or os.getenv("GEMINI_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named GEMINI_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use the temperature passed to generate, otherwise use the instance's temperature, otherwise default to 0.0
        temp = self.temperature if temperature is None else temperature
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temp,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineOpenRouter(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENROUTER_API_KEY"
            )

        self.base_url = base_url or os.getenv("OPEN_ROUTER_ENDPOINT_URL")
        if self.base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named OPEN_ROUTER_ENDPOINT_URL"
            )

        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.headers = {
            'Authorization': 'Bearer ' + self.api_key,
            'Content-Type': 'application/json',
        }

        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        
        url = self.base_url
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_new_tokens if max_new_tokens else 4096,
            "provider": {
                "allow_fallbacks": True,
                "sort": "latency"
            }
        }

        try:
            print(f"🔍 Making OpenRouter request to: {url}")
            print(f"🔍 Model: {self.model}")
            print(f"🔍 Message count: {len(messages)}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            print(f"🔍 Response status code: {response.status_code}")
            
            response.raise_for_status()
            
            response_data = response.json()
            print(f"🔍 Response keys: {list(response_data.keys())}")
            
            if 'choices' not in response_data or not response_data['choices']:
                print(f"❌ Error: No choices in response: {response_data}")
                raise ValueError(f"No choices in OpenRouter response: {response_data}")
            
            content = response_data['choices'][0]['message']['content']
            print(f"🔍 Content length: {len(content) if content else 0}")
            print(f"🔍 Content preview: {content[:200] if content else 'EMPTY'}")
            
            if not content:
                print(f"❌ Warning: Empty content from OpenRouter. Full response: {response_data}")
                raise ValueError("Empty content from OpenRouter API")
                
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"❌ OpenRouter API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"❌ Error response status: {e.response.status_code}")
                print(f"❌ Error response text: {e.response.text}")
            raise
        except (KeyError, IndexError) as e:
            print(f"❌ OpenRouter response parsing failed: {e}")
            print(f"❌ Response data: {response_data if 'response_data' in locals() else 'No response data'}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"❌ Raw response text: {response.text}")
            raise


class LMMEngineAzureOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        azure_endpoint=None,
        model=None,
        api_version=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_version = api_version
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.cost = 0.0
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )
        api_version = self.api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "api_version must be provided either as a parameter or as an environment variable named OPENAI_API_VERSION"
            )
        azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError(
                "An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_ENDPOINT"
            )
        if not self.llm_client:
            self.llm_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **kwargs,
        )
        total_tokens = completion.usage.total_tokens
        self.cost += 0.02 * ((total_tokens + 500) / 1000)
        return completion.choices[0].message.content


class LMMEnginevLLM(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(
        self,
        messages,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_new_tokens=512,
        **kwargs,
    ):
        api_key = self.api_key or os.getenv("vLLM_API_KEY")
        if api_key is None:
            raise ValueError(
                "A vLLM API key needs to be provided in either the api_key parameter or as an environment variable named vLLM_API_KEY"
            )
        base_url = self.base_url or os.getenv("vLLM_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        return completion.choices[0].message.content


class LMMEngineHuggingFace(LMMEngine):
    def __init__(self, base_url=None, api_key=None, rate_limit=-1, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("HF_TOKEN")
        if api_key is None:
            raise ValueError(
                "A HuggingFace token needs to be provided in either the api_key parameter or as an environment variable named HF_TOKEN"
            )
        base_url = self.base_url or os.getenv("HF_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "HuggingFace endpoint must be provided as base_url parameter or as an environment variable named HF_ENDPOINT_URL."
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        return (
            self.llm_client.chat.completions.create(
                model="tgi",
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineParasail(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "Parasail model id must be provided"
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("PARASAIL_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Parasail API key needs to be provided in either the api_key parameter or as an environment variable named PARASAIL_API_KEY"
            )
        base_url = self.base_url
        if base_url is None:
            raise ValueError(
                "Parasail endpoint must be provided as base_url parameter or as an environment variable named PARASAIL_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url=base_url if base_url else "https://api.parasail.io/v1",
                api_key=api_key,
            )
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )
