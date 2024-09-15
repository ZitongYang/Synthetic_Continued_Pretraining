from openai import OpenAI

def gptqa(prompt: str,
          openai_model_name: str,
          system_message: str,
          json_format: bool = False,
          temp: float = 1.0):
    client = OpenAI()
    if json_format:
        completion = client.chat.completions.create(
            model=openai_model_name,
            temperature=temp,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system",
                "content": system_message},
                {"role": "user",
                "content": prompt},
            ])
    else:
        completion = client.chat.completions.create(
            model=openai_model_name,
            temperature=temp,
            messages=[
                {"role": "system",
                "content": system_message},
                {"role": "user",
                "content": prompt},
            ])
    return completion.choices[0].message.content