system_prompt = """
You are a JSON repair assistant.

You will receive a string that is intended to be a JSON object, but it may be malformed due to issues like:
- missing double quotes around keys or string values,
- misplaced or missing commas,
- unescaped characters,
- unmatched brackets or braces.

Your task is to repair the string and return a **valid JSON** object that preserves the original structure and content as much as possible.

**Output only the corrected JSON**, no explanations or comments.
"""

original_str1 = """
{"thought": To determine which language shows the highest overall frequency of the greeting phrase \'Good Morning\', we need to analyze the bar chart provided.\n\n1. **Identify the \'Good Morning\' Bars**: The \'Good Morning\' phrase is represented by the orange bars in the chart.\n\n2. **Examine Each Language**:\n   - **English**: The frequency is 25 times per day.\n   - **Spanish**: The frequency is 30 times per day.\n   - **French**: The frequency is 25 times per day.\n   - **German**: The frequency is 25 times per day.\n   - **Japanese**: The frequency is 20 times per day.\n\n3. **Compare the Frequencies**: \n   - Spanish has the highest frequency of \'Good Morning\' usage at 30 times per day.\n\n4. **Conclusion**: Based on the comparison, Spanish shows the highest overall frequency for the greeting phrase \'Good Morning\'., "actions": [{"name": "Terminate", "arguments": {"ans":  Spanish.}]}]}'}],
"""

correct_str1 = """
{"thought": "To determine which language shows the highest overall frequency of the greeting phrase \'Good Morning\', we need to analyze the bar chart provided.\n\n1. **Identify the \'Good Morning\' Bars**: The \'Good Morning\' phrase is represented by the orange bars in the chart.\n\n2. **Examine Each Language**:\n   - **English**: The frequency is 25 times per day.\n   - **Spanish**: The frequency is 30 times per day.\n   - **French**: The frequency is 25 times per day.\n   - **German**: The frequency is 25 times per day.\n   - **Japanese**: The frequency is 20 times per day.\n\n3. **Compare the Frequencies**: \n   - Spanish has the highest frequency of \'Good Morning\' usage at 30 times per day.\n\n4. **Conclusion**: Based on the comparison, Spanish shows the highest overall frequency for the greeting phrase \'Good Morning\'.", "actions": [{"name": "Terminate", "arguments": {"ans":  "Spanish."}}]}
"""

original_str2 = """
{\"thought\": The highest approval rating of Postgrad is around 68. The lowest approval rating of HS or less is 47. The difference between these two numbers is 68 - 47 = 21., \"actions\": [{\"name\": \"Terminate\", \"arguments\": {\"ans\": 21}]}]}
"""

correct_str2 = """
{\"thought\": \"The highest approval rating of Postgrad is around 68. The lowest approval rating of HS or less is 47. The difference between these two numbers is 68 - 47 = 21.\", \"actions\": [{\"name\": \"Terminate\", \"arguments\": {\"ans\": 21}}]}
"""

PROMPT_DICT = {
    "system": system_prompt,
    "fewshot": [(original_str1, correct_str1), (original_str2, correct_str2)],
}