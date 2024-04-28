import base64
import os
import time
from datetime import datetime

import fire
import ollama
from halo import Halo
from openai import OpenAI
from pydantic import BaseModel
from recorder import get_active_monitor, screenshot
from utils import get_active_window_name, send_notification

# TODOs:
# - Ask which monitor should be recording. This is important because the user might have multiple monitors, and not all apps allow to figure out which monitor they are on.
# - Ask for the goal of the day
# - Figure out how to sort out the API KEY calls
# - "What do you currently want me to do?": have user input at the various stages of the process
# - "hard mode" where notification can't be removed
# - Update README
# - Add images from each screen when in multiple monitor mode
# - Maybe break down high res images into smaller chunks?
# - Handle folder to save to better
# - Get llava to be more reliable in its output format
# - Try larger llava models
# - To reduce false positives, could try to get the model to compare current description with previous one?

OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY="))

SUPPORTED_MODELS = {"llava": "llava", "gpt4": "gpt-4-vision-preview"}


class Activity(BaseModel):
    datetime: datetime
    application: str
    activity: str
    image_path: str
    model: str
    prompt: str
    goal: str
    is_productive: bool
    user_msg: str
    iteration_duration: float = float("inf")


def prompt_for_coach(goal):
    return f"""You are a productivity coach. You are helping me accomplish my goal for today. If I'm not working on something related to my goal, you should label my activity as not productive.

The image you have access to is my computer screen. Let me know if you think my current activity is in line with my goals. Below are example responses, so you know what format to use.
If you leave a message for the user, this will pop up as a notification on their screen, so be mindful of interrupting them if they are already productive.

Response:
- Productive: True
- Description: From the screen, it can be seen that the user is writing code for a project called "ABC", which is in line with their goal to "Finish project ABC".
- Message for user: None

Response:
- Productive: False
- Description: From the screen, it can be seen that the user is scrolling social media, which is not productive because it has nothing to do with accomplishing their goal.
- Message for user: You are scrolling social media, which is not productive because it has nothing to do with accomplishing your goal. You should really be working on your goal to "Finish project ABC" instead.

CURRENT GOAL: {goal}

Response:"""


def run_coach(image_path, model, prompt):
    spinner = Halo(text=f"👀 Running {model}...", spinner="dots")
    spinner.start()
    with open(image_path, "rb") as file:
        image_data = file.read()
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        image_uri = f"data:image/jpeg;base64,{encoded_image}"

        if "gpt4" in model:
            response = OPENAI_CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_uri,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

            response_msg = response.choices[0].message.content
            response_msg_split = response_msg.split("\n")[1:]
            assert len(response_msg_split) == 3, response_msg_split
            productive_unparsed = response_msg_split[0].split(": ")[1]
            assert productive_unparsed in ["True", "False"], productive_unparsed
            productive = True if productive_unparsed == "True" else False
            description = response_msg_split[1].split(": ")[1]
            user_msg = response_msg_split[2].split(": ")[1]

        elif "llava" in model:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data],
                    },
                ],
            )
            response_msg = response["message"]["content"]
            assert all(x in response_msg for x in ["Productive:", "Description:", "Message for user:"]), response_msg
            pre_msg, user_msg = response_msg.split("Message for user:")
            productive_unparsed, description = pre_msg.split("Description:")
            assert ("True" in productive_unparsed) or ("False" in productive_unparsed), productive_unparsed
            productive = "True" in productive_unparsed
        else:
            raise ValueError(f"Model {model} not supported")

    spinner.stop()
    return productive, description, user_msg


def run_llava(image_path, prompt, model="llava"):
    spinner = Halo(text=f"👀 Running Llava ({model})...", spinner="dots")
    spinner.start()
    print(f"🦙 Running {model}")
    with open(image_path, "rb") as file:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [file.read()],
                },
            ],
        )
    result = response["message"]["content"]
    spinner.stop()
    return result


def main(goal: str = "test", model: str = "llava", hard_mode: bool = False):
    assert (
        model in SUPPORTED_MODELS.keys()
    ), f"Supported models are only: {SUPPORTED_MODELS.keys()}.\n You specified: {model}"
    model = SUPPORTED_MODELS[model]

    print("🎯 Your goal is to ", goal)
    print("💪 HARD MODE ACTIVE" if hard_mode else "🐥 Easy mode.")
    print("")

    default_monitor = 0

    while True:
        iteration_start_time = time.time()  # Start timing the iteration

        print("------------------ NEW ITERATION ------------------")

        active_window_name = get_active_window_name()
        active_monitor = get_active_monitor(active_window_name, default_monitor)
        latest_image = screenshot(active_monitor)
        prompt = prompt_for_coach(goal)

        start = time.time()
        productive, description, user_msg = run_coach(latest_image, model, prompt)
        end = time.time()

        print(
            f"👀 Ask coach for opinion on current activity ({end - start:.2f}s)\nProductive: {productive}\nActivity description: {description}\nUser message: {user_msg}\nImage source: {latest_image}\n"
        )

        # create new Activity object
        activity = Activity(
            is_productive=productive,
            activity=description,
            user_msg=user_msg,
            application=active_window_name,
            datetime=datetime.now(),
            image_path=latest_image,
            model=model,
            prompt=prompt,
            goal=goal,
        )

        # Send a notification if the user is not being productive
        if not productive and user_msg is not None:
            send_notification("🛑 PROCRASTINATION ALERT 🛑", user_msg)

        # save the activity to a file
        with open("coach/logs/activities.jsonl", "a") as f:
            f.write(activity.model_dump_json() + "\n")

        iteration_end_time = time.time()  # End timing the iteration
        activity.iteration_duration = iteration_end_time - iteration_start_time
        print(f"\n⏱ Iteration took {activity.iteration_duration:.2f} seconds.\n\n")


if __name__ == "__main__":
    fire.Fire(main)
