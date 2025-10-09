import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", layout_file="layouts/notebook.slides.json")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import os
    import litellm

    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["OPENAI_BASE_URL"] = "http://10.200.206.231:8000/v1"


    resp = litellm.completion(
        model="openai/qwen3",
        messages=[
            {
                "role": "user",
                "content": "Write short complaint to The Boston Globe about the rat problem at Northeastern CS. Blame the math department. No more than 4 sentences.",
            }
        ],
        temperature=0,
    )
    print(resp.choices[0].message.content)
    return (litellm,)


@app.cell
def _():
    from datasets import load_dataset

    flights = load_dataset("nuprl/engineering-llm-systems", name="flights", split="train")
    return (flights,)


@app.cell
def _(flights):
    print(flights[0])
    return


@app.cell
def _():
    import datetime
    from typing import List, Optional, Dict

    class Flight:
        id: int
        date: str
        airline: str
        flight_number: str
        origin: str
        destination: str
        departure_time: str
        arrival_time: str
        available_seats: int

        def __init__(self, flight_number: str, origin: str, destination: str, date: datetime.date, departure_time: datetime.time, arrival_time: datetime.time, available_seats: int):
            self.flight_number = flight_number
            self.origin = origin
            self.destination = destination
            self.date = date
            self.departure_time = departure_time
            self.arrival_time = arrival_time
            self.available_seats = available_seats

        def __str__(self):
            return f"Flight from {self.origin} to {self.destination} on {self.date} departs at {self.departure_time} and arrives at {self.arrival_time}. Available seats: {self.available_seats}"

        def __repr__(self):
            return f"{self.flight_number}: {self.origin}-->{self.destination}, {self.date}@{self.departure_time}"
    return Dict, Flight, List, Optional, datetime


@app.cell
def _(Flight, List, datetime, flights):
    def find_flights(origin: str, destination: str, date: datetime.date) -> List[Flight]:
        """Find flights from origin to destination on a given date."""
        results = []
        for flight in flights:
            if (
                flight["origin"] == origin
                and flight["destination"] == destination
                and flight["date"] == date.isoformat()
            ):
                results.append(
                    Flight(
                        flight_number=flight["flight_number"],
                        origin=flight["origin"],
                        destination=flight["destination"],
                        date=datetime.date.fromisoformat(flight["date"]),
                        departure_time=datetime.time.fromisoformat(flight["departure_time"]),
                        arrival_time=datetime.time.fromisoformat(flight["arrival_time"]),
                        available_seats=flight["available_seats"],
                    )
                )
        print(f"Found {len(results)} flights from {origin} to {destination} on {date}")
        print("\n".join([str(flight) for flight in results]))
    return


@app.cell
def _(flights):
    def can_book(flight_number: str) -> int:
        """Check if a flight can be booked by flight number. If it can return the flight ID, otherwise return -1."""
        for flight in flights:
            if flight["flight_number"] == flight_number and flight["available_seats"] > 0:
                return flight["id"]
        return -1
    return


@app.cell
def _(List, flights):
    booked: List[int] = []

    def book_flight(flight_id: int) -> bool:
        """Book a flight by flight ID. Appends the id to the booked list"""
        for flight in flights:
            if flight["id"] == flight_id:
                if flight["available_seats"] > 0:
                    booked.append(flight_id)
                    return True
                break
        return False
    return (booked,)


@app.cell
def _(List, Optional):
    def update_memory(memory: List[tuple[str, str]], primary_request: Optional[str], user_input: str, response: str, new_primary_request: bool = False):
        """Update the conversation memory which has a maximum of 10 exchanges. 
        If new_primary_request is true, clear the memory and start a new conversation."""
        memory.append((user_input, response))
        if new_primary_request:
            primary_request = user_input
        if len(memory) > 10:
            memory.pop(0)
    return


@app.cell
def _():
    done_messages = []
    def display(message: str):
        """add a message to the done_messages list"""
        done_messages.append(message)
    return (done_messages,)


@app.cell
def _(datetime):
    SYSTEM_PROMPT = """
    You are a flight booking assistant. You can help users find and book flights.
    You can write python code to call the following functions to interact with the database:

    # find_flights
    ### takes in the origin and destination airport codes as well as the date of departure, it returns a list of Flight objects that match the criteria. Will be read in the format [f"{flight.flight_number}: {flight.origin}-->{flight.destination}, {flight.date}@{flight.departure_time}"]
    ```python
    def find_flights(origin: str, destination: str, date: datetime.date) -> List[Flight]:
    ```

    # can_book
    ### takes in a flight number and returns the flight ID if there are available seats, otherwise it returns -1.
    ```python
    def can_book(flight_number: str) -> int:
    ```

    # book_flight
    ### takes in a flight ID it returns true if the flight was successfully booked, otherwise it returns false.

    ```python
    def book_flight(flight_id: int) -> bool:
    ```

    # display
    ### takes in a message string to display to the user when you are done processeing, do not print manually
    ```python
    def display(message: str):
    ```

    # !!!IMPORTANT INSTRUCTIONS!!!
    Don't import anything and don't define helpers.
    Don't simulate data or function calls. I will run your code on my data.

    # Today is
    """ + f" {datetime.date.today().isoformat()}"

    PROMPT_PREFIX = [
        { "role": "system", 
          "content": SYSTEM_PROMPT },
        # Example 1
        { 
            "role": "user",
            "content": "Can I get a flight from SFO to BOS tomorrow or day-after?"
        },
        {
            "role": "assistant",
            "content": """I will use find_flights to first search for a flight tomorrow.

    ```python
    find_flights("SFO", "BOS", datetime.date(2023, 1, 2))
    ```""" },
        { 
            "role": "user", 
            "content": """The result was:

    Id 42: Delta DL111  03:51:00--05:51:00""" 
        },
        { "role": "assistant",
          "content": """There is a Delta flight (id 42) tomorrow. Let's see if it has available seats.

    ```python
    available_seats(42)
    ``` """
        },
        { 
            "role": "user",
            "content": """The result was:

    0"""
        },
        {
            "role": "assistant",
            "content": """No seats on that flight. Let's look for one the day after tomorrow.

    ```python
    find_flights("SFO", "BOS", datetime.date(2023, 1, 3))
    ```""" 
        },
        {
            "role": "user",
            "content": """The result was:

    Id 45: United UA222  06:00:00--08:00:00
    """
        },
        {
            "role": "assistant",
            "content": """There is a United flight (id 45) the day after tomorrow. Let's see if it has available seats.

    ```python
    available_seats(45)
    ```"""
        },
        {
            "role": "user",
            "content": """The result was:

    10"""
        },
        {
            "role": "assistant",
            "content": """Great! There are 10 available seats on that flight. I am done and will let the user
          know.

    ```python
    display("Yes, United 222 from SFO to BOS the day after tomorrow on United with 10 available seats. Unfortunately, nothing is available on the Delta flight tomorrow.")
    ```"""
        }
    ]
    return PROMPT_PREFIX, SYSTEM_PROMPT


@app.cell
def _():
    import re

    def extract_code_from_markdown(markdown_text: str):
        # Look for code blocks in the markdown text
        code_blocks = re.findall(r'```(?:python)?(.*?)```', markdown_text, re.DOTALL)

        # If no code blocks found, return empty string
        if not code_blocks:
            return None

        return "\n".join(code_blocks)
    return (extract_code_from_markdown,)


@app.cell
def _(Dict, List, booked: "List[int]", extract_code_from_markdown, litellm):
    import contextlib
    import io

    def process(messages: List[Dict[str, str]]):
        response = litellm.completion(
            model="openai/qwen3",
            messages=messages,
            temperature=0.2,
        ).choices[0].message.content

        messages.append({"role": "assistant", "content": response})

        code = extract_code_from_markdown(response)
        if code is None:
            return booked


        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            output: str = ""
            try:
                exec(code, globals(), locals())
                output = buf.getvalue()
            except Exception as e:
                output = f"Error running your code:\n{e}"
            finally:
                print("-" * 20)
            messages.append({"role": "system", "content": f"The result was:\n\n{output}"})
    return (process,)


@app.cell
def _(
    Dict,
    List,
    PROMPT_PREFIX,
    SYSTEM_PROMPT,
    booked: "List[int]",
    done_messages,
    process,
):
    def agent():
        booked.clear()
        messages: List[Dict[str, str]] = PROMPT_PREFIX + [{"role": "system", "content": SYSTEM_PROMPT}]

        print("Welcome to the flight booking assistant. How can I help you today?")

        for i in range(20):
            print("User: (blank to quit): ", end="")
            user_input = input()
            if not user_input:
                return booked
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            done_messages.clear()

            first_run = True
            while first_run or len(done_messages) == 0:
                first_run = False
                process(messages)

            if len(done_messages) > 0:
                print("\n".join(done_messages))

    return (agent,)


@app.cell
def _(agent):
    agent()
    return


if __name__ == "__main__":
    app.run()
