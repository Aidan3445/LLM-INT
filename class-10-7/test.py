import datasets

"""
FastMCP quickstart example.

cd to the `examples/snippets/clients` directory and run:
    uv run server fastmcp_quickstart stdio
"""

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


flights_dataset = datasets.load_dataset("nuprl/engineering-llm-systems", "flights", split="train")


@mcp.tool()
def find_flights(origin: str, destination: str, date: str):
    """Find flights from origin to destination on a given date. 
    Use airport codes for origin and destination
    """
    results = []
    found_any = False
    
    for flight in flights_dataset:
        if flight["origin"] != origin:
            continue
        if flight["destination"] != destination:
            continue
        if str(flight["date"]) != date:
            continue
        found_any = True
        print(f"Id: {flight['id']}: {flight['airline']} {flight['flight_number']} { flight['departure_time']}--{flight['arrival_time']}")
        results.append(flight)
    if not found_any:
        print("No flights found.")
    return results


@mcp.tool()
def available_seats(id: int):
    """Check available seats for a given flight ID."""
    for flight in flights_dataset:
        if flight["id"] == id:
            print(flight["available_seats"])
            return flight["available_seats"]
    print("Not a valid flight.")


@mcp.tool()
def find_all_flights(origin: str, destination: str):
    """Find all flights from origin to destination. 
    Use airport codes for origin and destination
    """
    results = []
    found_any = False
    
    for flight in flights_dataset:
        if flight["origin"] != origin:
            continue
        if flight["destination"] != destination:
            continue
        found_any = True
        print(f"Id: {flight['id']}: {flight['airline']} {flight['flight_number']} { flight['departure_time']}--{flight['arrival_time']}")
        results.append(flight)
    if not found_any:
        print("No flights found.")
    return results
