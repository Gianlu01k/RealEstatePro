from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from functools import lru_cache
import re
import geopandas as gpd
import requests
from shapely.geometry import Point
import pandas as pd
import os
from langchain.tools import tool


@lru_cache()
def get_vector_store(index_path):
    """Initialize and return the vector store. Results are cached."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

@tool
def retrieve(query: str, k: int = 3) -> dict:
    """Retrieve relevant real estate listings based on the query.
    If asked for locations, returns latitude and longitude for each listing."""
    vector_store = get_vector_store("faiss_index_dir")
    results = vector_store.similarity_search(query, k=k)
    if not results:
        return {"answer": "No relevant listings found.", "listings": []}
    
    listings = []
    for i, res in enumerate(results, 1):
        listing = {
            "index": i,
            "content": res.page_content,
            "metadata": {   
                "latitude": float(res.metadata.get('latitude', 0)),
                "longitude": float(res.metadata.get('longitude', 0)),
                "address": res.metadata.get('address', ""),
                "property_id": res.metadata.get('property_id', None)
            }
            
        }
        listings.append(listing)
    
    response = "Here are some relevant listings:\n"
    for listing in listings:
        meta = listing["metadata"]
        response += (
            f"{listing['index']}. {listing['content']} | "
            f"Latitude: {meta['latitude']}, Longitude: {meta['longitude']}, "
            f"Address: {meta['address']}, Property ID: {meta['property_id']}\n"
        )
    return {
        "answer": response,
        "listings": listings
    }

@tool
def average_price(location: str) -> dict:
    """Calculate the average price of houses in a given location."""
    vector_store = get_vector_store('faiss_index_dir')
    results = vector_store.similarity_search(f"houses in {location}", k=20)
    if not results:
        return {"answer": f"No listings found for location: {location}", "average": None}
    
    prices = [res.metadata.get('price') for res in results if res.metadata.get('price') is not None]
    if not prices:
        return {"answer": f"No price data available for listings in {location}", "average": None}
    
    avg_price = sum(prices) / len(prices)
    return {
        "answer": f"The average price in {location} is ${avg_price:,.2f}",
        "average": avg_price
    }

@tool
def find_places_near_location(query: str) -> dict:
    """
    Find nearby places using Google Maps APIs.
    Example queries:
    - "Find grocery stores near the first house"
    - "Find subway stations near 40.7484, -73.9857"
    """
    api_key = os.environ.get("POSITION_API_KEY")
    if not api_key:
        return {"error": "Missing POSITION_API_KEY environment variable."}

    match = re.search(r'near\s+(.+)', query, re.IGNORECASE)
    if not match:
        return {"error": "Please specify a location after 'near'."}

    location_from = match.group(1).strip()
    place_type_match = re.search(r'find\s+(.*?)\s+near', query, re.IGNORECASE)
    location_to = place_type_match.group(1).strip() if place_type_match else "point_of_interest"

    coord_match = re.match(r'^\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*$', location_from)
    if coord_match:
        lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
    else:
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        geocode_params = {"address": location_from, "key": api_key}
        geocode_response = requests.get(geocode_url, params=geocode_params).json()

        if not geocode_response.get("results"):
            return {"answer": f"Could not geocode the location: {location_from}"}

        location_coords = geocode_response["results"][0]["geometry"]["location"]
        lat, lon = location_coords["lat"], location_coords["lng"]

    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    places_params = {
        "location": f"{lat},{lon}",
        "radius": 2000,
        "keyword": location_to,
        "key": api_key
    }

    places_response = requests.get(places_url, params=places_params).json()
    results = places_response.get("results", [])

    if not results:
        return {"answer": f"No '{location_to}' found near '{location_from}'."}

    places = []
    for place in results[:5]:
        name = place.get("name")
        addr = place.get("vicinity", place.get("formatted_address", "Address unknown"))
        latlng = place["geometry"]["location"]
        places.append({"name": name, "address": addr, "latlng": latlng})

    answer = f"Top {len(places)} {location_to}(s) near {location_from}:\n" + "\n".join(
        [f"{p['name']} - {p['address']}" for p in places]
    )

    return {
        "answer": answer,
        "places": places,
        "tools_used": ["find_places_near_location"]
    }

@tool
def calculate_travel_time(query: str) -> dict:
    """
    Calculate travel time between two points using Google Distance Matrix API.
    Examples:
    - "travel time from 40.7128,-74.0060 to 40.7138,-74.0010 by walking"
    - "how long by driving from 159 Rivington St to Empire State Building"
    """
    api_key = os.environ.get("POSITION_API_KEY")
    if not api_key:
        return {"error": "Missing POSITION_API_KEY environment variable."}

    origin_match = re.search(r'from (.+?) to', query, re.IGNORECASE)
    destination_match = re.search(r'to (.+?)(?: by|$)', query, re.IGNORECASE)
    mode_match = re.search(r'by (walking|driving|transit|bicycling)', query, re.IGNORECASE)

    if not origin_match or not destination_match:
        return {
            "answer": "Please provide a valid origin and destination.",
            "tools_used": ["calculate_travel_time"]
        }

    origin = origin_match.group(1).strip()
    destination = destination_match.group(1).strip()
    mode = mode_match.group(1).strip() if mode_match else "driving"

    def geocode_if_needed(location: str) -> str | None:
        coord_match = re.match(r'^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$', location)
        if coord_match:
            return location.strip()

        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": location, "key": api_key}
        response = requests.get(geocode_url, params=params)

        if response.status_code != 200:
            return None

        data = response.json()
        results = data.get("results", [])
        if not results:
            return None

        loc = results[0]["geometry"]["location"]
        return f"{loc['lat']},{loc['lng']}"

    origin_coord = geocode_if_needed(origin)
    destination_coord = geocode_if_needed(destination)

    if not origin_coord or not destination_coord:
        return {
            "answer": "Could not geocode origin or destination.",
            "tools_used": ["calculate_travel_time"]
        }

    dist_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin_coord,
        "destinations": destination_coord,
        "mode": mode,
        "key": api_key
    }

    response = requests.get(dist_url, params=params)
    if response.status_code != 200:
        return {"answer": "Failed to reach Google Distance Matrix API."}

    data = response.json()
    if data.get("status") != "OK":
        return {"answer": f"Error from Google API: {data.get('status')}"}

    element = data["rows"][0]["elements"][0]
    if element.get("status") != "OK":
        return {"answer": f"No route found between {origin} and {destination}."}

    distance = element["distance"]["text"]
    duration = element["duration"]["text"]

    answer = f"Travel time from {origin} to {destination} by {mode} is {duration}, covering {distance}."
    return {
        "answer": answer,
        "distance": distance,
        "duration": duration,
        "mode": mode,
        "tools_used": ["calculate_travel_time"]
    }

@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user."""
    return {"answer": answer, "tools_used": tools_used}